#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>

#include <autoware_internal_debug_msgs/msg/int32_stamped.hpp>
#include <autoware_internal_debug_msgs/msg/float32_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include "visualization_msgs/msg/marker_array.hpp"

using namespace small_gicp;

class SmallGICPNode : public rclcpp::Node {
public:
  SmallGICPNode() : Node("small_gicp_node") {
    voxel_resolution = this->declare_parameter("voxel_resolution", 1.0);
    thread_count = this->declare_parameter("num_threads", 4);
    max_iteration_count = this->declare_parameter("max_iterations", 50);
    transformation_epsilon = this->declare_parameter("transformation_epsilon", 0.0001);
    neighbor_count = this->declare_parameter("num_neighbors", 20);
    downsampling_resolution = this->declare_parameter("downsampling_resolution", 0.25);
    max_correspondence_distance = this->declare_parameter("max_correspondence_distance", 1.0);

    // Initialize GICP parameters
    gicp_registration.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;
    gicp_registration.reduction.num_threads = thread_count;
    gicp_registration.optimizer.max_iterations = max_iteration_count;
    gicp_registration.criteria.translation_eps = transformation_epsilon;

    // Publishers
    pose_publisher = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("small_gicp/pose", 10);
    pointcloud_publisher = this->create_publisher<sensor_msgs::msg::PointCloud2>("small_gicp/pointcloud", 10);
    inlier_publisher = this->create_publisher<autoware_internal_debug_msgs::msg::Int32Stamped>("small_gicp/inlier_count", 10);
    iteration_publisher = this->create_publisher<autoware_internal_debug_msgs::msg::Int32Stamped>("small_gicp/iteration_count", 10);
    error_publisher = this->create_publisher<autoware_internal_debug_msgs::msg::Float32Stamped>("small_gicp/error", 10);
    tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    marker_publisher = this->create_publisher<visualization_msgs::msg::MarkerArray>("small_gicp/markers", 10);

    source_pointcloud_subscriber = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/localization/util/downsample/pointcloud",
      rclcpp::SensorDataQoS(),
      std::bind(&SmallGICPNode::pointcloud_callback, this, std::placeholders::_1));
    target_pointcloud_subscriber = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/localization/pose_estimator/debug/loaded_pointcloud_map_raw",
      1,
      std::bind(&SmallGICPNode::map_callback, this, std::placeholders::_1));

    pose_subscriber = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/localization/pose_with_covariance",
      rclcpp::SensorDataQoS(),
      std::bind(&SmallGICPNode::pose_callback, this, std::placeholders::_1));
  }

private:
  void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose_msg) {
    initial_pose_received = true;
    initial_pose = Eigen::Matrix4f::Identity();
    initial_pose(0, 3) = pose_msg->pose.pose.position.x;
    initial_pose(1, 3) = pose_msg->pose.pose.position.y;
    initial_pose(2, 3) = pose_msg->pose.pose.position.z;
    Eigen::Quaternionf initial_orientation(pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x, pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z);
    initial_pose.block<3, 3>(0, 0) = initial_orientation.toRotationMatrix();
  }

  void map_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_msg) {
    if (map_processing) {
      RCLCPP_WARN(this->get_logger(), "Map: processing already running, skipping...");
      return;
    }

    map_processing = true;

    // thread join safety
    if (map_thread.joinable()) {
      map_thread.join();
    }

    auto pointcloud_msg_copy = *pointcloud_msg;  // deep copy

    map_thread = std::thread([this, pointcloud_msg_copy]() {
      process_map(pointcloud_msg_copy);
      map_processing = false;
    });
  }

  void process_map(sensor_msgs::msg::PointCloud2 pointcloud_msg) {
    RCLCPP_INFO(this->get_logger(), "MAP: Start map processing in thread");

    auto target_pointcloud_new = std::make_shared<PointCloud>();

    pcl::PointCloud<pcl::PointXYZ> pcl_pointcloud;
    pcl::fromROSMsg(pointcloud_msg, pcl_pointcloud);

    for (const auto& point : pcl_pointcloud.points) {
      target_pointcloud_new->points.push_back(Eigen::Vector4d(point.x, point.y, point.z, 1.0));
    }

    auto target_tree_new = std::make_shared<KdTree<PointCloud>>(target_pointcloud_new, KdTreeBuilderOMP(thread_count));

    estimate_covariances_omp(*target_pointcloud_new, *target_tree_new, neighbor_count, thread_count);

    {
      std::lock_guard<std::mutex> lock(map_mutex);
      target_pointcloud = target_pointcloud_new;
      target_tree = target_tree_new;
    }

    RCLCPP_INFO(this->get_logger(), "MAP: Map processing finished. points=%zu", target_pointcloud_new->points.size());
  }

  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_msg) {
    if (!initial_pose_received) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 100, "Waiting for initial pose...");
      return;
    }

    Eigen::Quaternionf initial_orientation(initial_pose.block<3, 3>(0, 0));
    if (!std::isfinite(initial_orientation.w()) || !std::isfinite(initial_orientation.x())) {
      RCLCPP_WARN(this->get_logger(), "Invalid quaternion");
      return;
    }

    // RCLCPP_INFO(this->get_logger(), "Starting Small GICP alignment...");

    PointCloud::Ptr source_pointcloud(new PointCloud());
    pcl::PointCloud<pcl::PointXYZ> pcl_pointcloud;
    pcl::fromROSMsg(*pointcloud_msg, pcl_pointcloud);
    for (const auto& point : pcl_pointcloud.points) {
      source_pointcloud->points.push_back(Eigen::Vector4d(point.x, point.y, point.z, 1.0));
    }

    // Validate source pointcloud
    if (source_pointcloud->points.empty()) {
      RCLCPP_WARN(this->get_logger(), "Source pointcloud is empty");
      return;
    }

    auto source_tree = std::make_shared<KdTree<PointCloud>>(source_pointcloud, KdTreeBuilderOMP(thread_count));
    if (!source_tree) {
      RCLCPP_ERROR(this->get_logger(), "Failed to create KdTree for source points");
      return;
    }
    estimate_covariances_omp(*source_pointcloud, *source_tree, neighbor_count, thread_count);

    // RCLCPP_INFO(this->get_logger(), "Source pointcloud has %zu points", source_points->points.size());

    Eigen::Isometry3d initial_pose_transform = Eigen::Isometry3d(initial_pose.cast<double>());
    std::shared_ptr<PointCloud> target_pointcloud_local;
    std::shared_ptr<KdTree<PointCloud>> target_tree_local;

    {
      std::lock_guard<std::mutex> lock(map_mutex);
      target_pointcloud_local = target_pointcloud;
      target_tree_local = target_tree;
    }

    if (!target_pointcloud_local || !target_tree_local) {
      RCLCPP_WARN(this->get_logger(), "Target pointcloud not ready");
      return;
    }

    // RCLCPP_INFO(this->get_logger(), "Aligning pointclouds...");
    try {
      // Perform GICP alignment
      auto alignment_result = gicp_registration.align(*target_pointcloud_local, *source_pointcloud, *target_tree_local, initial_pose_transform);
      // RCLCPP_INFO(this->get_logger(), "Alignment complete.");

      // Validate result
      if (!std::isfinite(alignment_result.error) || alignment_result.num_inliers == 0) {
        RCLCPP_WARN(this->get_logger(), "Alignment produced invalid result - error: %f, inliers: %zu", alignment_result.error, alignment_result.num_inliers);
        return;
      }

      const Eigen::Isometry3d& transform_result = alignment_result.T_target_source;

      geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
      pose_msg.header.stamp = this->now();
      pose_msg.header.frame_id = "map";
      pose_msg.pose.pose.position.x = transform_result.translation().x();
      pose_msg.pose.pose.position.y = transform_result.translation().y();
      pose_msg.pose.pose.position.z = transform_result.translation().z();
      Eigen::Quaterniond output_orientation(transform_result.rotation());
      pose_msg.pose.pose.orientation.x = output_orientation.x();
      pose_msg.pose.pose.orientation.y = output_orientation.y();
      pose_msg.pose.pose.orientation.z = output_orientation.z();
      pose_msg.pose.pose.orientation.w = output_orientation.w();

      pose_msg.pose.covariance[0] = 0.0225;
      pose_msg.pose.covariance[7] = 0.0225;
      pose_msg.pose.covariance[14] = 0.0225;
      pose_msg.pose.covariance[21] = 0.000625;
      pose_msg.pose.covariance[28] = 0.000625;
      pose_msg.pose.covariance[35] = 0.000625;
      pose_publisher->publish(pose_msg);

      geometry_msgs::msg::TransformStamped transform_msg;
      transform_msg.header.stamp = this->now();
      transform_msg.header.frame_id = "map";
      transform_msg.child_frame_id = "small_gicp";
      transform_msg.transform.translation.x = pose_msg.pose.pose.position.x;
      transform_msg.transform.translation.y = pose_msg.pose.pose.position.y;
      transform_msg.transform.translation.z = pose_msg.pose.pose.position.z;
      transform_msg.transform.rotation = pose_msg.pose.pose.orientation;
      tf_broadcaster->sendTransform(transform_msg);

      pointcloud_msg->header.stamp = this->now();
      pointcloud_msg->header.frame_id = "small_gicp";
      pointcloud_publisher->publish(*pointcloud_msg);

      autoware_internal_debug_msgs::msg::Int32Stamped inlier_count_msg;
      inlier_count_msg.stamp = this->now();
      inlier_count_msg.data = alignment_result.num_inliers;
      inlier_publisher->publish(inlier_count_msg);

      autoware_internal_debug_msgs::msg::Int32Stamped iteration_count_msg;
      iteration_count_msg.stamp = this->now();
      iteration_count_msg.data = alignment_result.iterations;
      iteration_publisher->publish(iteration_count_msg);

      autoware_internal_debug_msgs::msg::Float32Stamped error_value_msg;
      error_value_msg.stamp = this->now();
      error_value_msg.data = alignment_result.error;
      error_publisher->publish(error_value_msg);

      if (!alignment_result.converged) {
        RCLCPP_WARN(this->get_logger(), "Small GICP did not converge");
      }

      visualization_msgs::msg::MarkerArray marker_array;
      visualization_msgs::msg::Marker line_marker;
      line_marker.header.frame_id = "map";
      line_marker.header.stamp = this->now();
      line_marker.ns = "small_gicp_trajectory_line";
      line_marker.id = 0;
      line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
      line_marker.action = visualization_msgs::msg::Marker::ADD;
      line_marker.scale.x = 0.01;
      line_marker.color.a = 0.8;
      line_marker.color.r = 0.0;
      line_marker.color.g = 1.0;
      line_marker.color.b = 0.0;

      double max_error = 0;
      double min_error = 1e9;

      // Check if history vectors are populated
      if (alignment_result.error_history.empty()) {
        RCLCPP_WARN(this->get_logger(), "Error history is empty, skipping trajectory visualization");
      } else {
        for (const auto& error_value : alignment_result.error_history) {
          if (error_value > max_error) max_error = error_value;
          if (error_value < min_error) min_error = error_value;
        }

        for (const auto& transform : alignment_result.T_target_source_history) {
          visualization_msgs::msg::Marker marker;
          marker.header.frame_id = "map";
          marker.header.stamp = this->now();
          marker.ns = "small_gicp_trajectory";
          marker.id = marker_array.markers.size();
          marker.type = visualization_msgs::msg::Marker::ARROW;
          marker.action = visualization_msgs::msg::Marker::ADD;
          Eigen::Matrix4d transform_matrix = transform.matrix();
          marker.pose.position.x = transform_matrix(0, 3);
          marker.pose.position.y = transform_matrix(1, 3);
          marker.pose.position.z = transform_matrix(2, 3);
          Eigen::Quaterniond marker_orientation(transform_matrix.block<3, 3>(0, 0));
          marker.pose.orientation.x = marker_orientation.x();
          marker.pose.orientation.y = marker_orientation.y();
          marker.pose.orientation.z = marker_orientation.z();
          marker.pose.orientation.w = marker_orientation.w();
          marker.scale.x = 0.02;
          marker.scale.y = 0.01;
          marker.scale.z = 0.01;

          if (marker.id < alignment_result.error_history.size()) {
            double error_value = alignment_result.error_history[marker.id];
            double error_ratio = (error_value - min_error) / (max_error - min_error + 1e-5);
            marker.color.a = 1.0;
            marker.color.r = 1.0 - error_ratio;
            marker.color.g = 0.0;
            marker.color.b = error_ratio;
          } else {
            marker.color.a = 1.0;
            marker.color.r = 0.5;
            marker.color.g = 0.5;
            marker.color.b = 0.5;
          }
          marker.lifetime = rclcpp::Duration::from_seconds(0.1);
          marker_array.markers.push_back(marker);
          line_marker.points.push_back(marker.pose.position);
        }
      }  // Close the else block from empty history check
      line_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
      marker_array.markers.push_back(line_marker);

      visualization_msgs::msg::Marker pose_marker;
      pose_marker.header.frame_id = "map";
      pose_marker.header.stamp = this->now();
      pose_marker.ns = "small_gicp_final_pose";
      pose_marker.id = marker_array.markers.size();
      pose_marker.type = visualization_msgs::msg::Marker::ARROW;
      pose_marker.action = visualization_msgs::msg::Marker::ADD;
      pose_marker.pose.position.x = pose_msg.pose.pose.position.x;
      pose_marker.pose.position.y = pose_msg.pose.pose.position.y;
      pose_marker.pose.position.z = pose_msg.pose.pose.position.z;
      pose_marker.pose.orientation = pose_msg.pose.pose.orientation;
      pose_marker.scale.x = 0.06;
      pose_marker.scale.y = 0.03;
      pose_marker.scale.z = 0.03;
      pose_marker.color.a = 1.0;
      pose_marker.color.r = 0.0;
      pose_marker.color.g = 1.0;
      pose_marker.color.b = 0.0;
      pose_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
      marker_array.markers.push_back(pose_marker);
      marker_publisher->publish(marker_array);

      // RCLCPP_INFO(this->get_logger(), "Small GICP inliers: %zu, iterations: %zu, error: %f", result.num_inliers, result.iterations, result.error);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Small GICP alignment failed with exception: %s", e.what());
      return;
    } catch (...) {
      RCLCPP_ERROR(this->get_logger(), "Small GICP alignment failed with unknown exception");
      return;
    }
  }

  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Int32Stamped>::SharedPtr inlier_publisher;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Int32Stamped>::SharedPtr iteration_publisher;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Float32Stamped>::SharedPtr error_publisher;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr source_pointcloud_subscriber;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr target_pointcloud_subscriber;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_subscriber;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher;

  Registration<GICPFactor, ParallelReductionOMP> gicp_registration;
  PointCloud::Ptr target_pointcloud;
  KdTree<PointCloud>::Ptr target_tree;

  std::mutex map_mutex;
  std::atomic<bool> map_processing{false};
  std::thread map_thread;

  double voxel_resolution;
  int thread_count;
  int max_iteration_count;
  double transformation_epsilon;
  int neighbor_count;
  double downsampling_resolution;
  double max_correspondence_distance;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
  bool initial_pose_received = false;
  Eigen::Matrix4f initial_pose;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SmallGICPNode>());
  rclcpp::shutdown();
  return 0;
}
