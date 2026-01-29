#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

// #include <small_gicp/ann/kdtree_omp.hpp>
// #include <small_gicp/points/point_cloud.hpp>
// #include <small_gicp/factors/gicp_factor.hpp>
// #include <small_gicp/factors/general_factor.hpp>
// #include <small_gicp/util/downsampling_omp.hpp>
// #include <small_gicp/util/normal_estimation_omp.hpp>
// #include <small_gicp/registration/reduction_omp.hpp>
// #include <small_gicp/registration/registration.hpp>
// #include "small_gicp/pcl/pcl_registration.hpp"


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

// #include <angles/angles.h>

using namespace small_gicp;

class SmallGICPNode : public rclcpp::Node {
public:
  SmallGICPNode() : Node("small_gicp_node") {
    voxel_resolution = this->declare_parameter("voxel_resolution", 1.0);
    num_threads = this->declare_parameter("num_threads", 4);
    max_iterations = this->declare_parameter("max_iterations", 50);
    transformation_epsilon = this->declare_parameter("transformation_epsilon", 0.0001);
    num_neighbors = this->declare_parameter("num_neighbors", 20);
    downsampling_resolution = this->declare_parameter("downsampling_resolution", 0.25);
    max_correspondence_distance = this->declare_parameter("max_correspondence_distance", 1.0);

    // Initialize GICP parameters
    small_gicp.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;
    small_gicp.reduction.num_threads = num_threads;
    small_gicp.optimizer.max_iterations = max_iterations;
    small_gicp.criteria.translation_eps = transformation_epsilon;

    // Publishers
    pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("small_gicp/pose", 10);
    pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("small_gicp/pointcloud", 10);
    inlier_pub_ = this->create_publisher<autoware_internal_debug_msgs::msg::Int32Stamped>("small_gicp/inlier_count", 10);
    iter_pub_ = this->create_publisher<autoware_internal_debug_msgs::msg::Int32Stamped>("small_gicp/iteration_count", 10);
    error_pub_ = this->create_publisher<autoware_internal_debug_msgs::msg::Float32Stamped>("small_gicp/error", 10);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("small_gicp/markers", 10);

    source_pointcloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/localization/util/downsample/pointcloud",
      rclcpp::SensorDataQoS(),
      std::bind(&SmallGICPNode::pointcloud_callback, this, std::placeholders::_1));

    target_pointcloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/localization/pose_estimator/debug/loaded_pointcloud_map_raw",
      1,
      [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        target_points = std::make_shared<PointCloud>();
        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::fromROSMsg(*msg, pcl_cloud);
        for (const auto& point : pcl_cloud.points) {
          target_points->points.push_back(Eigen::Vector4d(point.x, point.y, point.z, 1.0));
        }
        auto target_tree = std::make_shared<KdTree<PointCloud>>(target_points, KdTreeBuilderOMP(num_threads));
        estimate_covariances_omp(*target_points, *target_tree, num_neighbors, num_threads);
        
        // Build voxel map for VGICP
        target_voxelmap = std::make_shared<GaussianVoxelMap>(voxel_resolution);
        target_voxelmap->insert(*target_points);

        RCLCPP_INFO(this->get_logger(), "Loaded target pointcloud with %zu points into voxelmap", target_points->points.size());
      });

    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/localization/pose_with_covariance",
      rclcpp::SensorDataQoS(),
      std::bind(&SmallGICPNode::pose_callback, this, std::placeholders::_1));
  }

private:
  void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    initial_pose_received_ = true;
    initial_pose_ = Eigen::Matrix4f::Identity();
    initial_pose_(0, 3) = msg->pose.pose.position.x;
    initial_pose_(1, 3) = msg->pose.pose.position.y;
    initial_pose_(2, 3) = msg->pose.pose.position.z;
    Eigen::Quaternionf q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    initial_pose_.block<3, 3>(0, 0) = q.toRotationMatrix();
  }

  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!initial_pose_received_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 100, "Waiting for initial pose...");
      return;
    }

    Eigen::Quaternionf q(initial_pose_.block<3, 3>(0, 0));
    if (!std::isfinite(q.w()) || !std::isfinite(q.x())) {
      RCLCPP_WARN(this->get_logger(), "Invalid quaternion");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Starting Small GICP alignment...");

    PointCloud::Ptr source_points(new PointCloud());
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);
    for (const auto& point : pcl_cloud.points) {
      source_points->points.push_back(Eigen::Vector4d(point.x, point.y, point.z, 1.0));
    }

    // Validate source pointcloud
    if (source_points->points.empty()) {
      RCLCPP_WARN(this->get_logger(), "Source pointcloud is empty");
      return;
    }

    auto tree = std::make_shared<KdTree<PointCloud>>(source_points, KdTreeBuilderOMP(num_threads));
    if (!tree) {
      RCLCPP_ERROR(this->get_logger(), "Failed to create KdTree for source points");
      return;
    }
    estimate_covariances_omp(*source_points, *tree, num_neighbors, num_threads);

    RCLCPP_INFO(this->get_logger(), "Source pointcloud has %zu points", source_points->points.size());

    Eigen::Isometry3d initial_pose_isometry = Eigen::Isometry3d(initial_pose_.cast<double>());

    if (!target_points || !target_voxelmap) {
      RCLCPP_WARN(this->get_logger(), "Target voxelmap not yet received");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Aligning pointclouds with VGICP...");
    try {
      // Perform VGICP alignment: point cloud to voxel map
      auto result = small_gicp.align(*target_voxelmap, *source_points, *target_voxelmap, initial_pose_isometry);
      RCLCPP_INFO(this->get_logger(), "Alignment complete.");
      
      // Validate result
      if (!std::isfinite(result.error) || result.num_inliers == 0) {
        RCLCPP_WARN(this->get_logger(), "Alignment produced invalid result - error: %f, inliers: %zu", result.error, result.num_inliers);
        return;
      }

    const Eigen::Isometry3d& T_result = result.T_target_source;

    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = this->now();
    pose_msg.header.frame_id = "map";
    pose_msg.pose.pose.position.x = T_result.translation().x();
    pose_msg.pose.pose.position.y = T_result.translation().y();
    pose_msg.pose.pose.position.z = T_result.translation().z();
    Eigen::Quaterniond q_out(T_result.rotation());
    pose_msg.pose.pose.orientation.x = q_out.x();
    pose_msg.pose.pose.orientation.y = q_out.y();
    pose_msg.pose.pose.orientation.z = q_out.z();
    pose_msg.pose.pose.orientation.w = q_out.w();

    pose_msg.pose.covariance[0] = 0.0225;
    pose_msg.pose.covariance[7] = 0.0225;
    pose_msg.pose.covariance[14] = 0.0225;
    pose_msg.pose.covariance[21] = 0.000625;
    pose_msg.pose.covariance[28] = 0.000625;
    pose_msg.pose.covariance[35] = 0.000625;
    pose_publisher_->publish(pose_msg);

    geometry_msgs::msg::TransformStamped transform_msg;
    transform_msg.header.stamp = this->now();
    transform_msg.header.frame_id = "map";
    transform_msg.child_frame_id = "small_gicp";
    transform_msg.transform.translation.x = pose_msg.pose.pose.position.x;
    transform_msg.transform.translation.y = pose_msg.pose.pose.position.y;
    transform_msg.transform.translation.z = pose_msg.pose.pose.position.z;
    transform_msg.transform.rotation = pose_msg.pose.pose.orientation;
    tf_broadcaster_->sendTransform(transform_msg);

    msg->header.stamp = this->now();
    msg->header.frame_id = "small_gicp";
    pointcloud_publisher_->publish(*msg);

    autoware_internal_debug_msgs::msg::Int32Stamped inlier_msg;
    inlier_msg.stamp = this->now();
    inlier_msg.data = result.num_inliers;
    inlier_pub_->publish(inlier_msg);

    autoware_internal_debug_msgs::msg::Int32Stamped iter_msg;
    iter_msg.stamp = this->now();
    iter_msg.data = result.iterations;
    iter_pub_->publish(iter_msg);

    autoware_internal_debug_msgs::msg::Float32Stamped error_msg;
    error_msg.stamp = this->now();
    error_msg.data = result.error;
    error_pub_->publish(error_msg);

    if (!result.converged) {
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
    if (result.error_history.empty()) {
      RCLCPP_WARN(this->get_logger(), "Error history is empty, skipping trajectory visualization");
    } else {
      for (const auto& error : result.error_history) {
        if (error > max_error) max_error = error;
        if (error < min_error) min_error = error;
      }

    for (const auto& T : result.T_target_source_history) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "map";
      marker.header.stamp = this->now();
      marker.ns = "small_gicp_trajectory";
      marker.id = marker_array.markers.size();
      marker.type = visualization_msgs::msg::Marker::ARROW;
      marker.action = visualization_msgs::msg::Marker::ADD;
      Eigen::Matrix4d mat = T.matrix();
      marker.pose.position.x = mat(0, 3);
      marker.pose.position.y = mat(1, 3);
      marker.pose.position.z = mat(2, 3);
      Eigen::Quaterniond q_marker(mat.block<3, 3>(0, 0));
      marker.pose.orientation.x = q_marker.x();
      marker.pose.orientation.y = q_marker.y();
      marker.pose.orientation.z = q_marker.z();
      marker.pose.orientation.w = q_marker.w();
      marker.scale.x = 0.02;
      marker.scale.y = 0.01;
      marker.scale.z = 0.01;

      if (marker.id < result.error_history.size()) {
        double error = result.error_history[marker.id];
        double error_ratio = (error - min_error) / (max_error - min_error + 1e-5);
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
    marker_pub_->publish(marker_array);

    RCLCPP_INFO(this->get_logger(), "Small GICP inliers: %zu, iterations: %zu, error: %f", result.num_inliers, result.iterations, result.error);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Small GICP alignment failed with exception: %s", e.what());
      return;
    } catch (...) {
      RCLCPP_ERROR(this->get_logger(), "Small GICP alignment failed with unknown exception");
      return;
    }
  }

  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Int32Stamped>::SharedPtr inlier_pub_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Int32Stamped>::SharedPtr iter_pub_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Float32Stamped>::SharedPtr error_pub_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr source_pointcloud_subscriber_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr target_pointcloud_subscriber_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  Registration<GICPFactor, ParallelReductionOMP> small_gicp;
  PointCloud::Ptr target_points;
  GaussianVoxelMap::Ptr target_voxelmap;

  double voxel_resolution;
  int num_threads;
  int max_iterations;
  double transformation_epsilon;
  int num_neighbors;
  double downsampling_resolution;
  double max_correspondence_distance;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  bool initial_pose_received_ = false;
  Eigen::Matrix4f initial_pose_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SmallGICPNode>());
  rclcpp::shutdown();
  return 0;
}
