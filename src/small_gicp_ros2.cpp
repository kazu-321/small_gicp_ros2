#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>

#include <autoware_internal_debug_msgs/msg/int32_stamped.hpp>
#include <autoware_internal_debug_msgs/msg/float32_stamped.hpp>

using namespace small_gicp;

class SmallGICPNode : public rclcpp::Node {
public:
  SmallGICPNode() : Node("small_gicp_node") {
    /* Parameters */
    voxel_resolution = declare_parameter("voxel_resolution", 1.0);
    num_threads = declare_parameter("num_threads", 4);
    max_iterations = declare_parameter("max_iterations", 50);
    transformation_epsilon = declare_parameter("transformation_epsilon", 0.0001);
    num_neighbors = declare_parameter("num_neighbors", 20);
    downsampling_resolution = declare_parameter("downsampling_resolution", 0.25);
    max_correspondence_distance = declare_parameter("max_correspondence_distance", 1.0);

    small_gicp.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;
    small_gicp.reduction.num_threads = num_threads;
    small_gicp.optimizer.max_iterations = max_iterations;
    small_gicp.criteria.translation_eps = transformation_epsilon;

    /* Publishers */
    pose_pub = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("small_gicp/pose", 10);

    cloud_pub = create_publisher<sensor_msgs::msg::PointCloud2>("small_gicp/pointcloud", 10);

    inlier_pub = create_publisher<autoware_internal_debug_msgs::msg::Int32Stamped>("small_gicp/inlier_count", 10);

    iter_pub = create_publisher<autoware_internal_debug_msgs::msg::Int32Stamped>("small_gicp/iteration_count", 10);

    error_pub = create_publisher<autoware_internal_debug_msgs::msg::Float32Stamped>("small_gicp/error", 10);

    marker_pub = create_publisher<visualization_msgs::msg::MarkerArray>("small_gicp/markers", 10);

    tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    /* Subscribers */
    src_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/localization/util/downsample/pointcloud",
      rclcpp::SensorDataQoS(),
      std::bind(&SmallGICPNode::pointcloudCallback, this, std::placeholders::_1));

    map_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/localization/pose_estimator/debug/loaded_pointcloud_map_raw",
      1,
      std::bind(&SmallGICPNode::mapCallback, this, std::placeholders::_1));

    pose_sub = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/localization/pose_with_covariance",
      rclcpp::SensorDataQoS(),
      std::bind(&SmallGICPNode::poseCallback, this, std::placeholders::_1));
  }

private:
  /* ---------------- Pose Callback ---------------- */

  void poseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    initial_pose_received = true;

    initial_pose = Eigen::Matrix4f::Identity();
    initial_pose(0, 3) = msg->pose.pose.position.x;
    initial_pose(1, 3) = msg->pose.pose.position.y;
    initial_pose(2, 3) = msg->pose.pose.position.z;

    Eigen::Quaternionf q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);

    initial_pose.block<3, 3>(0, 0) = q.toRotationMatrix();
  }

  /* ---------------- Map Processing ---------------- */

  void mapCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (map_processing) {
      RCLCPP_WARN(get_logger(), "Map processing already running");
      return;
    }

    map_processing = true;

    if (map_thread.joinable()) map_thread.join();

    auto copy = *msg;

    map_thread = std::thread([this, copy]() {
      processMap(copy);
      map_processing = false;
    });
  }

  void processMap(sensor_msgs::msg::PointCloud2 msg) {
    RCLCPP_INFO(get_logger(), "Start map processing");

    auto target = std::make_shared<PointCloud>();

    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(msg, pcl_cloud);

    for (const auto& p : pcl_cloud.points) target->points.emplace_back(p.x, p.y, p.z, 1.0);

    auto tree = std::make_shared<KdTree<PointCloud>>(target, KdTreeBuilderOMP(num_threads));

    estimate_covariances_omp(*target, *tree, num_neighbors, num_threads);

    {
      std::lock_guard<std::mutex> lock(map_mutex);
      target_points = target;
      target_tree = tree;
    }

    RCLCPP_INFO(get_logger(), "Map ready (%zu points)", target->points.size());
  }

  /* ---------------- Scan Matching ---------------- */

  void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!initial_pose_received) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 100, "Waiting initial pose...");
      return;
    }

    /* Convert source cloud */
    auto source = std::make_shared<PointCloud>();
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);

    for (const auto& p : pcl_cloud.points) source->points.emplace_back(p.x, p.y, p.z, 1.0);

    if (source->points.empty()) {
      RCLCPP_WARN(get_logger(), "Empty source cloud");
      return;
    }

    auto tree = std::make_shared<KdTree<PointCloud>>(source, KdTreeBuilderOMP(num_threads));

    estimate_covariances_omp(*source, *tree, num_neighbors, num_threads);

    std::shared_ptr<PointCloud> tgt;
    std::shared_ptr<KdTree<PointCloud>> tgt_tree;

    {
      std::lock_guard<std::mutex> lock(map_mutex);
      tgt = target_points;
      tgt_tree = target_tree;
    }

    if (!tgt || !tgt_tree) {
      RCLCPP_WARN(get_logger(), "Map not ready");
      return;
    }

    try {
      Eigen::Isometry3d init(initial_pose.cast<double>());

      auto result = small_gicp.align(*tgt, *source, *tgt_tree, init);

      if (!std::isfinite(result.error) || result.num_inliers == 0) {
        RCLCPP_WARN(get_logger(), "Invalid alignment result");
        return;
      }

      publishPose(result);
      publishDebug(msg, result);
      publishMarkers(result);

    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Alignment failed: %s", e.what());
    }
  }

  /* ---------------- Publishing ---------------- */

  void publishPose(const RegistrationResult& result) {
    geometry_msgs::msg::PoseWithCovarianceStamped pose;
    pose.header.stamp = now();
    pose.header.frame_id = "map";

    const auto& T = result.T_target_source;
    pose.pose.pose.position.x = T.translation().x();
    pose.pose.pose.position.y = T.translation().y();
    pose.pose.pose.position.z = T.translation().z();

    Eigen::Quaterniond q(T.rotation());
    pose.pose.pose.orientation.x = q.x();
    pose.pose.pose.orientation.y = q.y();
    pose.pose.pose.orientation.z = q.z();
    pose.pose.pose.orientation.w = q.w();

    pose_pub->publish(pose);

    geometry_msgs::msg::TransformStamped tf;
    tf.header = pose.header;
    tf.child_frame_id = "small_gicp";
    tf.transform.translation.x = pose.pose.pose.position.x;
    tf.transform.translation.y = pose.pose.pose.position.y;
    tf.transform.translation.z = pose.pose.pose.position.z;
    tf.transform.rotation = pose.pose.pose.orientation;

    tf_broadcaster->sendTransform(tf);
  }

  void publishDebug(const sensor_msgs::msg::PointCloud2::SharedPtr msg, const RegistrationResult& result) {
    auto cloud = *msg;
    cloud.header.frame_id = "small_gicp";
    cloud.header.stamp = now();
    cloud_pub->publish(cloud);

    autoware_internal_debug_msgs::msg::Int32Stamped inlier;
    inlier.stamp = now();
    inlier.data = result.num_inliers;
    inlier_pub->publish(inlier);

    autoware_internal_debug_msgs::msg::Int32Stamped iter;
    iter.stamp = now();
    iter.data = result.iterations;
    iter_pub->publish(iter);

    autoware_internal_debug_msgs::msg::Float32Stamped err;
    err.stamp = now();
    err.data = result.error;
    error_pub->publish(err);
  }

  void publishMarkers(const RegistrationResult& result) {
    visualization_msgs::msg::MarkerArray array;

    visualization_msgs::msg::Marker pose_marker;
    pose_marker.header.frame_id = "map";
    pose_marker.header.stamp = now();
    pose_marker.ns = "final_pose";
    pose_marker.type = visualization_msgs::msg::Marker::ARROW;
    pose_marker.scale.x = 0.06;
    pose_marker.scale.y = 0.03;
    pose_marker.scale.z = 0.03;
    pose_marker.color.a = 1.0;
    pose_marker.color.g = 1.0;

    const auto& T = result.T_target_source;
    pose_marker.pose.position.x = T.translation().x();
    pose_marker.pose.position.y = T.translation().y();
    pose_marker.pose.position.z = T.translation().z();

    Eigen::Quaterniond q(T.rotation());
    pose_marker.pose.orientation.x = q.x();
    pose_marker.pose.orientation.y = q.y();
    pose_marker.pose.orientation.z = q.z();
    pose_marker.pose.orientation.w = q.w();

    array.markers.push_back(pose_marker);

    marker_pub->publish(array);
  }

  /* ---------------- ROS Members ---------------- */

  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Int32Stamped>::SharedPtr inlier_pub;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Int32Stamped>::SharedPtr iter_pub;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Float32Stamped>::SharedPtr error_pub;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr src_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  Registration<GICPFactor, ParallelReductionOMP> small_gicp;

  PointCloud::Ptr target_points;
  KdTree<PointCloud>::Ptr target_tree;

  std::mutex map_mutex;
  std::thread map_thread;
  std::atomic<bool> map_processing{false};

  Eigen::Matrix4f initial_pose;
  bool initial_pose_received = false;

  double voxel_resolution;
  int num_threads;
  int max_iterations;
  double transformation_epsilon;
  int num_neighbors;
  double downsampling_resolution;
  double max_correspondence_distance;
};

/* ---------------- main ---------------- */

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SmallGICPNode>());
  rclcpp::shutdown();
  return 0;
}
