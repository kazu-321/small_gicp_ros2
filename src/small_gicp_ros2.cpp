#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/factors/general_factor.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include "small_gicp/pcl/pcl_registration.hpp"

#include <autoware_internal_debug_msgs/msg/int32_stamped.hpp>
#include <autoware_internal_debug_msgs/msg/float32_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include "visualization_msgs/msg/marker_array.hpp"

// #include <angles/angles.h>

using namespace small_gicp;

class SmallGICPNode : public rclcpp::Node {
public:
  SmallGICPNode() : Node("small_gicp_node") {
    // Publishers
    pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("small_gicp/pose", 10);
    pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("small_gicp/pointcloud", 10);
    inlier_pub_ = this->create_publisher<autoware_internal_debug_msgs::msg::Int32Stamped>("small_gicp/inlier_count", 10);
    iter_pub_ = this->create_publisher<autoware_internal_debug_msgs::msg::Int32Stamped>("small_gicp/iteration_count", 10);
    error_pub_ = this->create_publisher<autoware_internal_debug_msgs::msg::Float32Stamped>("small_gicp/error", 10);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("small_gicp/markers", 10);

    // std::string PCD_FILE = this->declare_parameter("pcd_file", "/home/kazusahashimoto/ros2_ws/shinagawa_odaiba/map.pcd");
    double voxel_resolution = this->declare_parameter("voxel_resolution", 2.0);
    int num_threads = this->declare_parameter("num_threads", 4);
    int max_iterations = this->declare_parameter("max_iterations", 50);
    double transformation_epsilon = this->declare_parameter("transformation_epsilon", 0.0001);

    small_gicp = RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ>::Ptr(new RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ>());

    small_gicp->setVoxelResolution(voxel_resolution);
    small_gicp->setNumThreads(num_threads);
    small_gicp->setMaximumIterations(max_iterations);
    small_gicp->setTransformationEpsilon(transformation_epsilon);
    // small_gicp->setRotationEpsilon(0.0);
    small_gicp->setRegistrationType("VGICP");
    // small_gicp->setVerbosity(true);

    source_pointcloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/localization/util/downsample/pointcloud",
      rclcpp::SensorDataQoS(),
      std::bind(&SmallGICPNode::pointcloud_callback, this, std::placeholders::_1));

    target_pointcloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/localization/pose_estimator/debug/loaded_pointcloud_map",
      1,
      [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::fromROSMsg(*msg, pcl_cloud);
        small_gicp->setInputTarget(std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(pcl_cloud));
        RCLCPP_INFO(this->get_logger(), "Updated target pointcloud with %zu points", pcl_cloud.size());
        if (!initial_pose_received_) {
          small_gicp->setInputSource(std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(pcl_cloud));
          pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
          initial_pose_ = Eigen::Matrix4f::Identity();
          initial_pose_(0, 3) = pcl_cloud.points[0].x;
          initial_pose_(1, 3) = pcl_cloud.points[0].y;
          initial_pose_(2, 3) = pcl_cloud.points[0].z;
          small_gicp->align(*aligned, initial_pose_);  // 初回の重い処理を回避するためにダミーで実行
        }
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
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for initial pose...");
      return;
    }

    Eigen::Quaternionf q(initial_pose_.block<3, 3>(0, 0));
    if (!std::isfinite(q.w()) || !std::isfinite(q.x())) {
      RCLCPP_WARN(this->get_logger(), "Invalid quaternion");
      return;
    }

    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);

    small_gicp->setInputSource(std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(pcl_cloud));
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
    small_gicp->align(*aligned, initial_pose_);

    auto result = small_gicp->getRegistrationResult();
    Eigen::Matrix4f result_small_gicp = small_gicp->getFinalTransformation();

    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = this->now();
    pose_msg.header.frame_id = "map";
    pose_msg.pose.pose.position.x = result_small_gicp(0, 3);
    pose_msg.pose.pose.position.y = result_small_gicp(1, 3);
    pose_msg.pose.pose.position.z = result_small_gicp(2, 3);
    Eigen::Quaternionf q_out(result_small_gicp.block<3, 3>(0, 0));
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

      double error = result.error_history[marker.id];
      double error_ratio = (error - min_error) / (max_error - min_error + 1e-5);
      marker.color.a = 1.0;
      marker.color.r = 1.0 - error_ratio;
      marker.color.g = 0.0;
      marker.color.b = error_ratio;
      marker.lifetime = rclcpp::Duration::from_seconds(0.1);
      marker_array.markers.push_back(marker);
      line_marker.points.push_back(marker.pose.position);
    }
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

    RCLCPP_INFO(this->get_logger(), "Small GICP inliers: %d, iterations: %d, error: %f", result.num_inliers, result.iterations, result.error);
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

  RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ>::Ptr small_gicp;
  // initialize target_cloud_ to avoid dereferencing a null pointer later
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_{new pcl::PointCloud<pcl::PointXYZ>()};

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
