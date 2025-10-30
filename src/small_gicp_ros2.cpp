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

#include <autoware_internal_debug_msgs/msg/int32_stamped.hpp>
#include <autoware_internal_debug_msgs/msg/float32_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

#define TARGET_TOPIC "/localization/util/downsample/pointcloud"
// #define PCD_FILE "/home/kazusahashimoto/autoware_map/sample-map-rosbag/pointcloud_map.pcd"
// #define PCD_FILE "/home/kazusahashimoto/ros2_ws/shinagawa_odaiba/map.pcd"
// #define PCD_FILE "/home/kazusahashimoto/ros2_ws/shinagawa_odaiba/output.pcd"
using namespace small_gicp;

class SmallGICPNode : public rclcpp::Node {
public:
  SmallGICPNode() : Node("small_gicp_node") {
    pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("small_gicp/pose", 10);
    pointcloud_subscriber_ =
      this->create_subscription<sensor_msgs::msg::PointCloud2>(TARGET_TOPIC, rclcpp::SensorDataQoS(), std::bind(&SmallGICPNode::pointcloud_callback, this, std::placeholders::_1));
    pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("small_gicp/pointcloud", 10);
    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/localization/pose_with_covariance",
      rclcpp::SensorDataQoS(),
      [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) { initial_pose_ = msg->pose.pose; });

    inlier_pub_ = this->create_publisher<autoware_internal_debug_msgs::msg::Int32Stamped>("small_gicp/inlier_count", 10);
    iter_pub_ = this->create_publisher<autoware_internal_debug_msgs::msg::Int32Stamped>("small_gicp/iteration_count", 10);
    error_pub_ = this->create_publisher<autoware_internal_debug_msgs::msg::Float32Stamped>("small_gicp/error", 10);
    diff_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("small_gicp/diff_fixed", 10);
    this->declare_parameter("pcd_file", "/home/kazusahashimoto/ros2_ws/shinagawa_odaiba/map.pcd");
    this->declare_parameter("downsampling_resolution", 0.05);
    std::string PCD_FILE = this->get_parameter("pcd_file").as_string();

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(PCD_FILE, *target_cloud_) == -1) {
      RCLCPP_ERROR(this->get_logger(), "Couldn't read source PCD file");
      throw std::runtime_error("Failed to load source PCD file");
    }
    RCLCPP_INFO(this->get_logger(), "Loaded target PCD file successfully");
    // Convert target cloud to vector of Eigen::Vector3f
    for (const auto& point : target_cloud_->points) {
      Eigen::Vector3f vec(point.x, point.y, point.z);
      target_points_.push_back(vec);
    }
    RCLCPP_INFO(this->get_logger(), "Converted target PCD to Eigen::Vector3f");
    num_threads = 4;                                       // Number of threads to be used
    downsampling_resolution = this->get_parameter("downsampling_resolution").as_double(); // m
    num_neighbors = 10;                                    // Number of neighbor points used for normal and covariance estimation
    max_correspondence_distance = 1.0;                     // Maximum correspondence distance between points (e.g., triming threshold)
    init_T_target_source = Eigen::Isometry3d::Identity();  // Initial transformation from target to source
    target = std::make_shared<PointCloud>(target_points_);
    target = voxelgrid_sampling_omp(*target, downsampling_resolution, num_threads);
    target_tree = std::make_shared<KdTree<PointCloud>>(target, KdTreeBuilderOMP(num_threads));
    estimate_covariances_omp(*target, *target_tree, num_neighbors, num_threads);
    RCLCPP_INFO(this->get_logger(), "Estimated point covariances for target cloud");
    this->declare_parameter("max_iterations", 50);
  }

  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    init_T_target_source.translation() = Eigen::Vector3d(initial_pose_.position.x, initial_pose_.position.y, initial_pose_.position.z);
    init_T_target_source.linear() =
      Eigen::Quaterniond(initial_pose_.orientation.w, initial_pose_.orientation.x, initial_pose_.orientation.y, initial_pose_.orientation.z).toRotationMatrix();
    if (init_T_target_source.isApprox(Eigen::Isometry3d::Identity())) {
      RCLCPP_WARN(this->get_logger(), "Initial pose for target source transformation is not set yet");
      return;
    }

    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);

    std::vector<Eigen::Vector3f> source_points;
    for (const auto& pt : pcl_cloud.points) {
      source_points.emplace_back(pt.x, pt.y, pt.z);
    }

    // RCLCPP_INFO(this->get_logger(), "Converted source point cloud to Eigen::Vector3f");

    // Convert to small_gicp::PointCloud
    auto source = std::make_shared<PointCloud>(source_points);
    source = voxelgrid_sampling_omp(*source, downsampling_resolution, num_threads);

    // Create KdTree
    auto source_tree = std::make_shared<KdTree<PointCloud>>(source, KdTreeBuilderOMP(num_threads));

    // RCLCPP_INFO(this->get_logger(), "made target and source point clouds");

    // Estimate point covariances
    estimate_covariances_omp(*source, *source_tree, num_neighbors, num_threads);

    RCLCPP_INFO(this->get_logger(), "Estimated point covariances");

    // GICP + OMP-based parallel reduction
    Registration<GICPFactor, ParallelReductionOMP, RestrictDoFFactor> registration;
    registration.reduction.num_threads = num_threads;
    registration.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;
    registration.criteria.translation_eps = 0.0001;
    this->get_parameter("max_iterations", max_iterations);
    registration.optimizer.max_iterations = max_iterations;

    // Align point clouds

    auto result = registration.align(*target, *source, *target_tree, init_T_target_source);

    RCLCPP_INFO(this->get_logger(), "Aligned point clouds");

    // Publish pose
    // result.H = Final Hessian matrix (6x6)
    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = this->now();
    pose_msg.header.frame_id = "map";
    pose_msg.pose.pose.position.x = result.T_target_source.translation().x();
    pose_msg.pose.pose.position.y = result.T_target_source.translation().y();
    pose_msg.pose.pose.position.z = result.T_target_source.translation().z();
    Eigen::Quaterniond q(result.T_target_source.rotation());
    pose_msg.pose.pose.orientation.x = q.x();
    pose_msg.pose.pose.orientation.y = q.y();
    pose_msg.pose.pose.orientation.z = q.z();
    pose_msg.pose.pose.orientation.w = q.w();

    // covariance matrix
    // [
    //   0.0225, 0.0,   0.0,   0.0,      0.0,      0.0,
    //   0.0,   0.0225, 0.0,   0.0,      0.0,      0.0,
    //   0.0,   0.0,   0.0225, 0.0,      0.0,      0.0,
    //   0.0,   0.0,   0.0,   0.000625, 0.0,      0.0,
    //   0.0,   0.0,   0.0,   0.0,      0.000625, 0.0,
    //   0.0,   0.0,   0.0,   0.0,      0.0,      0.000625,
    // ]

    pose_msg.pose.covariance[0] = 0.0225;  // x position variance
    pose_msg.pose.covariance[7] = 0.0225;  // y position variance
    pose_msg.pose.covariance[14] = 0.0225; // z position variance
    pose_msg.pose.covariance[21] = 0.000625; // x orientation variance
    pose_msg.pose.covariance[28] = 0.000625; // y orientation variance
    pose_msg.pose.covariance[35] = 0.000625; // z orientation variance

    pose_publisher_->publish(pose_msg);

    geometry_msgs::msg::TransformStamped transform_msg;
    transform_msg.header.stamp = this->now();
    transform_msg.header.frame_id = "map";
    transform_msg.child_frame_id = "small_gicp";

    transform_msg.transform.translation.x = pose_msg.pose.pose.position.x;
    transform_msg.transform.translation.y = pose_msg.pose.pose.position.y;
    transform_msg.transform.translation.z = pose_msg.pose.pose.position.z;
    transform_msg.transform.rotation.x = pose_msg.pose.pose.orientation.x;
    transform_msg.transform.rotation.y = pose_msg.pose.pose.orientation.y;
    transform_msg.transform.rotation.z = pose_msg.pose.pose.orientation.z;
    transform_msg.transform.rotation.w = pose_msg.pose.pose.orientation.w;
    tf_broadcaster_->sendTransform(transform_msg);

    // Publish point cloud
    msg->header.stamp = this->now();
    msg->header.frame_id = "small_gicp";
    pointcloud_publisher_->publish(*msg);

    // Publish inlier count
    autoware_internal_debug_msgs::msg::Int32Stamped inlier_msg;
    inlier_msg.stamp = this->now();
    inlier_msg.data = result.num_inliers;
    inlier_pub_->publish(inlier_msg);

    // Publish iteration count
    autoware_internal_debug_msgs::msg::Int32Stamped iter_msg;
    iter_msg.stamp = this->now();
    iter_msg.data = result.iterations;
    iter_pub_->publish(iter_msg);

    // Publish error
    autoware_internal_debug_msgs::msg::Float32Stamped error_msg;
    error_msg.stamp = this->now();
    error_msg.data = result.error;
    error_pub_->publish(error_msg);

    if (!result.converged) {
      RCLCPP_WARN(this->get_logger(), "Small GICP did not converge");
    }

    // Publish difference
    geometry_msgs::msg::TwistStamped diff_msg;
    diff_msg.header.stamp = this->now();
    diff_msg.header.frame_id = "small_gicp";
    double diff_position_x = result.T_target_source.translation().x() - initial_pose_.position.x;
    double diff_position_y = result.T_target_source.translation().y() - initial_pose_.position.y;
    double diff_position_z = result.T_target_source.translation().z() - initial_pose_.position.z;
    double result_yaw = std::atan2(result.T_target_source.rotation().matrix()(1, 0), result.T_target_source.rotation().matrix()(0, 0)) - M_PI;

    // map基準のdiffなのでsmall_gicp用に回転させる
    diff_msg.twist.linear.x = diff_position_x * std::cos(result_yaw) + diff_position_y * std::sin(result_yaw);
    diff_msg.twist.linear.y = -diff_position_x * std::sin(result_yaw) + diff_position_y * std::cos(result_yaw);
    diff_msg.twist.linear.z = diff_position_z;
    diff_pub_->publish(diff_msg);
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Int32Stamped>::SharedPtr inlier_pub_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Int32Stamped>::SharedPtr iter_pub_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Float32Stamped>::SharedPtr error_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr diff_pub_;
  std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> target_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  std::vector<Eigen::Vector3f> target_points_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscriber_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
  Eigen::Isometry3d init_T_target_source;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  int num_threads;
  double downsampling_resolution;
  int num_neighbors;
  double max_correspondence_distance;
  std::shared_ptr<PointCloud> target;
  std::shared_ptr<KdTree<PointCloud>> target_tree;
  geometry_msgs::msg::Pose initial_pose_;
  int max_iterations = 50;  // Maximum iterations for the registration
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SmallGICPNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
