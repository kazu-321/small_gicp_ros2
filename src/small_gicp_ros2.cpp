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
#include <geometry_msgs/msg/twist_stamped.hpp>

#define TARGET_TOPIC "/localization/util/downsample/pointcloud"
#define PCD_FILE "/home/kazusahashimoto/ros2_ws/shinagawa_odaiba/map.pcd"
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
      [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        initial_pose_ = msg->pose.pose;
      });
      
      inlier_pub_ = this->create_publisher<autoware_internal_debug_msgs::msg::Int32Stamped>("small_gicp/inlier_count", 10);
      diff_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("small_gicp/diff", 10);
      
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
      downsampling_resolution = 0.5;                        // Downsampling resolution
      num_neighbors = 10;                                    // Number of neighbor points used for normal and covariance estimation
      max_correspondence_distance = 1.0;                     // Maximum correspondence distance between points (e.g., triming threshold)
      init_T_target_source = Eigen::Isometry3d::Identity();  // Initial transformation from target to source
      target = std::make_shared<PointCloud>(target_points_);
      target = voxelgrid_sampling_omp(*target, downsampling_resolution, num_threads);
      target_tree = std::make_shared<KdTree<PointCloud>>(target, KdTreeBuilderOMP(num_threads));
      estimate_covariances_omp(*target, *target_tree, num_neighbors, num_threads);
      RCLCPP_INFO(this->get_logger(), "Estimated point covariances for target cloud");
    }
    
    
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // init_T_target_source Isometry3d
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

    // RCLCPP_INFO(this->get_logger(), "Estimated point covariances");

    // GICP + OMP-based parallel reduction
    Registration<GICPFactor, ParallelReductionOMP, RestrictDoFFactor> registration;
    registration.reduction.num_threads = num_threads;
    registration.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;
    registration.criteria.translation_eps = 1e-5;
    registration.general_factor.set_rotation_mask(Eigen::Vector3d(0, 0, 0.1));  // Set rotation mask to allow only yaw rotation
    registration.general_factor.set_translation_mask(Eigen::Vector3d(0.1, 0.1, 0.1));  // Allow translation in all directions
    registration.optimizer.max_iterations = 40;

    // Align point clouds

    auto result = registration.align(*target, *source, *target_tree, init_T_target_source);

    // RCLCPP_INFO(this->get_logger(), "Aligned point clouds");

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

    // Publish difference
    geometry_msgs::msg::TwistStamped diff_msg;
    diff_msg.header.stamp = this->now();
    diff_msg.header.frame_id = "small_gicp";
    diff_msg.twist.linear.x = result.T_target_source.translation().x() - initial_pose_.position.x;
    diff_msg.twist.linear.y = result.T_target_source.translation().y() - initial_pose_.position.y;
    diff_msg.twist.linear.z = result.T_target_source.translation().z() - initial_pose_.position.z;
    Eigen::Quaterniond initial_q(initial_pose_.orientation.w,
                                 initial_pose_.orientation.x,
                                 initial_pose_.orientation.y,
                                 initial_pose_.orientation.z);
    Eigen::Quaterniond diff_q(result.T_target_source.rotation() * initial_q.inverse());
    diff_msg.twist.angular.z = std::atan2(2.0 * (diff_q.w() * diff_q.z() + diff_q.x() * diff_q.y()), 1.0 - 2.0 * (diff_q.y() * diff_q.y() + diff_q.z() * diff_q.z()));
      diff_msg.twist.angular.x = 0.0;  // Assuming no roll
    diff_msg.twist.angular.y = 0.0;  // Assuming no pitch
    diff_pub_->publish(diff_msg);
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Int32Stamped>::SharedPtr inlier_pub_;
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
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SmallGICPNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
