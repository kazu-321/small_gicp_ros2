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

// #include <angles/angles.h>

using namespace small_gicp;

#define TARGET_TOPIC "/localization/util/downsample/pointcloud"

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

    std::string PCD_FILE = this->declare_parameter("pcd_file", "/home/kazusahashimoto/ros2_ws/shinagawa_odaiba/map.pcd");
    double voxel_resolution = this->declare_parameter("voxel_resolution", 2.0);
    int num_threads = this->declare_parameter("num_threads", 4);
    int max_iterations = this->declare_parameter("max_iterations", 50);
    double transformation_epsilon = this->declare_parameter("transformation_epsilon", 0.0001);

    small_gicp = RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ>::Ptr(new RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ>());

    small_gicp->setVoxelResolution(voxel_resolution);
    small_gicp->setNumThreads(num_threads);
    small_gicp->setMaximumIterations(max_iterations);
    small_gicp->setTransformationEpsilon(transformation_epsilon);
    small_gicp->setRegistrationType("VGICP");

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(PCD_FILE, *target_cloud_) == -1) {
      RCLCPP_ERROR(this->get_logger(), "Couldn't read target PCD file: %s", PCD_FILE.c_str());
      throw std::runtime_error("Failed to load target PCD file");
    }
    RCLCPP_INFO(this->get_logger(), "Loaded target PCD file: %s", PCD_FILE.c_str());

    small_gicp->setInputTarget(target_cloud_);

    pointcloud_subscriber_ =
      this->create_subscription<sensor_msgs::msg::PointCloud2>(TARGET_TOPIC, rclcpp::SensorDataQoS(), std::bind(&SmallGICPNode::pointcloud_callback, this, std::placeholders::_1));

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
  }

  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Int32Stamped>::SharedPtr inlier_pub_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Int32Stamped>::SharedPtr iter_pub_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Float32Stamped>::SharedPtr error_pub_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscriber_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;

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
