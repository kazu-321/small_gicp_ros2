#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include <pcl_conversions/pcl_conversions.h> 
#include <pcl/io/pcd_io.h>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>

#define TARGET_TOPIC "/localization/util/downsample/pointcloud"
#define PCD_FILE "/home/kazusahashimoto/ros2_ws/shinagawa_odaiba/map.pcd"
using namespace small_gicp;

class SmallGICPNode : public rclcpp::Node {
public:
  SmallGICPNode() : Node("small_gicp_node") {
    // Subscriber for point cloud data
    pointcloud_subscriber_ =
      this->create_subscription<sensor_msgs::msg::PointCloud2>(TARGET_TOPIC, rclcpp::SensorDataQoS(), std::bind(&SmallGICPNode::pointcloud_callback, this, std::placeholders::_1));
    pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("small_gicp/pose", 10);
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
  }

  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received point cloud message");
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);

    std::vector<Eigen::Vector3f> source_points;
    for (const auto& pt : pcl_cloud.points) {
      source_points.emplace_back(pt.x, pt.y, pt.z);
    }

    RCLCPP_INFO(this->get_logger(), "Converted source point cloud to Eigen::Vector3f");

    int num_threads = 16;                       // Number of threads to be used
    double downsampling_resolution = 0.25;     // Downsampling resolution
    int num_neighbors = 10;                    // Number of neighbor points used for normal and covariance estimation
    double max_correspondence_distance = 1.0;  // Maximum correspondence distance between points (e.g., triming threshold)

    // Convert to small_gicp::PointCloud
    auto target = std::make_shared<PointCloud>(target_points_);
    auto source = std::make_shared<PointCloud>(source_points);

    // Create KdTree
    auto target_tree = std::make_shared<KdTree<PointCloud>>(target, KdTreeBuilderOMP(num_threads));
    auto source_tree = std::make_shared<KdTree<PointCloud>>(source, KdTreeBuilderOMP(num_threads));


    RCLCPP_INFO(this->get_logger(), "made target and source point clouds");

    // Estimate point covariances
    estimate_covariances_omp(*target, *target_tree, num_neighbors, num_threads);
    estimate_covariances_omp(*source, *source_tree, num_neighbors, num_threads);

    RCLCPP_INFO(this->get_logger(), "Estimated point covariances");

    // GICP + OMP-based parallel reduction
    Registration<GICPFactor, ParallelReductionOMP> registration;
    registration.reduction.num_threads = num_threads;
    registration.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;

    // Align point clouds
    Eigen::Isometry3d init_T_target_source = Eigen::Isometry3d::Identity();
    auto result = registration.align(*target, *source, *target_tree, init_T_target_source);

    RCLCPP_INFO(this->get_logger(), "Aligned point clouds");

    // Publish pose
    // result.H = Final Hessian matrix (6x6)
    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = this->now();
    pose_msg.header.frame_id = "map";  // Adjust frame_id as needed
    pose_msg.pose.pose.position.x = result.T_target_source.translation().x();
    pose_msg.pose.pose.position.y = result.T_target_source.translation().y();
    pose_msg.pose.pose.position.z = result.T_target_source.translation().z();
    Eigen::Quaterniond q(result.T_target_source.rotation());
    pose_msg.pose.pose.orientation.x = q.x();
    pose_msg.pose.pose.orientation.y = q.y();
    pose_msg.pose.pose.orientation.z = q.z();
    pose_msg.pose.pose.orientation.w = q.w();

    pose_publisher_->publish(pose_msg);
    RCLCPP_INFO(this->get_logger(), "Published pose with covariance");
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;
  std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> target_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  std::vector<Eigen::Vector3f> target_points_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscriber_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SmallGICPNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
