cmake_minimum_required(VERSION 3.16)
project(small_gicp_ros2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)
find_package(PCL REQUIRED COMPONENTS io registration)
find_package(pcl_conversions REQUIRED)
find_package(autoware_internal_debug_msgs REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(small_gicp_ros2
  src/small_gicp_ros2.cpp
  src/small_gicp/registration/registration.cpp
  src/small_gicp/registration/registration_helper.cpp
)

ament_target_dependencies(small_gicp_ros2
  rclcpp
  sensor_msgs
  geometry_msgs
  pcl_conversions
  tf2
  tf2_ros
  Eigen3
  autoware_internal_debug_msgs
)

target_link_libraries(small_gicp_ros2
${PCL_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(small_gicp_ros2 OpenMP::OpenMP_CXX)
endif()

install(TARGETS small_gicp_ros2
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
