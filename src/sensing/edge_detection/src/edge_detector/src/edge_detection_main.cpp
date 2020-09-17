
/*   edge_detection_main.cpp
 *   File: edge_detection_main0.cpp
 *   Created on: Sep , 2019
 *   Institute: ITRI ICL U300
 *   Author: bchsu1125@gmail.com
 */

#define PRINT_TIME 0
#define CUDA 0

#include "headers.h"
#include "edge_detection_tools.h"
#include "class_edge_detection.h"

#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <mutex>
#include <cmath>
#include <cstdio>
#include <pthread.h>
#include <boost/thread/recursive_mutex.hpp>

#include <ros/callback_queue.h>
#include <ros/spinner.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/String.h>
#include <std_msgs/Header.h>
#include <std_msgs/Empty.h>


static EdgeDetection TOPED;
static EdgeDetection FRED;
static EdgeDetection FLED;

static double theta_sample;
static bool lidar_all_flag;
static bool top_only_flag;

static float max_radius;

// grid_map
static double grid_length_x;
static double grid_length_y;
static double grid_resolution;
static double grid_position_x;
static double grid_position_y;
static double grid_min_value;
static double grid_max_value;
static double maximum_lidar_height_thres;
static double minimum_lidar_height_thres;

boost::recursive_mutex mergedMap;

static ros::Publisher mopho_input_pointCloudPublisher;
static sensor_msgs::PointCloud2 mopho_input_pointCloud_msg;

static ros::Publisher ground_pointCloudPublisher;
static sensor_msgs::PointCloud2 ground_pointCloud_msg;

static ros::Publisher non_ground_pointCloudPublisher;
static sensor_msgs::PointCloud2 non_ground_pointCloud_msg;

static ros::Publisher occupancy_grid_publisher;
static nav_msgs::OccupancyGrid occupancy_grid_msg;

static ros::Publisher pub_occupancy_dense_grid;
static nav_msgs::OccupancyGrid out_occupancy_dense_grid;

static ros::Publisher ring_edge_pointcloud_publisher;
static sensor_msgs::PointCloud2 ring_edge_pointcloud_publisher_msg;
static ros::Publisher pub_ring_edge_heartbeat;


static bool is_FT_grid_new, is_FR_grid_new, is_FL_grid_new;
static bool is_init_merged_map;
//static bool is_init_FT_map, is_init_FR_map;

static grid_map::GridMap merged_costmap_;
static grid_map::GridMap front_top_grid_map_data, front_right_grid_map_data, front_left_grid_map_data;

// grid_map generator
grid_map::GridMap initializeGridMap(grid_map::GridMap input_grid_map)
{
  std::cout << "Initializing grid_map data from front_top callback... " << std::endl;
  grid_map::GridMap tmp_grid_map;
  tmp_grid_map = input_grid_map;

  ROS_INFO("Created map with size %f x %f m (%i x %i cells).\n The center of the map is located at (%f, %f) in the %s "
           "frame.",
           tmp_grid_map.getLength().x(), tmp_grid_map.getLength().y(), tmp_grid_map.getSize()(0),
           tmp_grid_map.getSize()(1), tmp_grid_map.getPosition().x(), tmp_grid_map.getPosition().y(),
           tmp_grid_map.getFrameId().c_str());

  Eigen::Array2i size;
  size(0) = tmp_grid_map.getSize()(0);
  size(1) = tmp_grid_map.getSize()(1);

  tmp_grid_map.add("merged_costmap_layer", grid_map::Matrix::Constant(size(0), size(1), 0.0));
  tmp_grid_map.add("front_right_points_layer", grid_map::Matrix::Constant(size(0), size(1), 0.0));
  tmp_grid_map.add("front_left_points_layer", grid_map::Matrix::Constant(size(0), size(1), 0.0));

  return tmp_grid_map;
}

void callback_LidarFrontTop(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  if (lidar_all_flag)
  {
    return;
  }
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  std::chrono::high_resolution_clock::time_point check_ms = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> fp_ms = check_ms - start;

  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr LidAll_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr ring_edge_ftop_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr ring_edge_fright_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr ring_edge_fleft_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);

  pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);

  std_msgs::Header in_header = msg->header;

  pcl::fromROSMsg(*msg, *LidAll_cloudPtr);

#if PRINT_TIME
  check_ms = std::chrono::high_resolution_clock::now();
  fp_ms = check_ms - start;
  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "[Time] fromROSMsg " << fp_ms.count() << "ms\n";
  std::cout << "-----------------------------------------------------------------" << std::endl;
#endif

  *filtered_scan_ptr += *LidAll_cloudPtr;

#if PRINT_TIME
  check_ms = std::chrono::high_resolution_clock::now();
  fp_ms = check_ms - start;
  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "[Time] downsampling " << fp_ms.count() << "ms\n";
  std::cout << "-----------------------------------------------------------------" << std::endl;
#endif

  *filtered_cloudPtr = hollow_removal(filtered_scan_ptr, -13, 0.7, -2.0, 2.0, -3, 0.5, -15, 55, -15, 15, -5, -0.8);

  TOPED.setInputCloud(filtered_cloudPtr);
  TOPED.startThread();
  TOPED.waitThread();

#if PRINT_TIME

  check_ms = std::chrono::high_resolution_clock::now();
  fp_ms = check_ms - start;
  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "[Time] waitThread " << fp_ms.count() << "ms\n";
  std::cout << "-----------------------------------------------------------------" << std::endl;
#endif

  front_top_grid_map_data = TOPED.getGridMap();

  if (!is_init_merged_map)
  {
    boost::recursive_mutex::scoped_lock scopedLock(mergedMap);

    merged_costmap_ = initializeGridMap(front_top_grid_map_data);
    is_init_merged_map = true;
    scopedLock.unlock();

    return;
  }

  if (is_FR_grid_new && is_FL_grid_new)
  {
    boost::recursive_mutex::scoped_lock scopedLock(mergedMap);

    if (merged_costmap_.exists("merged_costmap_layer") && merged_costmap_.exists("front_right_points_layer") &&
        merged_costmap_.exists("front_left_points_layer") &&
        front_right_grid_map_data.exists("front_right_points_layer") &&
        front_left_grid_map_data.exists("front_left_points_layer"))
    {
      grid_map::GridMap tmp_map;
      grid_map::GridMap tmp_map_ft;
      grid_map::GridMap tmp_map_fr;
      grid_map::GridMap tmp_map_fl;

      tmp_map = merged_costmap_;
      tmp_map_ft = front_top_grid_map_data;
      tmp_map_fr = front_right_grid_map_data;
      tmp_map_fl = front_left_grid_map_data;

      front_top_grid_map_data.clearAll();
      front_top_grid_map_data.resetTimestamp();

      front_right_grid_map_data.clearAll();
      front_right_grid_map_data.resetTimestamp();

      front_left_grid_map_data.clearAll();
      front_left_grid_map_data.resetTimestamp();

      tmp_map.clear("merged_costmap_layer");
      tmp_map["merged_costmap_layer"] = tmp_map["merged_costmap_layer"].cwiseMax(tmp_map_ft["front_top_points_layer"]);
      tmp_map["merged_costmap_layer"] = tmp_map["merged_costmap_layer"].cwiseMax(tmp_map_fr["front_right_points_"
                                                                                            "layer"]);
      tmp_map["merged_costmap_layer"] = tmp_map["merged_costmap_layer"].cwiseMax(tmp_map_fl["front_left_points_layer"]);
      merged_costmap_["merged_costmap_layer"] = tmp_map["merged_costmap_layer"];

      is_FR_grid_new = false;
      is_FL_grid_new = false;

      if (occupancy_grid_publisher.getNumSubscribers() != 0)
      {
        grid_map::GridMapRosConverter::toOccupancyGrid(merged_costmap_, "merged_costmap_layer", 0, 1,
                                                       occupancy_grid_msg);
        occupancy_grid_msg.header = in_header;
        occupancy_grid_msg.header.frame_id = "base_link";
        occupancy_grid_publisher.publish(occupancy_grid_msg);
      }

      if (ring_edge_pointcloud_publisher.getNumSubscribers() != 0)
      {
        ring_edge_ftop_cloudPtr = TOPED.getRingEdgePointCloud();
        if (!top_only_flag)
        {
          ring_edge_fright_cloudPtr = FRED.getRingEdgePointCloud();
          ring_edge_fleft_cloudPtr = FLED.getRingEdgePointCloud();
          *ring_edge_ftop_cloudPtr += *ring_edge_fright_cloudPtr;
          *ring_edge_ftop_cloudPtr += *ring_edge_fleft_cloudPtr;
        }
        pcl::toROSMsg(*ring_edge_ftop_cloudPtr, ring_edge_pointcloud_publisher_msg);
        ring_edge_pointcloud_publisher_msg.header.stamp = msg->header.stamp;
        ring_edge_pointcloud_publisher_msg.header.seq = msg->header.seq;

        ring_edge_pointcloud_publisher_msg.header.frame_id = "base_link";
        ring_edge_pointcloud_publisher.publish(ring_edge_pointcloud_publisher_msg);
        
        // heartbeat
        std_msgs::Empty empty_msg;
        pub_ring_edge_heartbeat.publish(empty_msg);
      }

      scopedLock.unlock();
    }
    else
    {
      std::cout << "No layer named merged_costmap_layer " << std::endl;
      return;
    }
  }
  else
  {
    std::cout << "No new FR OR FL data" << std::endl;
    return;
  }

#if PRINT_TIME
  check_ms = std::chrono::high_resolution_clock::now();
  fp_ms = check_ms - start;
  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "[Time] merged_costmap_layer " << fp_ms.count() << "ms\n";
  std::cout << "-----------------------------------------------------------------" << std::endl;
#endif

  if (mopho_input_pointCloudPublisher.getNumSubscribers() != 0)
  {
    pcl::toROSMsg(*filtered_cloudPtr, mopho_input_pointCloud_msg);
    mopho_input_pointCloud_msg.header.stamp = msg->header.stamp;
    mopho_input_pointCloud_msg.header.seq = msg->header.seq;
    mopho_input_pointCloud_msg.header.frame_id = "base_link";
    mopho_input_pointCloudPublisher.publish(mopho_input_pointCloud_msg);
  }

  if (ground_pointCloudPublisher.getNumSubscribers() != 0)
  {
    ground_cloudPtr = TOPED.getGroundPointCloud();
    pcl::toROSMsg(*ground_cloudPtr, ground_pointCloud_msg);
    ground_pointCloud_msg.header.stamp = msg->header.stamp;
    ground_pointCloud_msg.header.seq = msg->header.seq;
    ground_pointCloud_msg.header.frame_id = "base_link";
    ground_pointCloudPublisher.publish(ground_pointCloud_msg);
  }

  if (non_ground_pointCloudPublisher.getNumSubscribers() != 0)
  {
    non_ground_cloudPtr = TOPED.getNonGroundPointCloud();

    pcl::toROSMsg(*non_ground_cloudPtr, non_ground_pointCloud_msg);
    non_ground_pointCloud_msg.header.stamp = msg->header.stamp;
    non_ground_pointCloud_msg.header.seq = msg->header.seq;
    non_ground_pointCloud_msg.header.frame_id = "base_link";
    non_ground_pointCloudPublisher.publish(non_ground_pointCloud_msg);
  }

  check_ms = std::chrono::high_resolution_clock::now();
  fp_ms = check_ms - start;
  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "[Time] Top main function " << fp_ms.count() << "ms\n";
  std::cout << "-----------------------------------------------------------------" << std::endl;
}

void callback_LidarFrontRight(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  if (lidar_all_flag)
  {
    return;
  }
  if (!is_init_merged_map)
  {
    // ros::Rate ros_rate(1);
    std::cout << "[front_right] Waiting for initializing... " << std::endl;
    // ros_rate.sleep();
    return;
  }
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  std::chrono::high_resolution_clock::time_point check_ms;
  std::chrono::duration<double, std::milli> fp_ms = check_ms - start;
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr LidAll_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr ring_edge_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);

  grid_map::GridMap grid_map_data;
  std_msgs::Header in_header = msg->header;

  pcl::fromROSMsg(*msg, *LidAll_cloudPtr);

  *filtered_scan_ptr += *LidAll_cloudPtr;
  *filtered_cloudPtr = hollow_removal(filtered_scan_ptr, -13, 0.7, -2.0, 2.0, -3, 0.5, -15, 55, -15, 15, -5, -0.8);

  FRED.setInputCloud(filtered_cloudPtr);
  FRED.startThread();
  FRED.waitThread();

  boost::recursive_mutex::scoped_lock scopedLock(mergedMap);

  front_right_grid_map_data = FRED.getGridMap();
  is_FR_grid_new = true;

  scopedLock.unlock();

#if PRINT_TIME
  check_ms = std::chrono::high_resolution_clock::now();
  fp_ms = check_ms - start;
  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "[Time] FR main function " << fp_ms.count() << "ms\n";
  std::cout << "-----------------------------------------------------------------" << std::endl;
#endif
}

void callback_LidarFrontLeft(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  if (lidar_all_flag)
  {
    return;
  }
  if (!is_init_merged_map)
  {
    // ros::Rate ros_rate(1);
    std::cout << "[front_right] Waiting for initializing... " << std::endl;
    // ros_rate.sleep();
    return;
  }
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr LidAll_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr ring_edge_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);

  grid_map::GridMap grid_map_data;
  std_msgs::Header in_header = msg->header;

  pcl::fromROSMsg(*msg, *LidAll_cloudPtr);

  *filtered_scan_ptr += *LidAll_cloudPtr;
  *filtered_cloudPtr = hollow_removal(filtered_scan_ptr, -13, 0.7, -2.0, 2.0, -3, 0.5, -15, 55, -15, 15, -5, -0.8);

  FLED.setInputCloud(filtered_cloudPtr);
  FLED.startThread();

  FLED.waitThread();

  boost::recursive_mutex::scoped_lock scopedLock(mergedMap);

  front_left_grid_map_data = FLED.getGridMap();
  is_FL_grid_new = true;

  scopedLock.unlock();
#if PRINT_TIME
  std::chrono::high_resolution_clock::time_point check_ms = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> fp_ms = check_ms - start;
  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "[Time] FL main function " << fp_ms.count() << "ms\n";
  std::cout << "-----------------------------------------------------------------" << std::endl;
#endif
}

void callback_LidarAll(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  if (!lidar_all_flag)
  {
    return;
  }
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  std::chrono::high_resolution_clock::time_point check_ms = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> fp_ms = check_ms - start;

  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr LidAll_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr ring_edge_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);

  std_msgs::Header in_header = msg->header;

  pcl::fromROSMsg(*msg, *LidAll_cloudPtr);

#if PRINT_TIME
  check_ms = std::chrono::high_resolution_clock::now();
  fp_ms = check_ms - start;
  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "[Time] fromROSMsg " << fp_ms.count() << "ms\n";
  std::cout << "-----------------------------------------------------------------" << std::endl;
#endif

  *filtered_scan_ptr += *LidAll_cloudPtr;

#if PRINT_TIME
  check_ms = std::chrono::high_resolution_clock::now();
  fp_ms = check_ms - start;
  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "[Time] downsampling " << fp_ms.count() << "ms\n";
  std::cout << "-----------------------------------------------------------------" << std::endl;
#endif

  *filtered_cloudPtr = hollow_removal(filtered_scan_ptr, -13, 0.7, -2.0, 2.0, -3, 0.5, -15, 55, -15, 15, -5, -0.8);

  TOPED.setInputCloud(filtered_cloudPtr);
  TOPED.startThread();
  TOPED.waitThread();

#if PRINT_TIME

  check_ms = std::chrono::high_resolution_clock::now();
  fp_ms = check_ms - start;
  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "[Time] waitThread " << fp_ms.count() << "ms\n";
  std::cout << "-----------------------------------------------------------------" << std::endl;
#endif

  front_top_grid_map_data = TOPED.getGridMap();

  if (!is_init_merged_map)
  {
    boost::recursive_mutex::scoped_lock scopedLock(mergedMap);

    merged_costmap_ = initializeGridMap(front_top_grid_map_data);
    is_init_merged_map = true;
    scopedLock.unlock();

    return;
  }

  if (is_FR_grid_new && is_FL_grid_new)
  {
    boost::recursive_mutex::scoped_lock scopedLock(mergedMap);

    if (merged_costmap_.exists("merged_costmap_layer") && merged_costmap_.exists("front_right_points_layer") &&
        merged_costmap_.exists("front_left_points_layer"))
    {
      grid_map::GridMap tmp_map;
      grid_map::GridMap tmp_map_ft;
      grid_map::GridMap tmp_map_fr;
      grid_map::GridMap tmp_map_fl;

      tmp_map = merged_costmap_;
      tmp_map_ft = front_top_grid_map_data;
      tmp_map_fr = front_right_grid_map_data;
      tmp_map_fl = front_left_grid_map_data;

      front_top_grid_map_data.clearAll();
      front_top_grid_map_data.resetTimestamp();

      front_right_grid_map_data.clearAll();
      front_right_grid_map_data.resetTimestamp();

      front_left_grid_map_data.clearAll();
      front_left_grid_map_data.resetTimestamp();

      tmp_map.clear("merged_costmap_layer");
      tmp_map["merged_costmap_layer"] = tmp_map["merged_costmap_layer"].cwiseMax(tmp_map_ft["front_top_points_layer"]);
      merged_costmap_["merged_costmap_layer"] = tmp_map["merged_costmap_layer"];

      if (occupancy_grid_publisher.getNumSubscribers() != 0)
      {
        grid_map::GridMapRosConverter::toOccupancyGrid(merged_costmap_, "merged_costmap_layer", 0, 1,
                                                       occupancy_grid_msg);
        occupancy_grid_msg.header = in_header;
        occupancy_grid_msg.header.frame_id = "base_link";
        occupancy_grid_publisher.publish(occupancy_grid_msg);
      }
      scopedLock.unlock();
    }
    else
    {
      std::cout << "No layer named merged_costmap_layer " << std::endl;
      return;
    }
  }
  else
  {
    std::cout << "No new FR OR FL data" << std::endl;
    return;
  }

#if PRINT_TIME
  check_ms = std::chrono::high_resolution_clock::now();
  fp_ms = check_ms - start;
  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "[Time] merged_costmap_layer " << fp_ms.count() << "ms\n";
  std::cout << "-----------------------------------------------------------------" << std::endl;
#endif

  if (mopho_input_pointCloudPublisher.getNumSubscribers() != 0)
  {
    pcl::toROSMsg(*filtered_cloudPtr, mopho_input_pointCloud_msg);
    mopho_input_pointCloud_msg.header.stamp = msg->header.stamp;
    mopho_input_pointCloud_msg.header.seq = msg->header.seq;
    mopho_input_pointCloud_msg.header.frame_id = "base_link";
    mopho_input_pointCloudPublisher.publish(mopho_input_pointCloud_msg);
  }

  if (ground_pointCloudPublisher.getNumSubscribers() != 0)
  {
    ground_cloudPtr = TOPED.getGroundPointCloud();
    pcl::toROSMsg(*ground_cloudPtr, ground_pointCloud_msg);
    ground_pointCloud_msg.header.stamp = msg->header.stamp;
    ground_pointCloud_msg.header.seq = msg->header.seq;
    ground_pointCloud_msg.header.frame_id = "base_link";
    ground_pointCloudPublisher.publish(ground_pointCloud_msg);
  }

  if (non_ground_pointCloudPublisher.getNumSubscribers() != 0)
  {
    non_ground_cloudPtr = TOPED.getNonGroundPointCloud();

    pcl::toROSMsg(*non_ground_cloudPtr, non_ground_pointCloud_msg);
    non_ground_pointCloud_msg.header.stamp = msg->header.stamp;
    non_ground_pointCloud_msg.header.seq = msg->header.seq;
    non_ground_pointCloud_msg.header.frame_id = "base_link";
    non_ground_pointCloudPublisher.publish(non_ground_pointCloud_msg);
  }

  if (ring_edge_pointcloud_publisher.getNumSubscribers() != 0)
  {
    ring_edge_cloudPtr = TOPED.getRingEdgePointCloud();

    pcl::toROSMsg(*ring_edge_cloudPtr, ring_edge_pointcloud_publisher_msg);
    ring_edge_pointcloud_publisher_msg.header.stamp = msg->header.stamp;
    ring_edge_pointcloud_publisher_msg.header.seq = msg->header.seq;

    ring_edge_pointcloud_publisher_msg.header.frame_id = "base_link";
    ring_edge_pointcloud_publisher.publish(ring_edge_pointcloud_publisher_msg);
    
    // heartbeat
    std_msgs::Empty empty_msg;
    pub_ring_edge_heartbeat.publish(empty_msg);
  }
#if PRINT_TIME
  check_ms = std::chrono::high_resolution_clock::now();
  fp_ms = check_ms - start;
  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "[Time] LidarAll main function " << fp_ms.count() << "ms\n";
  std::cout << "-----------------------------------------------------------------" << std::endl;
#endif
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "edge_detection");
  ros::NodeHandle n;
  ros::NodeHandle private_n("~");
  private_n.getParam("theta_sample", theta_sample);
  private_n.getParam("max_radius", max_radius);
  private_n.getParam("grid_min_value", grid_min_value);
  private_n.getParam("grid_max_value", grid_max_value);
  private_n.getParam("grid_length_x", grid_length_x);
  private_n.getParam("grid_length_y", grid_length_y);
  private_n.getParam("grid_resolution", grid_resolution);
  private_n.getParam("grid_position_x", grid_position_x);
  private_n.getParam("grid_position_y", grid_position_y);
  private_n.getParam("maximum_lidar_height_thres", maximum_lidar_height_thres);
  private_n.getParam("LidarAll_flag", lidar_all_flag);
  private_n.getParam("top_only_flag", top_only_flag);

  std::cout << "LidarAll_flag : " << lidar_all_flag << std::endl;

  TOPED.setLayerName("front_top_points_layer");
  TOPED.initGridmap();

  FRED.setLayerName("front_right_points_layer");
  FRED.initGridmap();

  FLED.setLayerName("front_left_points_layer");
  FLED.initGridmap();

  is_FT_grid_new = false;
  if (lidar_all_flag)
  {
    is_FR_grid_new = true;
    is_FL_grid_new = true;
  }
  else
  {
    is_FR_grid_new = false;
    is_FL_grid_new = false;
  }

  is_init_merged_map = false;

  ros::Subscriber LidAllSub = n.subscribe("LidarAll", 1, callback_LidarAll);
  ros::Subscriber LidFrontTopSub = n.subscribe("LidarFrontTop", 1, callback_LidarFrontTop);
  ros::Subscriber LidFrontRightSub = n.subscribe("LidarFrontRight", 1, callback_LidarFrontRight);
  ros::Subscriber LidFrontLeftSub = n.subscribe("LidarFrontLeft", 1, callback_LidarFrontLeft);

  mopho_input_pointCloudPublisher = n.advertise<sensor_msgs::PointCloud2>("mopho_input", 1, true);
  ground_pointCloudPublisher = n.advertise<sensor_msgs::PointCloud2>("ground_point_cloud", 1, true);
  non_ground_pointCloudPublisher = n.advertise<sensor_msgs::PointCloud2>("non_ground_point_cloud", 1, true);

  occupancy_grid_publisher = n.advertise<nav_msgs::OccupancyGrid>("occupancy_grid", 1, true);
  pub_occupancy_dense_grid = n.advertise<nav_msgs::OccupancyGrid>("occupancy_dense_grid", 1, true);

  ring_edge_pointcloud_publisher = n.advertise<sensor_msgs::PointCloud2>("ring_edge_point_cloud", 1, false);
  pub_ring_edge_heartbeat = n.advertise<std_msgs::Empty>("/ring_edge_point_cloud/heartbeat", 1);

  ros::MultiThreadedSpinner s(3);
  ros::spin(s);
  return 0;
}
