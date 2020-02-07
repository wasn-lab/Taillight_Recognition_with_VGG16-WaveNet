#ifndef CONVEX_FUSION_B1_H
#define CONVEX_FUSION_B1_H

// =============================================
//                      STD
// =============================================
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <mutex>

// =============================================
//                      CUDA
// =============================================
#include <cuda.h>
#include <cuda_runtime.h>

// =============================================
//                      PCL
// =============================================
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

// =============================================
//                      ROS
// =============================================

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/PointCloud2.h>

#include "msgs/DetectedObjectArray.h"
#include "msgs/ErrorCode.h"

#include "UserDefine.h"
#include "camera_params.h"

#include "costmap_generator.h"

class ConvexFusionB1
{
public:
  ros::Publisher error_code_pub_;
  ros::Publisher camera_detection_pub_;
  ros::Publisher occupancy_grid_publisher;
  grid_map::GridMap g_costmap_;
  bool g_use_gridmap_publish = true;
  CosmapGenerator g_cosmapGener;

  void initial(std::string nodename, int argc, char** argv);

  void registerCallBackLidarAllNonGround(void (*callback_nonground)(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&));

  void registerCallBackCameraDetection(void (*callback_front_60)(const msgs::DetectedObjectArray::ConstPtr&),
                                              void (*callback_top_front_120)(const msgs::DetectedObjectArray::ConstPtr&),
                                              void (*callback_top_rear_120)(const msgs::DetectedObjectArray::ConstPtr&));

  void sendErrorCode(unsigned int error_code, std::string& frame_id, int module_id);
  void sendCameraResults(CLUSTER_INFO* cluster_info, CLUSTER_INFO* cluster_info_bbox, int cluster_size,
                         ros::Time rostime, std::string& frame_id);
};

#endif  // CONVEX_FUSION_B1_H
