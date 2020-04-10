#ifndef CONVEX_FUSION_B1_V2_H
#define CONVEX_FUSION_B1_V2_H

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
#include "fusion_source_id.h"
#include "camera_params.h"
#include "drivenet/object_label_util.h"

#include "costmap_generator.h"

class ConvexFusionB1V2
{
public:
  ros::Publisher error_code_pub_;
  ros::Publisher camera_detection_pub_;
  ros::Publisher occupancy_grid_publisher_;
  grid_map::GridMap costmap_;
  bool use_gridmap_publish_ = true;
  CosmapGenerator cosmapGener_;

  void initial(const std::string& nodename, int argc, char** argv);

  void registerCallBackLidarAllNonGround(void (*callback_nonground)(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&));

  void registerCallBackCameraDetection(void (*callback_front_bottom_60)(const msgs::DetectedObjectArray::ConstPtr&));

  void sendErrorCode(unsigned int error_code, const std::string& frame_id, int module_id);
  void sendCameraResults(CLUSTER_INFO* cluster_info, CLUSTER_INFO* cluster_info_bbox, int cluster_size,
                         ros::Time rostime, const std::string& frame_id);
};

#endif  // CONVEX_FUSION_B1_V2_H
