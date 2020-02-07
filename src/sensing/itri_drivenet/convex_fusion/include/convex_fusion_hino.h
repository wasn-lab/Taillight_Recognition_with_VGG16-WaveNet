#ifndef CONVEX_FUSION_HINO_H
#define CONVEX_FUSION_HINO_H

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

class ConvexFusionHino
{
public:
  void initial(std::string nodename, int argc, char** argv);

  void RegisterCallBackLidarAllNonGround(void (*cb1)(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&));
  void RegisterCallBackCameraDetection(void (*cb1)(const msgs::DetectedObjectArray::ConstPtr&),
                                       void (*cb2)(const msgs::DetectedObjectArray::ConstPtr&),
                                       void (*cb3)(const msgs::DetectedObjectArray::ConstPtr&));

  void send_ErrorCode(unsigned int error_code);

  void Send_CameraResults(CLUSTER_INFO* cluster_info, int cluster_size, ros::Time rostime, std::string frameId);

private:
  ros::Publisher ErrorCode_pub;
  ros::Publisher CameraDetection_pub;
};

#endif  // CONVEX_FUSION_HINO_H
