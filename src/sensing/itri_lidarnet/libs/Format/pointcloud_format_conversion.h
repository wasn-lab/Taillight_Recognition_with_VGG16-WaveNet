#ifndef POINTCLOUD_FORMAT_CONVERSION_H
#define POINTCLOUD_FORMAT_CONVERSION_H

#include "../UserDefine.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <sensor_msgs/PointCloud2.h>
#include "lidar_hardware.h"

pcl::PointCloud<pcl::PointXYZIR>::Ptr SensorMsgs_to_XYZIR(const sensor_msgs::PointCloud2& cloud_msg, lidar::Hardware brand);

pcl::PointCloud<pcl::PointXYZRGB> XYZIR_to_XYZRGB(pcl::PointCloud<pcl::PointXYZIR>::Ptr input_cloud);

pcl::PointCloud<pcl::PointXYZIR> XYZRGB_to_XYZIR(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud);

#endif
