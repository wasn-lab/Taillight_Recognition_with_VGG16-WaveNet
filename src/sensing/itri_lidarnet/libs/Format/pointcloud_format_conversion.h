#ifndef POINTCLOUD_FORMAT_CONVERSION_H
#define POINTCLOUD_FORMAT_CONVERSION_H

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <omp.h>
#include <thread>
#include <unistd.h>  //sleep
#include <functional>
#include <cerrno>
#include <cstdlib>

#include "../UserDefine.h"
#include <ros/ros.h>
#include <std_msgs/Header.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/compression/octree_pointcloud_compression.h>

pcl::PointCloud<pcl::PointXYZIR> SensorMsgs_to_XYZIR(const sensor_msgs::PointCloud2& cloud_msg, string brand);

pcl::PointCloud<pcl::PointXYZRGBA> XYZIR_to_XYZRBGA(pcl::PointCloud<pcl::PointXYZIR>::Ptr input_cloud);

pcl::PointCloud<pcl::PointXYZIR> XYZRBGA_to_XYZIR(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr input_cloud);

#endif