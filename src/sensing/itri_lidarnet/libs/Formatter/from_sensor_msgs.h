#ifndef FROM_SENSOR_MSGS_H
#define FROM_SENSOR_MSGS_H

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

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_planar.h>

// std::string lidar_brand;
pcl::PointCloud<pcl::PointXYZIR> get_ring_pcl_from_sensor_msgs(const sensor_msgs::PointCloud2 & cloud_msg);

pcl::RangeImage PointCloudtoRangeImage(pcl::PointCloud<pcl::PointXYZIR>::Ptr input_cloud, std::string lidar_brand, int ring_num);


#endif // FROM_SENSOR_MSGS_H