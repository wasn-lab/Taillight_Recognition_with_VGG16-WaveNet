#pragma once
#include <string>
#include "pcl_ros/point_cloud.h"

namespace pc2_compressor
{
std::string compress(const sensor_msgs::PointCloud2ConstPtr& in_pcd_message);
sensor_msgs::PointCloud2Ptr decompress(const std::string& in_str);

};  // namespace pc2_compressor
