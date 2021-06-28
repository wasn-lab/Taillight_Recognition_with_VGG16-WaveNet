/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#pragma once
#include <sensor_msgs/PointCloud2.h>

namespace pc_transform
{
pcl::PointCloud<pcl::PointXYZI>::Ptr pc2_msg_to_xyzi(const sensor_msgs::PointCloud2ConstPtr& msg_ptr);
uint32_t checksum_of(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_ptr);
uint32_t checksum_of(const sensor_msgs::PointCloud2ConstPtr& msg);
};  // namespace pc_transform
