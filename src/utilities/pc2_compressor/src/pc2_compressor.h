/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#pragma once
#include <string>
#include <pcl_ros/point_cloud.h>
#include "msgs/CompressedPointCloud2.h"
#include "pc2_compression_format.h"

namespace pc2_compressor
{
msgs::CompressedPointCloud2ConstPtr compress_msg(const sensor_msgs::PointCloud2ConstPtr& in_msg,
                                                 const int32_t fmt = compression_format::lzf);
sensor_msgs::PointCloud2ConstPtr decompress_msg(const msgs::CompressedPointCloud2ConstPtr& cmpr_msg);

bool is_equal_pc2(const sensor_msgs::PointCloud2ConstPtr& a, const sensor_msgs::PointCloud2ConstPtr& b);
uint64_t size_of_msg(const sensor_msgs::PointCloud2ConstPtr& msg);
uint64_t size_of_msg(const msgs::CompressedPointCloud2ConstPtr& msg);
};  // namespace pc2_compressor
