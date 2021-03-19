/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#pragma once 
#include <pcl_ros/point_cloud.h>

void gen_random_msg();
sensor_msgs::PointCloud2::ConstPtr get_msg_ptr();
pcl::PointCloud<ouster_ros::OS1::PointOS1>::ConstPtr get_cloud_ptr();
