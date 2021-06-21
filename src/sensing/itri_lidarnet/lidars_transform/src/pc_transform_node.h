/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#pragma once
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include "pc_transform_gpu.h"

namespace pc_transform
{
class PCTransformNode
{
private:
  // member variables
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  ros::Publisher heartbeat_publisher_;
  ros::NodeHandle node_handle_;
  PCTransformGPU<pcl::PointXYZI> pc_transform_gpu_;

  // functions
  void callback(sensor_msgs::PointCloud2Ptr msg);
  void publish(sensor_msgs::PointCloud2ConstPtr msg);
  int set_subscriber();
  int set_publisher();
  int set_transform_parameters();


public:
  PCTransformNode();
  ~PCTransformNode() = default;
  sensor_msgs::PointCloud2ConstPtr transform(const sensor_msgs::PointCloud2ConstPtr& msg);
  void run();
};
};  // namespace pc_transform
