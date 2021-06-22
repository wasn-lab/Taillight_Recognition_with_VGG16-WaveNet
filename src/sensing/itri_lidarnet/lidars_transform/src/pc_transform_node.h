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
  uint32_t max_latency_in_ms_ = 0;
  uint32_t num_msgs_per_second_ = 0;

  // functions
  void callback(sensor_msgs::PointCloud2Ptr msg);
  void publish(sensor_msgs::PointCloud2ConstPtr msg);
  int set_subscriber();
  int set_publisher();

public:
  PCTransformNode();
  ~PCTransformNode() = default;
  sensor_msgs::PointCloud2ConstPtr transform(const sensor_msgs::PointCloud2ConstPtr& msg);
  int set_transform_parameters();
  int set_transform_parameters(const float tx, const float ty, const float tz, const float rx, const float ry,
                               const float rz);
  void run();
};
};  // namespace pc_transform
