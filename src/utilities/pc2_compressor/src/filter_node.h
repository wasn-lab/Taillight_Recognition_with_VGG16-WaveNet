/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#pragma once
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

namespace pc2_compressor
{

class FilterNode
{
private:
  // member variables
  ros::Subscriber subscriber_;
  ros::Publisher publisher_, heartbeat_publisher_;
  ros::NodeHandle node_handle_;

  // functions
  void callback(const sensor_msgs::PointCloud2ConstPtr& msg);
  void publish_filtered_pc2(const sensor_msgs::PointCloud2ConstPtr& msg);
  int set_subscriber();
  int set_publisher();

public:
  FilterNode();
  ~FilterNode();
  void run();
};
};  // namespace pc2_compressor

