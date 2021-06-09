/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#pragma once
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

namespace pc2_compressor
{

class Ouster64ToXYZIRNode
{
private:
  // member variables
  ros::Subscriber subscriber_;
  ros::Publisher xyzir_publisher_, xyzir_heartbeat_publisher_;
  ros::Publisher raw_heartbeat_publisher_;
  ros::NodeHandle node_handle_;
  int32_t msgs_per_second_;
  int32_t latency_wrt_raw_in_ms_;

  // functions
  void callback(const sensor_msgs::PointCloud2ConstPtr& msg);
  void publish_filtered_pc2(const sensor_msgs::PointCloud2ConstPtr& msg);
  int set_subscriber();
  int set_publisher();

public:
  Ouster64ToXYZIRNode() = default;
  ~Ouster64ToXYZIRNode() = default;
  void run();
};
};  // namespace pc2_compressor

