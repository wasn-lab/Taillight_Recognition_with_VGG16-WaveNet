/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#pragma once
#include <mutex>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include "pc2_compressor.h"

namespace pc2_compressor
{
class PC2CompressorNode
{
private:
  // member variables
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  ros::NodeHandle node_handle_;
  std::mutex mu_publisher_;  // guard publisher_
  compression_format cmpr_fmt_;

  // functions
  void callback(const sensor_msgs::PointCloud2ConstPtr& msg);
  void publish_compressed_pc2(const sensor_msgs::PointCloud2ConstPtr& msg);
  int set_subscriber();
  int set_publisher();

public:
  PC2CompressorNode();
  ~PC2CompressorNode();
  void run();
};
};  // namespace pc2_compressor
