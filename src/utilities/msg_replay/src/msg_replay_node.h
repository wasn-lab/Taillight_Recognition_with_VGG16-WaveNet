/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#pragma once
#include <string>
#include <atomic>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Int8MultiArray.h>
#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include "msgs/CompressedPointCloud2.h"

namespace msg_replay
{
class MsgReplayNode
{
private:
  // member variables
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  ros::NodeHandle node_handle_;
  std::atomic_uint32_t num_replayed_msg_;
  std::string input_topic_, output_topic_;

  // functions
  void callback_bool(const std_msgs::BoolConstPtr& msg);
  void callback_empty(const std_msgs::EmptyConstPtr& msg);
  void callback_float64(const std_msgs::Float64ConstPtr& msg);
  void callback_int32(const std_msgs::Int32ConstPtr& msg);
  void callback_int8_multi_array(const std_msgs::Int8MultiArrayConstPtr& msg);
  void callback_string(const std_msgs::StringConstPtr& msg);
  void callback_pc2(const sensor_msgs::PointCloud2ConstPtr& msg);
  void callback_cmpr_pc2(const msgs::CompressedPointCloud2ConstPtr& msg);
  void callback_image(const sensor_msgs::ImageConstPtr& msg);
  void callback_cmpr_image(const sensor_msgs::CompressedImageConstPtr& msg);
  int set_subscriber_and_publisher();

public:
  MsgReplayNode();
  ~MsgReplayNode();
  void run();
};
};  // namespace msg_replay
