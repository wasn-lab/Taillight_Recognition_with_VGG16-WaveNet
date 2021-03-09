/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <chrono>
#include <thread>
#include <string>
#include <glog/logging.h>
#include "args_parser.h"
#include "msg_replay_node.h"

namespace msg_replay
{
static std::string get_msg_datatype(const std::string& topic)
{
  ros::master::V_TopicInfo master_topics;
  LOG(INFO) << "wait for " << topic;
  bool found = false;
  std::string ret;
  while (ros::ok() && !found)
  {
    ros::master::getTopics(master_topics);
    for (auto& master_topic : master_topics)
    {
      if (master_topic.name == topic)
      {
        LOG(INFO) << topic << " is ready.";
        found = true;
        ret = master_topic.datatype;
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  return ret;
}

MsgReplayNode::MsgReplayNode(): num_replayed_msg_(0) {}
MsgReplayNode::~MsgReplayNode() = default;

void MsgReplayNode::callback_bool(const std_msgs::BoolConstPtr& msg)
{
  num_replayed_msg_++;
  publisher_.publish(msg);
}

void MsgReplayNode::callback_empty(const std_msgs::EmptyConstPtr& msg)
{
  num_replayed_msg_++;
  publisher_.publish(msg);
}

void MsgReplayNode::callback_float64(const std_msgs::Float64ConstPtr& msg)
{
  num_replayed_msg_++;
  publisher_.publish(msg);
}

void MsgReplayNode::callback_int32(const std_msgs::Int32ConstPtr& msg)
{
  num_replayed_msg_++;
  publisher_.publish(msg);
}

void MsgReplayNode::callback_int8_multi_array(const std_msgs::Int8MultiArrayConstPtr& msg)
{
  num_replayed_msg_++;
  publisher_.publish(msg);
}

void MsgReplayNode::callback_string(const std_msgs::StringConstPtr& msg)
{
  num_replayed_msg_++;
  publisher_.publish(msg);
}

void MsgReplayNode::callback_pc2(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  num_replayed_msg_++;
  publisher_.publish(msg);
}

void MsgReplayNode::callback_cmpr_pc2(const msgs::CompressedPointCloud2ConstPtr& msg)
{
  num_replayed_msg_++;
  publisher_.publish(msg);
}

void MsgReplayNode::callback_image(const sensor_msgs::ImageConstPtr& msg)
{
  num_replayed_msg_++;
  publisher_.publish(msg);
}

void MsgReplayNode::callback_cmpr_image(const sensor_msgs::CompressedImageConstPtr& msg)
{
  num_replayed_msg_++;
  publisher_.publish(msg);
}

int MsgReplayNode::set_subscriber_and_publisher()
{
  input_topic_ = msg_replay::get_input_topic();
  if (input_topic_.empty())
  {
    LOG(ERROR) << "Empty input topic name is not allow. Please pass it with -input_topic in the command line";
    return EXIT_FAILURE;
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " subscribe " << input_topic_;

  output_topic_ = msg_replay::get_output_topic();
  if (output_topic_.empty())
  {
    LOG(ERROR) << "Empty output topic name is not allow. Please pass it with -output_topic in the command line";
    return EXIT_FAILURE;
  }

  auto datatype = get_msg_datatype(input_topic_);
  if (datatype.empty())
  {
    LOG(INFO) << input_topic_ << " datatype is empty";
    return EXIT_FAILURE;
  }
  else
  {
    LOG(INFO) << input_topic_ << " datatype is " << datatype;
  }

  if (datatype == "std_msgs/Bool")
  {
    subscriber_ = node_handle_.subscribe(input_topic_, /*queue size*/ 1, &MsgReplayNode::callback_bool, this);
    publisher_ = node_handle_.advertise<std_msgs::Bool>(output_topic_, /*queue size=*/2);
  }
  else if (datatype == "std_msgs/Empty")
  {
    subscriber_ = node_handle_.subscribe(input_topic_, /*queue size*/ 1, &MsgReplayNode::callback_empty, this);
    publisher_ = node_handle_.advertise<std_msgs::Empty>(output_topic_, /*queue size=*/2);
  }
  else if (datatype == "std_msgs/Float64")
  {
    subscriber_ = node_handle_.subscribe(input_topic_, /*queue size*/ 1, &MsgReplayNode::callback_float64, this);
    publisher_ = node_handle_.advertise<std_msgs::Float64>(output_topic_, /*queue size=*/2);
  }
  else if (datatype == "std_msgs/Int32")
  {
    subscriber_ = node_handle_.subscribe(input_topic_, /*queue size*/ 1, &MsgReplayNode::callback_int32, this);
    publisher_ = node_handle_.advertise<std_msgs::Int32>(output_topic_, /*queue size=*/2);
  }
  else if (datatype == "std_msgs/Int8MultiArray")
  {
    subscriber_ = node_handle_.subscribe(input_topic_, /*queue size*/ 1, &MsgReplayNode::callback_int8_multi_array, this);
    publisher_ = node_handle_.advertise<std_msgs::Int8MultiArray>(output_topic_, /*queue size=*/2);
  }
  else if (datatype == "std_msgs/String")
  {
    subscriber_ = node_handle_.subscribe(input_topic_, /*queue size*/ 1, &MsgReplayNode::callback_string, this);
    publisher_ = node_handle_.advertise<std_msgs::String>(output_topic_, /*queue size=*/2);
  }
  else if (datatype == "sensor_msgs/PointCloud2")
  {
    subscriber_ = node_handle_.subscribe(input_topic_, /*queue size*/ 1, &MsgReplayNode::callback_pc2, this);
    publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(output_topic_, /*queue size=*/2);
  }
  else if (datatype == "msgs/CompressedPointCloud2")
  {
    subscriber_ = node_handle_.subscribe(input_topic_, /*queue size*/ 1, &MsgReplayNode::callback_cmpr_pc2, this);
    publisher_ = node_handle_.advertise<msgs::CompressedPointCloud2>(output_topic_, /*queue size=*/2);
  }
  else if (datatype == "sensor_msgs/Image")
  {
    subscriber_ = node_handle_.subscribe(input_topic_, /*queue size*/ 1, &MsgReplayNode::callback_image, this);
    publisher_ = node_handle_.advertise<sensor_msgs::Image>(output_topic_, /*queue size=*/2);
  }
  else if (datatype == "sensor_msgs/CompressedImage")
  {
    subscriber_ = node_handle_.subscribe(input_topic_, /*queue size*/ 1, &MsgReplayNode::callback_cmpr_image, this);
    publisher_ = node_handle_.advertise<sensor_msgs::CompressedImage>(output_topic_, /*queue size=*/2);
  }
  else
  {
    LOG(ERROR) << "Cannot handle datatype " << datatype;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

void MsgReplayNode::run()
{
  if (set_subscriber_and_publisher() != EXIT_SUCCESS)
  {
    return;
  }
  ros::AsyncSpinner spinner(/*thread_count*/ 1);
  spinner.start();
  ros::Rate r(1);
  while (ros::ok())
  {
    LOG(INFO) << "replay " << input_topic_ << " at " << output_topic_ << ", FPS = " << num_replayed_msg_;
    num_replayed_msg_ = 0;
    r.sleep();
  }
  spinner.stop();
}
};  // namespace msg_replay
