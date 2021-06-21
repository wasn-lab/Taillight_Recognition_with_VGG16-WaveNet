/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <string>
#include <glog/logging.h>
#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Empty.h>
#include "pc_transform_args_parser.h"
#include "pc_transform_node.h"

namespace pc_transform
{
PCTransformNode::PCTransformNode() : pc_transform_gpu_()
{
}

void PCTransformNode::callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
}

void PCTransformNode::publish(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  publisher_.publish(msg);

  std_msgs::Empty empty_msg;
  heartbeat_publisher_.publish(empty_msg);

  ros::Time now = ros::Time::now();
  int32_t latency = (now.sec - msg->header.stamp.sec) * 1000 + (now.nsec - msg->header.stamp.nsec) / 1000000;
  LOG_EVERY_N(INFO, 64) << publisher_.getTopic() << " latency: " << latency << " ms.";
}

int PCTransformNode::set_subscriber()
{
  std::string topic = pc_transform::get_input_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty input topic name is not allow. Please pass it with -input_topic in the command line";
    return EXIT_FAILURE;
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " subscribe " << topic;
  subscriber_ = node_handle_.subscribe(topic, /*queue size*/ 2, &PCTransformNode::callback, this);
  return EXIT_SUCCESS;
}

int PCTransformNode::set_publisher()
{
  std::string topic = pc_transform::get_output_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty output topic name is not allow. Please pass it with -output_topic in the command line";
    return EXIT_FAILURE;
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " publish compressed pointcloud at topic " << topic;
  publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(topic, /*queue size=*/2);
  heartbeat_publisher_ = node_handle_.advertise<std_msgs::Empty>(topic + "/heartbeat", /*queue size=*/2);

  return EXIT_SUCCESS;
}

void PCTransformNode::run()
{
  if ((set_subscriber() != EXIT_SUCCESS) || (set_publisher() != EXIT_SUCCESS))
  {
    return;
  }
  ros::AsyncSpinner spinner(/*thread_count*/ 1);
  spinner.start();
  ros::Rate r(1);
  while (ros::ok())
  {
    r.sleep();
  }
  spinner.stop();
}
};  // namespace pc_transform
