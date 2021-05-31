/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <unordered_map>
#include <string>
#include <glog/logging.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointField.h>
#include <std_msgs/Empty.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include "pc2_args_parser.h"
#include "pc2_compressor.h"
#include "filter_node.h"

namespace pc2_compressor
{

FilterNode::FilterNode()
{
}

FilterNode::~FilterNode() = default;

void FilterNode::callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  auto filtered_msg = filter_ouster64_pc2(msg);
  publisher_.publish(filtered_msg);

  std_msgs::Empty empty_msg;
  heartbeat_publisher_.publish(empty_msg);
}

int FilterNode::set_subscriber()
{
  std::string topic = pc2_compressor::get_input_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty input topic name is not allow. Please pass it with -input_topic in the command line";
    return EXIT_FAILURE;
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " subscribe " << topic;
  subscriber_ = node_handle_.subscribe(topic, /*queue size*/ 2, &FilterNode::callback, this);
  return EXIT_SUCCESS;
}

int FilterNode::set_publisher()
{
  std::string topic = pc2_compressor::get_output_topic();
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

void FilterNode::run()
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
};  // namespace pc2_compressor
