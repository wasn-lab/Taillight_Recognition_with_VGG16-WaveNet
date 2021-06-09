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
#include "ouster64_to_xyzir_node.h"

namespace pc2_compressor
{
void Ouster64ToXYZIRNode::callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  msgs_per_second_++;
  auto filtered_msg = ouster64_to_xyzir(msg);
  xyzir_publisher_.publish(filtered_msg);

  std_msgs::Empty empty_msg;
  xyzir_heartbeat_publisher_.publish(empty_msg);
}

int Ouster64ToXYZIRNode::set_subscriber()
{
  std::string topic = pc2_compressor::get_input_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty input topic name is not allow. Please pass it with -input_topic in the command line";
    return EXIT_FAILURE;
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " subscribe " << topic;
  subscriber_ = node_handle_.subscribe(topic, /*queue size*/ 2, &Ouster64ToXYZIRNode::callback, this);
  return EXIT_SUCCESS;
}

int Ouster64ToXYZIRNode::set_publisher()
{
  std::string xyzir_topic = pc2_compressor::get_input_topic() + "/xyzir";

  LOG(INFO) << ros::this_node::getName() << ":"
            << " publish compressed pointcloud at topic " << xyzir_topic;
  xyzir_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(xyzir_topic, /*queue size=*/2);
  xyzir_heartbeat_publisher_ = node_handle_.advertise<std_msgs::Empty>(xyzir_topic + "/heartbeat", /*queue size=*/2);
  return EXIT_SUCCESS;
}

void Ouster64ToXYZIRNode::run()
{
  if ((set_subscriber() != EXIT_SUCCESS) || (set_publisher() != EXIT_SUCCESS))
  {
    return;
  }
  msgs_per_second_ = 0;
  ros::AsyncSpinner spinner(/*thread_count*/ 1);
  spinner.start();
  ros::Rate r(1);
  while (ros::ok())
  {
    LOG(INFO) << "#msgs per second: " << msgs_per_second_;
    msgs_per_second_ = 0;

    r.sleep();
  }
  spinner.stop();
}
};  // namespace pc2_compressor
