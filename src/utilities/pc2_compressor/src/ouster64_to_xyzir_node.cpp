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
#include "point_os1.h"
#include "point_xyzir.h"

namespace pc2_compressor
{
constexpr bool OUSTER64_XYZIR_NODE_DEBUG = false;

static bool equal_xyzir_fields(const sensor_msgs::PointCloud2ConstPtr& ouster64_msg,
                               const sensor_msgs::PointCloud2ConstPtr& xyzir_msg)
{
  if (ouster64_msg->header.seq != xyzir_msg->header.seq)
  {
    LOG(INFO) << "unmatched header seq!";
    return false;
  }
  if (ouster64_msg->header.stamp != xyzir_msg->header.stamp)
  {
    LOG(INFO) << "unmatched header stamp!";
    return false;
  }
  if (ouster64_msg->header.frame_id != xyzir_msg->header.frame_id)
  {
    LOG(INFO) << "unmatched header frame_id" << ouster64_msg->header.frame_id
              << ", xyzir: " << xyzir_msg->header.frame_id;
    return false;
  }

  if (ouster64_msg->width != xyzir_msg->width)
  {
    LOG(INFO) << "unmatched width! ouster64: " << ouster64_msg->width << ", xyzir: " << xyzir_msg->width;
    return false;
  }

  if (ouster64_msg->height != xyzir_msg->height)
  {
    LOG(INFO) << "unmatched height! ouster64: " << ouster64_msg->height << ", xyzir: " << xyzir_msg->height;
    return false;
  }

  if (ouster64_msg->is_bigendian != xyzir_msg->is_bigendian)
  {
    LOG(INFO) << "unmatched is_bigendian!";
    return false;
  }

  if (ouster64_msg->is_dense != xyzir_msg->is_dense)
  {
    LOG(INFO) << "unmatched is_dense!";
    return false;
  }

  pcl::PointCloud<ouster_ros::OS1::PointOS1> ouster64_cloud;
  pcl::PointCloud<ouster_ros::OS1::PointXYZIR> xyzir_cloud;
  pcl::fromROSMsg(*ouster64_msg, ouster64_cloud);  // Emits Failed to find match for field 'noise'.
  pcl::fromROSMsg(*xyzir_msg, xyzir_cloud);
  const size_t npoints = ouster64_cloud.points.size();
  for (size_t i = 0; i < npoints; i++)
  {
    const auto& op = ouster64_cloud.points[i];
    const auto& xp = xyzir_cloud.points[i];
    if (op.x != xp.x || op.y != xp.y || op.z != xp.z || op.intensity != xp.intensity || op.ring != xp.ring)
    {
      LOG(INFO) << "Point " << i << " is unmatched! ";
      return false;
    }
  }
  LOG_EVERY_N(INFO, 60) << "Consistent xyzir data";
  return true;
}

void Ouster64ToXYZIRNode::callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  msgs_per_second_++;
  auto xyzir_msg = ouster64_to_xyzir(msg);
  xyzir_publisher_.publish(xyzir_msg);

  if (OUSTER64_XYZIR_NODE_DEBUG)
  {
    // Debug code.
    CHECK(equal_xyzir_fields(msg, xyzir_msg));
  }

  ros::Time now = ros::Time::now();
  latency_wrt_raw_in_ms_ = (now.sec - msg->header.stamp.sec) * 1000 + (now.nsec - msg->header.stamp.nsec) / 1000000;

  std_msgs::Empty empty_msg;
  xyzir_heartbeat_publisher_.publish(empty_msg);
  raw_heartbeat_publisher_.publish(empty_msg);
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

  raw_heartbeat_publisher_ =
      node_handle_.advertise<std_msgs::Empty>(pc2_compressor::get_input_topic() + "/heartbeat", /*queue size=*/2);
  return EXIT_SUCCESS;
}

void Ouster64ToXYZIRNode::run()
{
  if ((set_subscriber() != EXIT_SUCCESS) || (set_publisher() != EXIT_SUCCESS))
  {
    return;
  }
  msgs_per_second_ = 0;
  latency_wrt_raw_in_ms_ = 0;
  ros::AsyncSpinner spinner(/*thread_count*/ 1);
  spinner.start();
  ros::Rate r(1);
  while (ros::ok())
  {
    LOG(INFO) << xyzir_publisher_.getTopic() << " fps: " << msgs_per_second_ << ", latency: " << latency_wrt_raw_in_ms_
              << " ms";
    msgs_per_second_ = 0;

    r.sleep();
  }
  spinner.stop();
}
};  // namespace pc2_compressor
