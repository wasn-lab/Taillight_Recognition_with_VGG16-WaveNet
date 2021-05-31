/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <unordered_map>
#include <string>
#include <glog/logging.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointField.h>
#include "pc2_args_parser.h"
#include "filter_node.h"
#include <std_msgs/Empty.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

namespace pc2_compressor
{
sensor_msgs::PointCloud2Ptr filter_ouster64_pc2(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  // Ouster has 9 fields: x y z intensity t reflectivity ring noise range
  // We only needs 5 fields: x, y, z, intensity and ring.
  constexpr int32_t msg_ring_field_idx = 6;
  constexpr int32_t res_ring_field_idx = 4;
  sensor_msgs::PointCloud2Ptr res{ new sensor_msgs::PointCloud2 };
  res->fields.reserve(res_ring_field_idx + 1);
  for (int32_t i = 0; i < res_ring_field_idx; i++)
  {
    res->fields.push_back(msg->fields[i]);
  }
  res->fields.push_back(msg->fields[msg_ring_field_idx]);

  res->point_step = 0;
  for (const auto& field : res->fields)
  {
    res->point_step += pcl::getFieldSize(field.datatype);
  }
  // dest_front_size = 16 (x, y, z, intensity: 4 bytes each)
  const int32_t ring_size = pcl::getFieldSize(res->fields[res_ring_field_idx].datatype);
  const int32_t dest_front_size = res->point_step - ring_size;
  res->fields[res_ring_field_idx].offset = dest_front_size;

  const int32_t num_points = msg->width * msg->height;
  const int32_t msg_ring_field_offset = msg->fields[msg_ring_field_idx].offset;
  res->data.resize(num_points * res->point_step);
  for (int32_t i = 0, src_offset = 0, dest_offset = 0; i < num_points;
       i++, src_offset += msg->point_step, dest_offset += res->point_step)
  {
    memcpy(&(res->data[dest_offset]), &(msg->data[src_offset]), dest_front_size);
    memcpy(&(res->data[dest_offset + dest_front_size]), &(msg->data[src_offset + msg_ring_field_offset]), ring_size);
  }

  res->header = msg->header;
  res->width = msg->width;
  res->height = msg->height;
  res->is_bigendian = msg->is_bigendian;
  res->row_step = msg->width * res->point_step;
  res->is_dense = msg->is_dense;
  return res;
}

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
