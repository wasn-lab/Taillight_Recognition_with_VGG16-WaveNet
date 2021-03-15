/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <unordered_map>
#include <string>
#include <thread>
#include <glog/logging.h>
#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud.h>
#include "msgs/CompressedPointCloud2.h"
#include "pc2_args_parser.h"
#include "pc2_compressor_node.h"
#include "pc2_compressor.h"
#include <std_msgs/Empty.h>

namespace pc2_compressor
{
const std::unordered_map<std::string, compression_format> FMT_ENUMS{
  { "snappy", compression_format::snappy },
  { "lzf", compression_format::lzf },
  { "none", compression_format::none },
  { "zlib", compression_format::zlib },
};

PC2CompressorNode::PC2CompressorNode()
{
  const auto fmt = get_compression_format();
  auto it = FMT_ENUMS.find(fmt);
  if (it != FMT_ENUMS.end())
  {
    cmpr_fmt_ = it->second;
  }
  else
  {
    CHECK(false) << "Unknown compression format";
  }
  LOG(INFO) << "Use compression format: " << fmt << " enum: " << cmpr_fmt_;
}

PC2CompressorNode::~PC2CompressorNode() = default;

void PC2CompressorNode::callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  if (pc2_compressor::use_threading())
  {
    std::thread t(&PC2CompressorNode::publish_compressed_pc2, this, msg);
    t.detach();
  }
  else
  {
    publish_compressed_pc2(msg);
  }
}

void PC2CompressorNode::publish_compressed_pc2(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  auto cmpr_msg = compress_msg(msg, cmpr_fmt_);
  if (pc2_compressor::use_threading())
  {
    std::lock_guard<std::mutex> lk(mu_publisher_);
    publisher_.publish(cmpr_msg);

    std_msgs::Empty empty_msg;
    heartbeat_publisher_.publish(empty_msg);
    
  }
  else
  {
    publisher_.publish(cmpr_msg);

    std_msgs::Empty empty_msg;
    heartbeat_publisher_.publish(empty_msg);
  }

  if (pc2_compressor::should_verify_decompressed_data())
  {
    auto decmpr_msg = pc2_compressor::decompress_msg(cmpr_msg);
    CHECK(is_equal_pc2(decmpr_msg, msg));
  }
}

int PC2CompressorNode::set_subscriber()
{
  std::string topic = pc2_compressor::get_input_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty input topic name is not allow. Please pass it with -input_topic in the command line";
    return EXIT_FAILURE;
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " subscribe " << topic;
  subscriber_ = node_handle_.subscribe(topic, /*queue size*/ 2, &PC2CompressorNode::callback, this);
  return EXIT_SUCCESS;
}

int PC2CompressorNode::set_publisher()
{
  std::string topic = pc2_compressor::get_output_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty output topic name is not allow. Please pass it with -output_topic in the command line";
    return EXIT_FAILURE;
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " publish compressed pointcloud at topic " << topic;
  publisher_ = node_handle_.advertise<msgs::CompressedPointCloud2>(topic, /*queue size=*/2);
  heartbeat_publisher_ = node_handle_.advertise<std_msgs::Empty>(topic+"/heartbeat", /*queue size=*/2);
  return EXIT_SUCCESS;
}

void PC2CompressorNode::run()
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
