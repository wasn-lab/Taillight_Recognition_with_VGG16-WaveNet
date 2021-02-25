/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <thread>
#include <glog/logging.h>
#include <pcl_ros/point_cloud.h>
#include "pc2_args_parser.h"
#include "pc2_decompressor_node.h"
#include "pc2_compressor.h"
#include "pc2_compression_format.h"

namespace pc2_compressor
{
PC2DecompressorNode::PC2DecompressorNode()
{
}
PC2DecompressorNode::~PC2DecompressorNode() = default;

void PC2DecompressorNode::callback_v1(const msgs::CompressedPointCloudConstPtr& msg)
{
  msgs::CompressedPointCloud2Ptr msg_v2{ new msgs::CompressedPointCloud2 };
  msg_v2->header = msg->header;
  msg_v2->data = msg->data;
  msg_v2->compression_format = compression_format::lzf;
  publisher_.publish(pc2_compressor::decompress_msg(msg_v2));
}

void PC2DecompressorNode::callback_v2(const msgs::CompressedPointCloud2ConstPtr& msg)
{
  publisher_.publish(pc2_compressor::decompress_msg(msg));
}

static bool is_topic_published(const std::string& topic, int* use_v1)
{
  ros::master::V_TopicInfo master_topics;
  ros::master::getTopics(master_topics);

  for (auto& master_topic : master_topics)
  {
    if (master_topic.name == topic)
    {
      if (master_topic.datatype == "msgs/CompressedPointCloud")
      {
        *use_v1 = 1;
      }
      return true;
    }
  }
  return false;
}

int PC2DecompressorNode::set_subscriber()
{
  std::string topic = pc2_compressor::get_input_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty input topic name is not allow. Please pass it with -input_topic in the command line";
    return EXIT_FAILURE;
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " subscribe " << topic;
  int32_t use_v1 = 0;  // for backward compatibility
  while (ros::ok() && !is_topic_published(topic, &use_v1))
  {
    LOG(INFO) << "wait 1 second for topic " << topic;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  if (use_v1 != 0)
  {
    subscriber_ = node_handle_.subscribe(topic, /*queue_size*/ 2, &PC2DecompressorNode::callback_v1, this);
  }
  else
  {
    subscriber_ = node_handle_.subscribe(topic, /*queue_size*/ 2, &PC2DecompressorNode::callback_v2, this);
  }
  return EXIT_SUCCESS;
}

int PC2DecompressorNode::set_publisher()
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
  return EXIT_SUCCESS;
}

void PC2DecompressorNode::run()
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
