#include <thread>
#include <chrono>
#include <glog/logging.h>
#include <sensor_msgs/CompressedImage.h>
#include "image_compressor.h"
#include "image_compressor_args_parser.h"
#include "image_decompressor_node.h"

namespace image_compressor
{
ImageDecompressorNode::ImageDecompressorNode() : num_decompression_(0)
{
}
ImageDecompressorNode::~ImageDecompressorNode() = default;

void ImageDecompressorNode::callback(const sensor_msgs::CompressedImageConstPtr& msg)
{
  num_decompression_ += 1;
  publish(msg);
}

void ImageDecompressorNode::publish(const sensor_msgs::CompressedImageConstPtr& msg)
{
  auto decmpr_msg_ptr = decompress_msg(msg);
  {
    std::lock_guard<std::mutex> lk(mu_publisher_);
    publisher_.publish(decmpr_msg_ptr);
  }
}

static bool is_topic_published(const std::string& topic)
{
  ros::master::V_TopicInfo master_topics;
  ros::master::getTopics(master_topics);

  for (auto& master_topic : master_topics)
  {
    if (master_topic.name == topic)
    {
      return true;
    }
  }
  return false;
}

int ImageDecompressorNode::set_subscriber()
{
  std::string topic = image_compressor::get_input_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty input topic name is not allow. Please pass it with -input_topic in the command line";
    return EXIT_FAILURE;
  }
  while (ros::ok() && !is_topic_published(topic))
  {
    LOG(INFO) << "wait 1 second for topic " << topic;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " subscribe " << topic;
  subscriber_ = node_handle_.subscribe(topic, 2, &ImageDecompressorNode::callback, this);
  return EXIT_SUCCESS;
}

int ImageDecompressorNode::set_publisher()
{
  std::string topic = image_compressor::get_output_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty output topic name is not allow. Please pass it with -output_topic in the command line";
    return EXIT_FAILURE;
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " publish decompressed image at topic " << topic;
  publisher_ = node_handle_.advertise<sensor_msgs::Image>(topic, /*queue size=*/2);
  return EXIT_SUCCESS;
}

void ImageDecompressorNode::run()
{
  CHECK(set_subscriber() == EXIT_SUCCESS);
  CHECK(set_publisher() == EXIT_SUCCESS);
  set_use_threading(false);
  ros::AsyncSpinner spinner(/*thread_count*/1);
  spinner.start();
  ros::Rate r(1);
  while (ros::ok())
  {
    LOG(INFO) << "decompress " << image_compressor::get_input_topic() << " in 1s: " << num_decompression_;
    num_decompression_ = 0;
    r.sleep();
  }
  spinner.stop();
}
};  // namespace image_compressor
