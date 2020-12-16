#include <thread>
#include <glog/logging.h>
#include <sensor_msgs/CompressedImage.h>
#include "image_compressor.h"
#include "image_compressor_args_parser.h"
#include "image_decompressor_node.h"

namespace image_compressor
{
ImageDecompressorNode::ImageDecompressorNode() : num_compression_(0)
{
}
ImageDecompressorNode::~ImageDecompressorNode() = default;

void ImageDecompressorNode::callback(const sensor_msgs::CompressedImageConstPtr msg)
{
  num_compression_ += 1;
  if (image_compressor::use_threading())
  {
    std::thread t(&ImageDecompressorNode::publish, this, msg);
    t.detach();
  }
  else
  {
    publish(msg);
  }
}

void ImageDecompressorNode::publish(const sensor_msgs::CompressedImageConstPtr msg)
{
  auto decmpr_msg_ptr = decompress_msg(msg);
  {
    std::lock_guard<std::mutex> lk(mu_publisher_);
    publisher_.publish(decmpr_msg_ptr);
  }
}

int ImageDecompressorNode::set_subscriber()
{
  std::string topic = image_compressor::get_input_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty input topic name is not allow. Please pass it with -input_topic in the command line";
    return EXIT_FAILURE;
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
  ros::AsyncSpinner spinner(/*num_threads=*/1);
  spinner.start();
  ros::Rate r(1);
  while (ros::ok())
  {
    LOG(INFO) << "decompress " << image_compressor::get_input_topic() << " in 1s: " << num_compression_;
    num_compression_ = 0;
    r.sleep();
  }
  spinner.stop();
}
};  // namespace image_compressor
