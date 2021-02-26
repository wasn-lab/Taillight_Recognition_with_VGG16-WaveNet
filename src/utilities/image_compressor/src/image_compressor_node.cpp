#include <thread>
#include <glog/logging.h>
#include <std_msgs/Empty.h>
#include <sensor_msgs/CompressedImage.h>
#include "image_compressor_args_parser.h"
#include "image_compressor_node.h"
#include "image_compressor.h"

namespace image_compressor
{
ImageCompressorNode::ImageCompressorNode() : num_compression_(0)
{
}
ImageCompressorNode::~ImageCompressorNode() = default;

void ImageCompressorNode::callback(const sensor_msgs::ImageConstPtr& msg)
{
  num_compression_ += 1;
  if (image_compressor::use_threading())
  {
    std::thread t(&ImageCompressorNode::publish, this, msg);
    t.detach();
  }
  else
  {
    publish(msg);
  }
}

void ImageCompressorNode::publish(const sensor_msgs::ImageConstPtr& msg)
{
  auto cmpr_msg_ptr = compress_msg(msg);
  {
    std::lock_guard<std::mutex> lk(mu_publisher_);
    publisher_.publish(cmpr_msg_ptr);
  }
  heartbeat_publisher_.publish(std_msgs::Empty{});
}

int ImageCompressorNode::set_subscriber()
{
  std::string topic = image_compressor::get_input_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty input topic name is not allow. Please pass it with -input_topic in the command line";
    return EXIT_FAILURE;
  }
  subscriber_ = node_handle_.subscribe(topic, 2, &ImageCompressorNode::callback, this);
  return EXIT_SUCCESS;
}

int ImageCompressorNode::set_publisher()
{
  std::string topic = image_compressor::get_output_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty output topic name is not allow. Please pass it with -output_topic in the command line";
    return EXIT_FAILURE;
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " publish compressed image at topic " << topic;
  publisher_ = node_handle_.advertise<sensor_msgs::CompressedImage>(topic, /*queue size=*/2);
  heartbeat_publisher_ = node_handle_.advertise<std_msgs::Empty>(topic + "/heartbeat", 1);
  return EXIT_SUCCESS;
}

void ImageCompressorNode::run()
{
  CHECK(set_subscriber() == EXIT_SUCCESS);
  CHECK(set_publisher() == EXIT_SUCCESS);

  ros::AsyncSpinner spinner(/*threads_count*/ 1);
  spinner.start();
  ros::Rate r(1);
  while (ros::ok())
  {
    LOG(INFO) << "compress " << image_compressor::get_input_topic() << " in 1s: " << num_compression_;
    num_compression_ = 0;
    r.sleep();
  }
  spinner.stop();
}
};  // namespace image_compressor
