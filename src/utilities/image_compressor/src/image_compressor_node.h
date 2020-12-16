#pragma once
#include <mutex>
#include <atomic>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>

namespace image_compressor
{
class ImageCompressorNode
{
private:
  // member variables
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  ros::NodeHandle node_handle_;
  std::mutex mu_publisher_;  // guard publisher_
  std::atomic_int num_compression_;

  // functions
  void callback(const sensor_msgs::ImageConstPtr& msg);
  void publish(const sensor_msgs::ImageConstPtr& msg);
  int set_subscriber();
  int set_publisher();

public:
  ImageCompressorNode();
  ~ImageCompressorNode();
  void run();
};
};  // namespace image_compressor
