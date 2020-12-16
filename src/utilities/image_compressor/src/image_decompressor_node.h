#pragma once
#include <mutex>
#include <atomic>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>

namespace image_compressor
{
class ImageDecompressorNode
{
private:
  // member variables
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  ros::NodeHandle node_handle_;
  std::mutex mu_publisher_;  // guard publisher_
  std::atomic_int num_compression_;

  // functions
  void callback(const sensor_msgs::CompressedImageConstPtr& msg);
  void publish(const sensor_msgs::CompressedImageConstPtr& msg);
  int set_subscriber();
  int set_publisher();

public:
  ImageDecompressorNode();
  ~ImageDecompressorNode();
  void run();
};
};  // namespace image_compressor
