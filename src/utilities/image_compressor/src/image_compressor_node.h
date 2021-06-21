#pragma once
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
  ros::Publisher heartbeat_publisher_;
  ros::NodeHandle node_handle_;
  int32_t num_compression_;
  int32_t latency_wrt_raw_in_ms_;

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
