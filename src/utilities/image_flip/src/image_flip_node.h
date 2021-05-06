#pragma once
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

namespace image_flip
{
class ImageFlipNode
{
private:
  // member variables
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  ros::NodeHandle node_handle_;

  // functions
  void callback(const sensor_msgs::ImageConstPtr& msg);
  int set_subscriber();
  int set_publisher();

public:
  ImageFlipNode();
  ~ImageFlipNode();
  void run();
};
};  // namespace image_flip
