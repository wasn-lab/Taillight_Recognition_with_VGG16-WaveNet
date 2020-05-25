#pragma once

#include "deeplab_segmenter.h"
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

namespace deeplab
{
class DeeplabNodeImpl
{
  // ROS
  image_transport::Subscriber image_subscriber_;
  ros::Publisher image_publisher_;
  ros::NodeHandle node_handle_;

  // NN
  DeeplabSegmenter segmenter_;

  // private functions
  void gracefully_stop_threads();
  void subscribe_topics();
  void advertise_topics();

public:
  DeeplabNodeImpl();
  ~DeeplabNodeImpl();
  void image_callback(const sensor_msgs::ImageConstPtr& msg_in);
  void run(int argc, char* argv[]);
};
};  // namespace deeplab
