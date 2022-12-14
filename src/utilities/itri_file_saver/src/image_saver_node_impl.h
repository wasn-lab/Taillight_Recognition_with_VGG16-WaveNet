#pragma once

#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

#include <cv_bridge/cv_bridge.h>

namespace image_saver
{
class ImageSaverNodeImpl
{
private:
  // member variables
  image_transport::Subscriber im_subscriber_;
  ros::NodeHandle node_handle_;

  // functions
  void image_callback(const sensor_msgs::ImageConstPtr& in_image_message);
  void subscribe();
  void save(const cv_bridge::CvImageConstPtr& cv_ptr, int sec, int nsec);

public:
  ImageSaverNodeImpl();
  ~ImageSaverNodeImpl();
  void run();
};
};  // namespace image_saver
