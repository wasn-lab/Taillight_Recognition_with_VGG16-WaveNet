#ifndef __IMAGE_SAVER_NODE_IMPL_H__
#define __IMAGE_SAVER_NODE_IMPL_H__

#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <atomic>

class VideoSaverNodeImpl
{
private:
  // member variables
  image_transport::Subscriber im_subscriber_;
  ros::NodeHandle node_handle_;
  cv::VideoWriter video_;
  std::atomic_uint num_images_;

  // functions
  void image_callback(const sensor_msgs::ImageConstPtr& in_image_message);
  void subscribe();
  void save_video();

public:
  VideoSaverNodeImpl();
  ~VideoSaverNodeImpl();
  void run();
};

#endif  // __IMAGE_SAVER_NODE_IMPL_H__
