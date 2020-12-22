#ifndef __TRACKING_VIEW_H__
#define __TRACKING_VIEW_H__

#include "ros/ros.h"
#include <ros/spinner.h>
#include <ros/callback_queue.h>
#include <sensor_msgs/Image.h>
#include "msgs/BoxPoint.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"

#include <opencv2/opencv.hpp>  // opencv general include file
#include <cv_bridge/cv_bridge.h>
#include <boost/circular_buffer.hpp>
// C++ std library dependencies
#include <chrono>  // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <thread>  // std::this_thread
#include <fstream>

#define PRINT_MESSAGE 0
#define USE_GLOG 1
#if USE_GLOG
#include "glog/logging.h"
#define LOG_INFO LOG(INFO)
#define LOG_WARNING LOG(WARNING)
#define LOG_ERROR LOG(ERROR)
#define LOG_FATAL LOG(FATAL)
#else
#define LOG_INFO std::cout
#define LOG_WARNING std::cout
#define LOG_ERROR std::cout
#define LOG_FATAL std::cout
#endif

namespace tra
{
class TrackingView
{
public:
  TrackingView()
  {
  }

  ~TrackingView()
  {
  }

  // Functions
  void run();
  void cache_image_callback(const sensor_msgs::Image::ConstPtr& msg);
  void detection_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void tracking_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void draw_tracking_with_detection();
  void pedestrian_event();

  // All buffer components
  boost::circular_buffer<std::pair<ros::Time, cv::Mat>> image_cache;
  msgs::DetectedObjectArray latest_detection;
  msgs::DetectedObjectArray latest_tracking;
  std::string delay_from_camera = "NA";
  int buffer_size = 60;

  // ROS components
  ros::Publisher chatter_pub;

  // Variables
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_1;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_2;
  bool g_enable = false;
  bool g_trigger = false;
  int count;

  // Setup variables
  const double scaling_ratio_width = 0.3167;
  const double scaling_ratio_height = 0.3179;
};
}  // namespace tra

#endif  // __TRACKING_VIEW_H__
