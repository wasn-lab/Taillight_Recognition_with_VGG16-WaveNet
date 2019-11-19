#ifndef __PEDESTRIAN_EVENT_H__
#define __PEDESTRIAN_EVENT_H__

#include "ros/ros.h"
#include <ros/spinner.h>
#include <ros/callback_queue.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/Image.h>
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/PedObject.h"
#include "msgs/PedObjectArray.h"
#include <opencv2/opencv.hpp>  // opencv general include file
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>  // opencv machine learning include file
#include <opencv2/imgcodecs.hpp>
// #include <opencv2/gpu/gpu.hpp>
// #include <caffe/caffe.hpp>
#include <ped_def.h>
#include <cv_bridge/cv_bridge.h>
#include <map>

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

#define M_PIl 3.141592653589793238462643383279502884L /* pi */

namespace ped
{
class PedestrianEvent
{
public:
  PedestrianEvent()
  {
  }

  ~PedestrianEvent()
  {
  }

  void run();
  void cache_image_callback(const sensor_msgs::Image::ConstPtr& msg);
  std::deque<std::pair<ros::Time, cv::Mat> > imageCache;
  unsigned int buffer_size = 60;
  void chatter_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void pedestrian_event();
  std::vector<cv::Point> get_openpose_keypoint(cv::Mat input_image);
  double crossing_predict(double bb_x1, double bb_y1, double bb_x2, double bb_y2, std::vector<cv::Point> keypoint);
  double* get_triangle_angle(double x1, double y1, double x2, double y2, double x3, double y3);
  double get_distance2(double x1, double y1, double x2, double y2);
  double get_angle2(double x1, double y1, double x2, double y2);
  double predict_rf(cv::Mat input_data);
  cv::dnn::Net net_openpose;
  cv::Ptr<cv::ml::RTrees> rf;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner;
  ros::Publisher chatter_pub;
  ros::Publisher box_pub;
  ros::Publisher pose_pub;
  ros::Time total_time;
  bool g_enable = false;
  bool g_trigger = false;
  int count;
};
}  // namespace ped

#endif  // __PEDESTRIAN_EVENT_H__
