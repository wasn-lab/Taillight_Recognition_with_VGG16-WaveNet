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
  std::vector<sensor_msgs::Image> imageCache;
  std::vector<cv::Mat> imageCacheMat;
  unsigned int buffer_size = 60;
  void chatter_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void pedestrian_event();
  std::vector<cv::Point> get_openpose_keypoint(cv::Mat input_image);
  float load_model(float u, float v, float w, float h);
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
