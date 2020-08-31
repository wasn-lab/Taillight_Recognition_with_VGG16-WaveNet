#ifndef __BAG_TO_MOT_H__
#define __BAG_TO_MOT_H__

#include "ros/ros.h"
#include <ros/spinner.h>
#include <ros/callback_queue.h>
#include <sensor_msgs/Image.h>
#include "msgs/BoxPoint.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"

#include <opencv2/opencv.hpp>  // opencv general include file
#include <cv_bridge/cv_bridge.h>
#include <gmphd_def.h>
#include <boost/circular_buffer.hpp>
// C++ std library dependencies
#include <fstream>

#define USE_2D_FOR_ALARM 0
#define DUMP_LOG 0
#define PRINT_MESSAGE 1

namespace ped
{
class BagToMOT
{
public:
  BagToMOT()
  {
  }

  ~BagToMOT()
  {
  }

  // Functions
  void run();
  void cache_image_callback(const sensor_msgs::Image::ConstPtr& msg);
  void chatter_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void pedestrian_event();

  // All buffer components
  boost::circular_buffer<std::pair<ros::Time, cv::Mat>> image_cache;
  std::vector<std::string> ped_info;
  int buffer_size = 60;

  // ROS components
  ros::Publisher chatter_pub;
  ros::Publisher box_pub;
  ros::Publisher alert_pub;
  ros::Time total_time;

  // Variables
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_1;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_2;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_3;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_4;
  bool g_enable = false;
  bool g_trigger = false;
  int count;
  std::ofstream file;
  std::ofstream file_1;
  std::ofstream file_2;
  std::ofstream file_3;
  std::ofstream file_4;
  std::ofstream file_5;
  std::ofstream file_6;
  std::ofstream file_7;
  std::ofstream file_8;

  // Setup variables
  const double scaling_ratio_width = 0.3167;
  const double scaling_ratio_height = 0.3179;

  // ROS param
  bool show_probability = true;
  int input_source = 3;
  double max_distance = 50;
  double danger_zone_distance = 2;
  bool use_2d_for_alarm = false;
};
}  // namespace ped

#endif  // __PEDESTRIAN_EVENT_H__
