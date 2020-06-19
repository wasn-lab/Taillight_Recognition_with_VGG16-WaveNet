#ifndef __BAG_TO_MOT_H__
#define __BAG_TO_MOT_H__

#include "ros/ros.h"
#include <ros/spinner.h>
#include <ros/callback_queue.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/TransformStamped.h>
#include "msgs/BoxPoint.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/PedObject.h"
#include "msgs/PedObjectArray.h"

#include <opencv2/opencv.hpp>  // opencv general include file
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>  // opencv machine learning include file
#include <cv_bridge/cv_bridge.h>
#include <gmphd_def.h>
#include <map>
#include <boost/circular_buffer.hpp>
// C++ std library dependencies
#include <chrono>  // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <thread>  // std::this_thread
#include <fstream>
#include <sys/ioctl.h>
#include <unistd.h>

#define USE_2D_FOR_ALARM 0
#define DUMP_LOG 0
#define PRINT_MESSAGE 1

#define M_PIl 3.141592653589793238462643383279502884L /* pi */

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
  void cache_crop_image_callback(const sensor_msgs::Image::ConstPtr& msg);
  void chatter_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void pedestrian_event();

  // All buffer components
  boost::circular_buffer<std::pair<ros::Time, cv::Mat>> image_cache;
  boost::circular_buffer<std::pair<ros::Time, cv::Mat>> crop_image_cache;
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
  struct winsize terminal_size;

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
