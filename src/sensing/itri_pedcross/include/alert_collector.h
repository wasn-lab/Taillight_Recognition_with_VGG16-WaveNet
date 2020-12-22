#ifndef __ALERT_COLLECTOR_H__
#define __ALERT_COLLECTOR_H__

#include "ros/ros.h"
#include <ros/spinner.h>
#include <ros/callback_queue.h>
#include "msgs/DetectedObjectArray.h"

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

namespace alr
{
class AlertCollector
{
public:
  AlertCollector()
  {
  }

  ~AlertCollector()
  {
  }

  // Functions
  void run();
  void alert_front_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void alert_left_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void alert_right_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void alert_fov30_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void alert_collector();
  void collect_and_publish();

  // All buffer components
  msgs::DetectedObjectArray alert_front;
  msgs::DetectedObjectArray alert_left;
  msgs::DetectedObjectArray alert_right;
  msgs::DetectedObjectArray alert_fov30;

  // ROS components
  ros::Publisher chatter_pub;

  // Variables
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_1;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_2;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_3;
  bool g_enable = false;
  bool g_trigger = false;
  int count;
};
}  // namespace alr

#endif  // __ALERT_COLLECTOR_H__
