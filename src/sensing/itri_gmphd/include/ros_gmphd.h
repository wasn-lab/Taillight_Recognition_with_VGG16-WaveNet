#ifndef __BAG_TO_MOT_H__
#define __BAG_TO_MOT_H__

#include "ros/ros.h"
#include <ros/spinner.h>
#include <ros/callback_queue.h>
#include <sensor_msgs/Image.h>
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

#include "stdafx.h"

#define PRINT_MESSAGE 1

namespace ped
{
class BagToMOT
{
public:
  BagToMOT()
  {
	  count=0;
	  tracker=new GMPHD_OGM();
  }

  ~BagToMOT()
  {
  }

  // Functions
  void run();
  void chatter_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void pedestrian_event();

  // ROS components
  ros::Publisher chatter_pub_front;
  ros::Time total_time;

  // Variables
  int count;
  GMPHD_OGM* tracker;
};
}  // namespace ped

#endif  // __PEDESTRIAN_EVENT_H__
