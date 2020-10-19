#ifndef __BAG_TO_MOT_H__
#define __BAG_TO_MOT_H__

#include "ros/ros.h"
#include <ros/spinner.h>
#include <ros/callback_queue.h>
#include "msgs/BoxPoint.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"
#include <gmphd_def.h>

#include "stdafx.h"

#define PRINT_MESSAGE 1

namespace ped
{
class ROSGMPHD
{
public:
  ROSGMPHD()
  {
    count = 0;
    tracker_front = new GMPHD_OGM();
    tracker_fov30 = new GMPHD_OGM();
    tracker_left_back = new GMPHD_OGM();
    tracker_right_back = new GMPHD_OGM();
  }

  ~ROSGMPHD()
  {
    delete tracker_front;
    delete tracker_fov30;
    delete tracker_left_back;
    delete tracker_right_back;
  }

  // Functions
  void run();
  void chatter_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void pedestrian_event();

  // ROS components
  ros::Publisher chatter_pub_front;
  ros::Publisher chatter_pub_fov30;
  ros::Publisher chatter_pub_left_back;
  ros::Publisher chatter_pub_right_back;
  ros::Time total_time;

  // Variables
  int count;
  GMPHD_OGM* tracker_front;
  GMPHD_OGM* tracker_fov30;
  GMPHD_OGM* tracker_left_back;
  GMPHD_OGM* tracker_right_back;
};
}  // namespace ped

#endif  // __PEDESTRIAN_EVENT_H__
