#ifndef __TRACK2D_NODE_H__
#define __TRACK2D_NODE_H__

#include "track2d.h"
#include "kalman_trackers.h"
#include "velocity.h"

#include <fstream>

namespace track2d
{
class Track2DNode
{
public:
  Track2DNode()
  {
  }

  ~Track2DNode()
  {
  }

  int run();

private:
  DISALLOW_COPY_AND_ASSIGN(Track2DNode);

  int in_source_ = 0;

  std_msgs::Header objs_header_;
  std_msgs::Header objs_header_prev_;

  KalmanTrackers KTs_;

  Velocity vel_;

  ros::NodeHandle nh_;

  ros::CallbackQueue queue_;  // custom callback queue

  ros::Publisher track2d_pub_;

  ros::Subscriber camera_sub_;
  void callback_camera(const msgs::DetectedObjectArray_SB::ConstPtr& input);

  msgs::DetectedObjectArray_SB track2d_obj_array;

  bool is_legal_dt_ = false;

  float dt_ = 0.f;

  void publish();

  void set_ros_params();
  void subscribe_and_advertise_topics();
};
}  // namespace track2d

#endif  // __TRACK2D_NODE_H__