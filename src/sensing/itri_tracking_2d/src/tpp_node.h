#ifndef __TPP_NODE_H__
#define __TPP_NODE_H__

#include "tpp.h"
#include "kalman_trackers.h"
#include "velocity.h"

#include <fstream>

namespace tpp
{
class TPPNode
{
public:
  TPPNode()
  {
  }

  ~TPPNode()
  {
  }

  int run();

private:
  DISALLOW_COPY_AND_ASSIGN(TPPNode);

  int in_source_ = 0;

  std_msgs::Header objs_header_;
  std_msgs::Header objs_header_prev_;

  KalmanTrackers KTs_;

  Velocity vel_;

  ros::NodeHandle nh_;

  ros::CallbackQueue queue_;  // custom callback queue

  ros::Publisher track2d_pub_;

  ros::Subscriber camera_sub_;
  void callback_camera(const msgs::DetectedObjectArray::ConstPtr& input);

  msgs::DetectedObjectArray track2d_obj_array;

  bool is_legal_dt_ = false;

  float dt_ = 0.f;

  void publish();

  void set_ros_params();
  void subscribe_and_advertise_topics();
};
}  // namespace tpp

#endif  // __TPP_NODE_H__