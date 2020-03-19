#ifndef __TPP_NODE_H__
#define __TPP_NODE_H__

#include "tpp.h"
#include "kalman_trackers.h"
#include "velocity.h"
#include "ros_params_parser.h"
#include "ego_param.h"

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
  ros::NodeHandle nh2_;

  // custom callback queue
  ros::CallbackQueue queue_;

  ros::Publisher track2d_pub_;

  ros::Subscriber camera_sub_;
  void callback_camera(const msgs::DetectedObjectArray::ConstPtr& input);

  bool is_legal_dt_ = false;

  float dt_ = 0.f;
  float ego_x_abs_ = 0.f;
  float ego_y_abs_ = 0.f;
  float ego_z_abs_ = 0.f;
  float ego_heading_ = 0.f;
  float ego_dx_abs_ = 0.f;
  float ego_dy_abs_ = 0.f;
  double ego_speed_kmph_ = 0.;
  double ego_velx_abs_kmph_ = 0.;
  double ego_vely_abs_kmph_ = 0.;

  void init_velocity(msgs::TrackInfo& track);

  void publish_tracking();

  void set_ros_params();
  void subscribe_and_advertise_topics();
};
}  // namespace tpp

#endif  // __TPP_NODE_H__