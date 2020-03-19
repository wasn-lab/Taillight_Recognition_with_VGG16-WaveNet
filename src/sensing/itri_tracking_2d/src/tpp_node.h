#ifndef __TPP_NODE_H__
#define __TPP_NODE_H__

#include "tpp.h"
#include "kalman_trackers.h"
#include "velocity.h"
#include "tpp_args_parser.h"
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

  int in_source_ = get_in_source();
  bool use_ego_speed_ = get_ego_speed();

  std_msgs::Header objs_header_;
  std_msgs::Header objs_header_prev_;

  std::vector<msgs::DetectedObject> pp_objs_;

#if VIRTUAL_INPUT
  double gt_x_ = 0.;
  double gt_y_ = 0.;
#endif

  KalmanTrackers KTs_;

  EgoParam ego_x_m_;
  EgoParam ego_y_m_;
  EgoParam ego_heading_rad_;

  Velocity vel_;

  PathPredict pp_;

  ros::NodeHandle nh_;
  ros::NodeHandle nh2_;

  // custom callback queue
  ros::CallbackQueue queue_;

  ros::Publisher pp_pub_;

  ros::Subscriber fusion_sub_;
  void callback_fusion(const msgs::DetectedObjectArray::ConstPtr& input);

#if TTC_TEST
  unsigned int seq_ = 0;
  unsigned int seq_cb_ = 0;

  ros::Subscriber seq_sub_;
  void callback_seq(const std_msgs::Int32::ConstPtr& input);
#endif

  bool is_legal_dt_ = false;
  double loop_begin = 0.;    // seconds
  double loop_elapsed = 0.;  // seconds

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

  // compute DetectedObject.relSpeed:
  // i.e., speed of relative velocity on relative coordinate projection onto object-to-ego-vehicle vector
  float compute_relative_speed_obj2ego(const Vector3_32 rel_v_rel, const MyPoint32 obj_rel);

  float compute_radar_absolute_velocity(const float radar_speed_rel, const float box_center_x_abs,
                                        const float box_center_y_abs);

  void compute_velocity_kalman();

  void push_to_vector(BoxCenter a, std::vector<MyPoint32>& b);
  void publish_tracking();

  void set_ros_params();
  void subscribe_and_advertise_topics();
  void get_current_ego_data_main();
  void get_current_ego_data(const tf2_ros::Buffer& tf_buffer, const ros::Time fusion_stamp);

  void save_output_to_txt(const std::vector<msgs::DetectedObject>& objs);
#if TTC_TEST
  float closest_distance_of_obj_pivot(const msgs::DetectedObject& obj);
  void save_ttc_to_csv(std::vector<msgs::DetectedObject>& objs);
#endif
};
}  // namespace tpp

#endif  // __TPP_NODE_H__