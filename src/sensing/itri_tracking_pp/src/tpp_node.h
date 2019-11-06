#ifndef __TPP_NODE_H__
#define __TPP_NODE_H__

#include "tpp.h"
#include "kalman_trackers.h"
#include "velocity.h"
#include "path_predict.h"
#include "tpp_args_parser.h"
#include "ros_params_parser.h"
#include "ego_param.h"
#include "marker_gen.h"
#include <visualization_msgs/MarkerArray.h>

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

  bool gen_markers_ = false;
  MarkerConfig mc_;

  std_msgs::Header objs_header_;
  std_msgs::Header objs_header_prev_;

  std::vector<msgs::DetectedObject> pp_objs_;
  std::vector<std::vector<PPLongDouble> > ppss;

#if FPS_EXTRAPOLATION
  std::vector<Point32> box_centers_kalman_rel_;
  std::vector<Point32> box_centers_kalman_next_rel_;
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

  MarkerGen mg_;

  ros::Subscriber fusion_sub_;
  ros::Subscriber localization_sub_;

  double loop_begin = 0.;    // seconds
  double loop_elapsed = 0.;  // seconds

  bool is_legal_dt_ = false;

  void callback_fusion(const msgs::DetectedObjectArray::ConstPtr& input);
  void callback_localization(const msgs::LocalizationToVeh::ConstPtr& input);

  void fill_convex_hull(const msgs::BoxPoint& bPoint, msgs::ConvexPoint& cPoint, const std::string frame_id);

  void init_velocity(msgs::TrackInfo& track);

  // compute DetectedObject.relSpeed:
  // i.e., speed of relative velocity on relative coordinate projection onto object-to-ego-vehicle vector
  float compute_relative_speed_obj2ego(const Vector3_32 rel_v_rel, const Point32 obj_rel);

  float compute_radar_absolute_velocity(const float radar_speed_rel, const float box_center_x_abs,
                                        const float box_center_y_abs);

  void compute_velocity_kalman(const float ego_dx_abs, const float ego_dy_abs);

  void push_to_vector(BoxCenter a, std::vector<Point32>& b);
  void publish_tracking();

  void control_sleep(const double loop_interval);
  void publish_pp(ros::Publisher pub, std::vector<msgs::DetectedObject>& objs, const unsigned int pub_offset,
                  const float time_offset);
#if FPS_EXTRAPOLATION
  void publish_pp_extrapolation(ros::Publisher pub, std::vector<msgs::DetectedObject>& objs,
                                std::vector<Point32> box_centers_rel, std::vector<Point32> box_centers_next_rel);
#endif

  void set_ros_params();
  void subscribe_and_advertise_topics();

  void save_output_to_txt(const std::vector<msgs::DetectedObject>& objs);
};
}  // namespace tpp

#endif  // __TPP_NODE_H__