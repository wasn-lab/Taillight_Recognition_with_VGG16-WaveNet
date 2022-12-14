#ifndef __TPP_NODE_H__
#define __TPP_NODE_H__

#include "tpp.h"
#include "kalman_trackers.h"
#include "velocity.h"
#include "path_predict.h"
#include "ros_params_parser.h"
#include "ego_param.h"
#include "marker_gen.h"
#if TO_GRIDMAP
#include "points_to_costmap.h"
#endif
#include <visualization_msgs/MarkerArray.h>
#include <tf2/utils.h>
#include <geometry_msgs/PolygonStamped.h>

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
  double tf_map_orig_x_ = 0.;
  double tf_map_orig_y_ = 0.;
  double tf_map_orig_z_ = 0.;

  DISALLOW_COPY_AND_ASSIGN(TPPNode);

  int input_source_ = InputSource::LidarDet;
  int occ_source_ = OccupancySource::PlannedPathBased;

  geometry_msgs::TransformStamped tf_stamped_;

  bool save_output_txt_ = false;
  bool output_tf_map_ = false;

  bool use_tracking2d = false;

  bool gen_markers_ = false;
  MarkerConfig mc_;

  std_msgs::Header objs_header_;
  std_msgs::Header objs_header_prev_;

  std::vector<msgs::DetectedObject> pp_objs_;
  std::vector<std::vector<PPLongDouble> > ppss;

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

  nav_msgs::OccupancyGrid wayarea_;

  ros::NodeHandle nh_;
  ros::NodeHandle nh2_;

  // custom callback queue
  ros::CallbackQueue queue_;

  ros::Publisher pp_pub_;
#if HEARTBEAT == 1
  ros::Publisher pp_pub_heartbeat_;
#endif
#if TO_GRIDMAP
  ros::Publisher pp_grid_pub_;
#endif

  ros::Publisher drivable_area_pub_;

  MarkerGen mg_;

  ros::Subscriber fusion_sub_;
  void callback_fusion(const msgs::DetectedObjectArray::ConstPtr& input);

  ros::Subscriber wayarea_sub_;
  void callback_wayarea(const nav_msgs::OccupancyGrid& input);

  tf2_ros::Buffer tf_buffer_;

  ros::Subscriber ego_speed_kmph_sub_;
  void callback_ego_speed_kmph(const msgs::VehInfo::ConstPtr& input);

  ros::Subscriber lanelet2_route_sub_;
  void callback_lanelet2_route(const visualization_msgs::MarkerArray::ConstPtr& input);

  std::string frame_id_source_ = "base_link";
  std::string frame_id_target_ = "map";

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

  double ground_z_ = -3.1;

  bool drivable_area_filter_ = true;
  double expand_left_ = 2.2;
  double expand_right_ = 0.;

  std::vector<cv::Point3f> lanelet2_route_left;
  std::vector<cv::Point3f> lanelet2_route_right;

  std::vector<cv::Point2f> expanded_route_left;
  std::vector<cv::Point2f> expanded_route_right;

  geometry_msgs::Point get_transform_coordinate(geometry_msgs::Point origin_point, double yaw,
                                                geometry_msgs::Vector3 translation);
  bool check_in_polygon(cv::Point2f position, std::vector<cv::Point2f>& polygon);
  bool drivable_area_filter(const msgs::BoxPoint box_point);

  bool create_bbox_from_polygon_ = false;
  void create_bbox_from_polygon(msgs::DetectedObject& obj);

  bool create_polygon_from_bbox_ = false;
  void create_polygon_from_bbox(const msgs::BoxPoint& bPoint, msgs::ConvexPoint& cPoint, const std::string frame_id);

  void init_velocity(msgs::TrackInfo& track);

  // compute DetectedObject.speed_rel:
  // i.e., speed of relative velocity on relative coordinate projection onto object-to-ego-vehicle vector
  float compute_relative_speed_obj2ego(const Vector3_32 rel_v_rel, const MyPoint32 obj_rel);

  float compute_radar_absolute_velocity(const float radar_speed_rel, const float box_center_x_abs,
                                        const float box_center_y_abs);

  void compute_velocity_kalman();

  void push_to_vector(BoxCenter a, std::vector<MyPoint32>& b);
  void publish_tracking();

  void control_sleep(const double loop_interval);
  void publish_pp(const ros::Publisher& pub, std::vector<msgs::DetectedObject>& objs, const unsigned int pub_offset,
                  const float time_offset);
#if TO_GRIDMAP
  void publish_pp_grid(ros::Publisher pub, const std::vector<msgs::DetectedObject>& objs);
#endif

  void set_ros_params();
  void subscribe_and_advertise_topics();
  void get_current_ego_data_main();
  void get_current_ego_data(const ros::Time fusion_stamp);

  void object_yaw(msgs::DetectedObject& obj);

  // output bbox and pp points in tf_map
  void convert(msgs::PointXYZ& p, geometry_msgs::Quaternion& q);
  void convert_all_to_map_tf(std::vector<msgs::DetectedObject>& objs);
  void heading_enu(std::vector<msgs::DetectedObject>& objs);

  void save_output_to_txt(const std::vector<msgs::DetectedObject>& objs, const std::string& out_filename);
};
}  // namespace tpp

#endif  // __TPP_NODE_H__
