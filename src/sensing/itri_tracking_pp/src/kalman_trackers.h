#ifndef __KALMAN_TRACKERS_H__
#define __KALMAN_TRACKERS_H__

#include "tpp.h"
#include "kalman_tracker.h"
#include "hungarian.h"
#include <vector>
#include <msgs/DetectedObject.h>
#include "utils.h"

namespace tpp
{
class KalmanTrackers
{
public:
  std_msgs::Header header_;
  std::vector<msgs::DetectedObject> objs_;
  std::vector<KalmanTracker> tracks_;

  KalmanTrackers()
  {
  }

  ~KalmanTrackers()
  {
  }

  float get_dt()
  {
    return dt_;
  }

  void kalman_tracker_main(const long long dt, const float ego_x_abs, const float ego_y_abs, const float ego_z_abs,
                           const float ego_heading);

  float get_Q1()
  {
    return Q1;
  }

  float get_Q2()
  {
    return Q2;
  }

  float get_Q3()
  {
    return Q3;
  }

  float get_R()
  {
    return R;
  }

  float get_P0()
  {
    return P0;
  }

private:
  DISALLOW_COPY_AND_ASSIGN(KalmanTrackers);

  float dt_ = 1.f;
  float half_dt_square_ = 0.5f;

  // kalman: box center position and velocity
  static constexpr int num_vars_ = 6;
  static constexpr int num_measures_ = 2;
  static constexpr int num_controls_ = 0;

  static constexpr float Q1 = 0.25f;  // 0.5^2
  static constexpr float Q2 = 0.25f;  // 0.5^2
  static constexpr float Q3 = 0.09f;  // 0.3^2
  static constexpr float R = 0.04f;   // 0.2^2: +/-20cm
  static constexpr float P0 = 0.25f;  // 0.5^2: +/-50cm

  static constexpr unsigned int TRACK_ID_MIN = 1;
  static constexpr unsigned int TRACK_ID_MAX = 4294967295;
  const float TRACK_INVALID = 100.f;  // distance 100m

  unsigned int trackid_new_ = TRACK_ID_MIN;

  static constexpr float TRACK_RANGE_SED = 16.f;          // 4^2
  static constexpr float TRACK_RANGE_SED_WARMUP = 16.f;  // 5^2

  float ego_x_abs_ = 0.f;
  float ego_y_abs_ = 0.f;
  float ego_z_abs_ = 0.f;
  float ego_heading_ = 0.f;

  static constexpr float BOX_SIZE_TH = 0.3f;

  const float BOX_VOL_MIN_FOR_RATIO = 1.f;
  static constexpr float BOX_VOL_RATIO_MAX = 1.7f;
  static constexpr float COST_BOX_DIST_W = 0.5f;
  static constexpr float COST_BOX_VOL_RATIO_W = 1.f - COST_BOX_DIST_W;
  static constexpr float PUNISH_RATIO = 0.5f;

  std::vector<BoxCenter> box_centers_;
  std::vector<std::vector<BoxCorner> > box_corners_of_boxes_;

  std::vector<std::vector<float> > distance_table_;

  void set_time_displacement(const long long dt);

  void set_ego_data(const float ego_x_abs, const float ego_y_abs, const float ego_z_abs, const float ego_heading);

  void extract_box_center(BoxCenter& box_center, const msgs::BoxPoint& box);
  void extract_box_centers();

  void extract_box_corner(BoxCorner& box_corner, const MyPoint32& corner, const signed char order);
  void extract_box_corners_of_boxes();

  void extract_box_two_axes_of_boxes();

  void init_objs();

  void init_distance_table();
  void compute_distance_table();

  void associate_data();

  void give_ids_to_unassociated_objs();
  void correct_duplicate_track_ids();

  void new_tracker(const msgs::DetectedObject& box, BoxCenter& box_center, const std::vector<BoxCorner>& box_corners);
  void correct_tracker(KalmanTracker& track, const float x_measure, const float y_measure);

  void update_associated_trackers();
  void mark_lost_trackers();
  void delete_lost_trackers();
  void add_new_trackers();

  void increase_track_id();
  void increase_tracktime();

  void set_new_box_corners_absolute(const unsigned int i, const BoxCenter& box_center);
  void set_new_box_corners_of_boxes_absolute();
  void set_new_box_corners_of_boxes_relative();

  void update_boxes();
};
}  // namespace tpp

#endif  // __KALMAN_TRACKERS_H__