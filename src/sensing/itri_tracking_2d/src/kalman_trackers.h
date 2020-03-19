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

  void kalman_tracker_main(const long long dt);

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

  static constexpr float Q1 = 400.f;  // 20^2
  static constexpr float Q2 = 100.f;  // 10^2
  static constexpr float Q3 = 25.f;   // 5^2
  static constexpr float R = 25.f;    // 5^2: +/-20cm
  static constexpr float P0 = 100.f;  // 5^2: +/-50cm

  static constexpr unsigned int TRACK_ID_MIN = 1;
  static constexpr unsigned int TRACK_ID_MAX = 4294967295;
  const float TRACK_INVALID = 10000.f;  // distance 10000px

  unsigned int trackid_new_ = TRACK_ID_MIN;

  static constexpr float TRACK_RANGE_SED = 900.f;          // 30^2
  static constexpr float TRACK_RANGE_SED_WARMUP = 1600.f;  // 40^2

  static constexpr float BOX_SIZE_TH = 0.3f;

  const float BOX_VOL_MIN_FOR_RATIO = 1.f;
  static constexpr float BOX_VOL_RATIO_MAX = 1.5f;
  static constexpr float COST_BOX_DIST_W = 0.5f;
  static constexpr float COST_BOX_VOL_RATIO_W = 1.f - COST_BOX_DIST_W;

  std::vector<BoxCenter> box_centers_;

  std::vector<std::vector<float> > distance_table_;

  void set_time_displacement(const long long dt);

  void extract_2dbox_center(BoxCenter& box_center, const msgs::CamInfo& box);
  void extract_box_centers();

  void init_objs();

  void init_distance_table();
  void compute_distance_table();

  void associate_data();

  void give_ids_to_unassociated_objs();
  void correct_duplicate_track_ids();

  void new_tracker(const msgs::DetectedObject& box, BoxCenter& box_center);
  void correct_tracker(KalmanTracker& track, const float x_measure, const float y_measure);

  void update_associated_trackers();
  void mark_lost_trackers();
  void delete_lost_trackers();
  void add_new_trackers();

  void increase_track_id();
  void increase_tracktime();
};
}  // namespace tpp

#endif  // __KALMAN_TRACKERS_H__