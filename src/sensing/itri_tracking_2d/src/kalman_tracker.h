#ifndef __KALMAN_TRACKER_H__
#define __KALMAN_TRACKER_H__

#include "tpp.h"
#include "track_hist.h"
#include <msgs/DetectedObject.h>
#include <opencv2/video/tracking.hpp>

namespace tpp
{
class KalmanTracker
{
public:
  unsigned int id_ = 0;  // indicate the tracking obj it belongs

  unsigned int tracktime_ = 0;
  static constexpr unsigned int warmup_time_ = 3;
  bool tracked_ = false;

  unsigned int lost_time_ = 0;
  static constexpr unsigned int lost_time_max_ = 1;
  bool lost_ = false;

  cv::KalmanFilter kalman_;

  msgs::DetectedObject box_;
  TrackHist hist_;

  BoxCenter box_center_;
  BoxCenter box_center_prev_;

  std::vector<BoxCorner> box_corners_;

  float x_predict_ = 0.f;
  float y_predict_ = 0.f;

  float vx_predict_ = 0.f;
  float vy_predict_ = 0.f;

  float ax_predict_ = 0.f;
  float ay_predict_ = 0.f;

  KalmanTracker()
  {
  }

  ~KalmanTracker()
  {
  }

  void predict();

private:
  // DISALLOW_COPY_AND_ASSIGN(KalmanTracker);  // cause build error, root cause unknown
};
}  // namespace tpp

#endif  // __KALMAN_TRACKER_H__