#ifndef __SIMPLE_KALMAN_H__
#define __SIMPLE_KALMAN_H__

#include "tpp.h"
#include <opencv2/video/tracking.hpp>

namespace tpp
{
class SimpleKalman
{
public:
  SimpleKalman()
  {
  }

  ~SimpleKalman()
  {
  }

  void set_dt(const double dt);

  float get_prediction();
  float get_measurement();
  float get_estimate();

  void new_tracker(const float x);
  float predict();
  float update(const float x_measure);

private:
  DISALLOW_COPY_AND_ASSIGN(SimpleKalman);

  cv::KalmanFilter kalman_;

  // kalman: box center position and velocity
  static constexpr int num_vars_ = 3;
  static constexpr int num_measures_ = 1;
  static constexpr int num_controls_ = 0;

  static constexpr float Q1 = 0.25f;  // 0.5^2
  static constexpr float Q2 = 0.25f;  // 0.5^2
  static constexpr float Q3 = 0.09f;  // 0.3^2
  static constexpr float R = 0.04f;   // 0.2^2: +/-20cm
  static constexpr float P0 = 0.25f;  // 0.5^2: +/-50cm

  float dt_ = 1.f;
  float half_dt_square_ = 0.5f;
  float x_predict_ = 0.f;
  float x_measure_ = 0.f;
  float x_estimate_ = 0.f;

  void set_transition_matrix();
  void set_process_noise_cov();
};
}  // namespace tpp

#endif  // __SIMPLE_KALMAN_H__