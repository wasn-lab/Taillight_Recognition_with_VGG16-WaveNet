#include "kalman_tracker.h"

namespace track2d
{
void KalmanTracker::predict()
{
  cv::Mat prediction = kalman_.predict();

  x_predict_ = prediction.at<float>(0);
  y_predict_ = prediction.at<float>(1);

  vx_predict_ = prediction.at<float>(2);
  vy_predict_ = prediction.at<float>(3);

  ax_predict_ = prediction.at<float>(4);
  ay_predict_ = prediction.at<float>(5);
}
}  // namespace track2d