#include "simple_kalman.h"

namespace tpp
{
void SimpleKalman::set_transition_matrix()
{
  kalman_.transitionMatrix = (cv::Mat_<float>(num_vars_, num_vars_) << 1.f, dt_, half_dt_square_,  //
                              0.f, 1.f, dt_,                                                       //
                              0.f, 0.f, 1.f);
}

void SimpleKalman::set_process_noise_cov()
{
  kalman_.processNoiseCov = (cv::Mat_<float>(num_vars_, num_vars_) << Q1, 0.f, 0.f,  //
                             0.f, Q2, 0.f,                                           //
                             0.f, 0.f, Q3);
}

void SimpleKalman::set_dt(const double dt)
{
  dt_ = dt;
  half_dt_square_ = 0.5 * pow(dt_, 2);

  set_transition_matrix();
  set_process_noise_cov();
}

float SimpleKalman::get_prediction()
{
  return x_predict_;
}

float SimpleKalman::get_measurement()
{
  return x_measure_;
}

float SimpleKalman::get_estimate()
{
  return x_estimate_;
}

void SimpleKalman::new_tracker(const float x)
{
  kalman_.init(num_vars_, num_measures_, num_controls_, CV_32F);

  set_transition_matrix();
  set_process_noise_cov();

  cv::setIdentity(kalman_.measurementMatrix);
  cv::setIdentity(kalman_.measurementNoiseCov, cv::Scalar::all(R));
  cv::setIdentity(kalman_.errorCovPost, cv::Scalar::all(1.f));

  kalman_.statePost.at<float>(0) = x;
  kalman_.statePost.at<float>(1) = 0.f;
  kalman_.statePost.at<float>(2) = 0.f;
}

float SimpleKalman::predict()
{
  x_predict_ = kalman_.predict().at<float>(0);
  return x_predict_;
}

float SimpleKalman::update(const float x_measure)
{
  x_measure_ = x_measure;

  cv::Mat measurement = cv::Mat::zeros(num_measures_, 1, CV_32F);
  measurement.at<float>(0) = x_measure;

  kalman_.correct(measurement);

  x_estimate_ = kalman_.statePost.at<float>(0);
  return x_estimate_;
}
}  // namespace tpp