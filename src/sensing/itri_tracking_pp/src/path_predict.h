#ifndef __PATH_PREDICT_H__
#define __PATH_PREDICT_H__

#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <numeric>
#include <nav_msgs/OccupancyGrid.h>

#include "tpp.h"
#include "ar1.h"
#include "utils.h"

namespace tpp
{
struct PointLD
{
  long double x;
  long double y;
  long double z;
};

class PathPredict
{
public:
  PathPredict()
  {
  }
  ~PathPredict()
  {
  }

  void callback_tracking(std::vector<msgs::DetectedObject>& pp_objs_, const float ego_x_abs, const float ego_y_abs,
                         const float ego_z_abs, const float ego_heading, const int input_source);

  void main(std::vector<msgs::DetectedObject>& pp_objs_, std::vector<std::vector<PPLongDouble> >& ppss,
            const unsigned int show_pp, const nav_msgs::OccupancyGrid& wayarea);

  void set_input_shift_m(const long double shift_m)
  {
    input_shift_m_ = shift_m;
  }

  void set_pp_obj_min_kmph(const double pp_obj_min_kmph)
  {
    pp_obj_min_kmph_ = pp_obj_min_kmph;
  }

  void set_pp_obj_max_kmph(const double pp_obj_max_kmph)
  {
    pp_obj_max_kmph_ = pp_obj_max_kmph;
  }

  void set_num_pp_input_min(const std::size_t num_pp_input_min)
  {
    num_pp_input_min_ = std::min(std::max(num_pp_input_min, (std::size_t)3), num_pp_input_max_ - (std::size_t)2);
  }

private:
  DISALLOW_COPY_AND_ASSIGN(PathPredict);

  int input_source_ = InputSource::CameraDetV2;
  unsigned int show_pp_ = 0;

  static constexpr std::size_t max_order_ = 1;
  std::size_t num_pp_input_min_ = 6;
  const std::size_t num_pp_input_max_ = 20;

  static constexpr float pp_allow_x_min_m = -10.f;
  static constexpr float pp_allow_x_max_m = 100.f;
  static constexpr float pp_allow_y_min_m = -30.f;
  static constexpr float pp_allow_y_max_m = 30.f;

  static constexpr float box_length_thr_xy = 0.7f;
  static constexpr float box_length_thr_xy_thin = 0.4f;
  static constexpr float box_length_thr_z = 0.5f;

  float ego_x_abs_ = 0.f;
  float ego_y_abs_ = 0.f;
  float ego_z_abs_ = 0.f;
  float ego_heading_ = 0.f;

  std::size_t num_pp_input_in_use_ = 0;

  std::vector<PointLD> offsets_;
  // set input_shift_m_ large enough to ensure all input data and pp points far from (0, 0)
  // warning: near 0 would distord pp results
  long double input_shift_m_ = 0.;

  double pp_obj_min_kmph_ = 0.;
  double pp_obj_max_kmph_ = 1000.;

  float confidence_thr_ = 0.f;

  void compute_pos_offset(const std::vector<long double>& data_x, const std::vector<long double>& data_y);
  void normalize_pos(std::vector<long double>& data_x, std::vector<long double>& data_y);

  void create_pp_input(const MyPoint32 point, std::vector<long double>& data_x, std::vector<long double>& data_y);

  void create_pp_input_main(const msgs::TrackInfo& track, std::vector<long double>& data_x,
                            std::vector<long double>& data_y);

  long double variance(const std::vector<long double>& samples, const long double sum_samples);

  long double standard_deviation(const long double covariance);

  void ar1_pp(const long double beta0, const long double beta1, long double& pos, long double& observation,
              long double& sum_samples, long double& mean, long double& cov, long double& stdev,
              std::vector<long double>& data);

  long double covariance(const std::vector<long double>& samples_x, const std::vector<long double>& samples_y,
                         const long double mean_x, const long double mean_y);

  long double correlation(const long double cov_xy, const long double stdev_x, const long double stdev_y);

  void confidence_ellipse(PPLongDouble& pp, const float alpha);

  int ar1_params_main(PPLongDouble& pp, std::vector<long double>& data_x, std::vector<long double>& data_y);

  void covariance_matrix(PPLongDouble& pp, std::vector<long double>& data_x, std::vector<long double>& data_y);

  int predict(std::size_t max_order_, const std::size_t num_forecasts_, std::vector<long double>& data_x,
              std::vector<long double>& data_y, std::vector<PPLongDouble>& pps);

  void confidence_threshold(const unsigned int confidence_lv);

  void confidence_ellipse_main(const std::size_t num_forecasts_, std::vector<long double>& data_x,
                               std::vector<long double>& data_y, std::vector<PPLongDouble>& pps);

  void pp_vertices(PPLongDouble& pps, const msgs::PathPrediction forecast, const int pp_idx, const float abs_speed);
};
}  // namespace tpp

#endif  // __PATH_PREDICT_H__
