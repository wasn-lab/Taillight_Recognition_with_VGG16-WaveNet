#ifndef __PATH_PREDICT_H__
#define __PATH_PREDICT_H__

#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <numeric>

#include "tpp.h"
#include "ar1.h"
#include "utils.h"

namespace tpp
{
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
                         const float ego_z_abs, const float ego_heading);

  void main(std::vector<msgs::DetectedObject>& pp_objs_, std::vector<std::vector<PPLongDouble> >& ppss,
            const bool show_pp);

private:
  DISALLOW_COPY_AND_ASSIGN(PathPredict);

  bool show_pp_ = 0;

  static constexpr std::size_t max_order_ = 1;
  const std::size_t num_pp_input_min_ = 6;
  const std::size_t num_pp_input_max_ = 20;

  float ego_x_abs_ = 0.f;
  float ego_y_abs_ = 0.f;
  float ego_z_abs_ = 0.f;
  float ego_heading_ = 0.f;

  std::size_t num_pp_input_in_use_ = 0;

  void create_pp_input(const Point32 point, std::vector<long double>& data_x, std::vector<long double>& data_y);

  void create_pp_input_main(const msgs::TrackInfo& track, std::vector<long double>& data_x,
                            std::vector<long double>& data_y);

  void resolve_repeating_number(long double& x1, long double& x2, long double& x3, long double& x4);

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
};
}  // namespace tpp

#endif  // __PATH_PREDICT_H__
