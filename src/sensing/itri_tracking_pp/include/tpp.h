#ifndef __TPP_H__
#define __TPP_H__

#include "tpp_base.h"
#include "../src/point32_impl.h"

namespace tpp
{
static double input_fps = 5;    // known callback rate
static double output_fps = 10;  // expected publish rate

static unsigned int num_publishs_per_loop =
    std::max((unsigned int)1, (unsigned int)std::floor(std::floor(output_fps / input_fps)));

constexpr unsigned int num_forecasts_ = 20;

constexpr unsigned int num_2dbox_corners = 4;

struct BoxCenter
{
  unsigned int id;

  Point32Impl pos;

  float x_length;
  float y_length;
  float z_length;

  float area;
  float volumn;

  float dist_to_ego;

  float vec1_x_abs;
  float vec1_y_abs;
  float vec1_z_abs;

  float vec2_x_abs;
  float vec2_y_abs;
  float vec2_z_abs;
};

struct BoxCorner
{
  unsigned int id;
  signed char order;  // order of the four corners of a bbox

  float x_rel;
  float y_rel;
  float z_rel;

  float x_abs;
  float y_abs;
  float z_abs;

  float new_x_rel;
  float new_y_rel;
  float new_z_rel;

  float new_x_abs;
  float new_y_abs;
  float new_z_abs;
};

struct PPLongDouble
{
  unsigned int id;

  long double beta0_x;
  long double beta1_x;

  long double beta0_y;
  long double beta1_y;

  long double sum_samples_x;
  long double sum_samples_y;

  long double observation_x;
  long double observation_y;

  long double pos_x;
  long double pos_y;

  long double mean_x;
  long double mean_y;

  long double stdev_x;
  long double stdev_y;

  long double cov_xx;
  long double cov_yy;
  long double cov_xy;
  long double corr_xy;

  float a1;  // length of vector1 of confidence ellipse
  float a2;  // length of vector2 of confidence ellipse

  tf2::Quaternion q1;  // direction of vector1 of confidence ellipse
  tf2::Quaternion q2;  // direction of vector2 of confidence ellipse
};
}  // namespace tpp

#endif  // __TPP_H__
