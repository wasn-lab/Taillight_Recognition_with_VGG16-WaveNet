#ifndef __TPP_H__
#define __TPP_H__

#include "tpp_base.h"
#include "../src/point32_impl.h"

namespace tpp
{
constexpr unsigned int num_forecasts_ = 20;

constexpr unsigned int num_2dbox_corners = 4;

struct BoxCenter
{
  unsigned int id;

  MyPoint32Impl pos;

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

}  // namespace tpp

#endif  // __TPP_H__
