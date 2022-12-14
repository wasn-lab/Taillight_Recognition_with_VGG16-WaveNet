#ifndef __TPP_H__
#define __TPP_H__

#include "tpp_base.h"
#include "../src/point32_impl.h"

namespace tpp
{
constexpr unsigned int NUM_2DBOX_CORNERS = 4;

enum InputSource
{
  Fusion = 0,
  LidarDet,
  LidarDet_PointPillars_Car,
  LidarDet_PointPillars_Ped_Cyc,
  VirtualBBoxAbs,
  VirtualBBoxRel,
  CameraDetV2,
  Tracking2D,
  NumInputSources
};

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
