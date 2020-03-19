#ifndef __TRACK2D_H__
#define __TRACK2D_H__

#include "track2d_base.h"

namespace track2d
{
constexpr unsigned int num_2dbox_corners = 4;

struct BoxCenter
{
  unsigned int id;

  float x_rel;
  float y_rel;
  float z_rel;

  float x_length;
  float y_length;
  float z_length;

  float area;
  float volumn;
};
}  // namespace track2d

#endif  // __TRACK2D_H__
