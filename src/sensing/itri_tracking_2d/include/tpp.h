#ifndef __TPP_H__
#define __TPP_H__

#include "tpp_base.h"

namespace tpp
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
}  // namespace tpp

#endif  // __TPP_H__
