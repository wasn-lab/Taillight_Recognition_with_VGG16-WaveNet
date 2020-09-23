#ifndef __POINT32IMPL_H__
#define __POINT32IMPL_H__

#include "tpp_base.h"
#include "utils.h"

namespace tpp
{
class MyPoint32Impl
{
public:
  MyPoint32Impl()
  {
  }

  ~MyPoint32Impl()
  {
  }

  void set_point_abs(MyPoint32 abs);
  void set_point_rel(MyPoint32 rel);
  void set_anchor_abs(PoseRPY32 anchor_abs);

  void init_point_z();

  void get_point_abs(MyPoint32& abs);
  void get_point_rel(MyPoint32& rel);
  void get_anchor_abs(PoseRPY32& anchor_abs);

  void transform_abs2rel();
  void transform_rel2abs();

private:
  MyPoint32 abs_;
  MyPoint32 rel_;
  PoseRPY32 anchor_abs_;

  bool has_abs_ = false;
  bool has_rel_ = false;
  bool has_anchor_abs_ = false;
};
}  // namespace tpp

#endif  // __POINT32IMPL_H__