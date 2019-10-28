#ifndef __POINT32IMPL_H__
#define __POINT32IMPL_H__

#include "tpp_base.h"
#include "utils.h"

namespace tpp
{
class Point32Impl
{
public:
  Point32Impl()
  {
  }

  ~Point32Impl()
  {
  }

  void set_point_abs(Point32 abs);
  void set_point_rel(Point32 rel);
  void set_anchor_abs(PoseRPY32 anchor_abs);

  void init_point_z();

  void get_point_abs(Point32& abs);
  void get_point_rel(Point32& rel);
  void get_anchor_abs(PoseRPY32& anchor_abs);

  void transform_abs2rel();
  void transform_rel2abs();

private:
  Point32 abs_;
  Point32 rel_;
  PoseRPY32 anchor_abs_;

  bool has_abs_ = false;
  bool has_rel_ = false;
  bool has_anchor_abs_ = false;
};
}  // namespace tpp

#endif  // __POINT32IMPL_H__