#include "point32_impl.h"

namespace tpp
{
void MyPoint32Impl::set_point_abs(MyPoint32 abs)
{
  has_abs_ = true;
  set_MyPoint32(abs_, abs);
}

void MyPoint32Impl::set_point_rel(MyPoint32 rel)
{
  has_rel_ = true;
  set_MyPoint32(rel_, rel);
}

void MyPoint32Impl::set_anchor_abs(PoseRPY32 anchor_abs)
{
  has_anchor_abs_ = true;
  set_PoseRPY32(anchor_abs_, anchor_abs);
}

void MyPoint32Impl::init_point_z()
{
  abs_.z = 0.f;
  rel_.z = 0.f;
}

void MyPoint32Impl::get_point_abs(MyPoint32& abs)
{
  if (!has_abs_)
  {
    std::runtime_error("No point_abs_ for return");
  }

  set_MyPoint32(abs, abs_);
}

void MyPoint32Impl::get_point_rel(MyPoint32& rel)
{
  if (!has_rel_)
  {
    std::runtime_error("No point_rel_ for return");
  }

  set_MyPoint32(rel, rel_);
}

void MyPoint32Impl::get_anchor_abs(PoseRPY32& anchor_abs)
{
  if (!has_anchor_abs_)
  {
    std::runtime_error("No anchor_abs_ for return");
  }

  set_PoseRPY32(anchor_abs, anchor_abs_);
}

void MyPoint32Impl::transform_abs2rel()
{
  if (!has_abs_ || !has_anchor_abs_)
  {
    std::runtime_error("No data for transform_point_abs2rel()");
    return;
  }

  transform_point_abs2rel(abs_.x, abs_.y, abs_.z, rel_.x, rel_.y, rel_.z,  //
                          anchor_abs_.x, anchor_abs_.y, anchor_abs_.z,     //
                          anchor_abs_.yaw);

  has_rel_ = true;
}

void MyPoint32Impl::transform_rel2abs()
{
  if (!has_rel_ || !has_anchor_abs_)
  {
    std::runtime_error("No data for transform_point_rel2abs()");
    return;
  }

  transform_point_rel2abs(rel_.x, rel_.y, rel_.z, abs_.x, abs_.y, abs_.z,  //
                          anchor_abs_.x, anchor_abs_.y, anchor_abs_.z,     //
                          anchor_abs_.yaw);

  has_abs_ = true;
}
}  // namespace tpp
