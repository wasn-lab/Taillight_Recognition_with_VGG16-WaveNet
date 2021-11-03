#include "ego_param.h"

namespace tpp
{
void EgoParam::update(const float ego_param_input)
{
  if (!is_kf_)
  {
    t_ = ros::Time::now().toSec();
    kf_.new_tracker(ego_param_input);
    is_kf_ = true;
  }
  else
  {
    t_prev_ = t_;
    t_ = ros::Time::now().toSec();
    dt_ = t_ - t_prev_;

    kf_.set_dt(dt_);
    kf_.predict();
    kf_.update(ego_param_input);
  }
}
}  // namespace tpp
