#include "velocity.h"

namespace tpp
{
long long Velocity::get_dt()
{
  return dt_;
}

void Velocity::set_dt(const long long dt)
{
  dt_ = dt;
}

int Velocity::init_time(const double secs, const double secs_prev)
{
  time_ = secs * 1000000000;            // nanoseconds
  time_prev_ = secs_prev * 1000000000;  // nanoseconds

  dt_ = time_ - time_prev_;  // nanoseconds

  if (dt_ == 0)
  {
    dt_ = 100000000;
  }
  else if (dt_ < 0)
  {
#if DEBUG_COMPACT
    LOG_INFO << "Warning: dt = " << (dt_ / 1000000.0) << "ms ! Illegal time input !" << std::endl;

    LOG_INFO << "time t-1: " << time_prev_ << std::endl;
    LOG_INFO << "time t  : " << time_ << std::endl;
#endif

    return 1;
  }

#if DEBUG_COMPACT
  LOG_INFO << "dt = " << (dt_ / 1000000.0) << " ms" << std::endl;
#endif

  return 0;
}
}  // namespace tpp
