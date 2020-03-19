#ifndef __VELOCITY_H__
#define __VELOCITY_H__

#include "tpp.h"
#include <cstdio>  // puts
#include <iostream>
#include <cmath>  // sin, cos
#include "utils.h"

namespace tpp
{
class Velocity
{
public:
  Velocity()
  {
  }
  ~Velocity()
  {
  }

  int init_time(const double secs, const double secs_prev);

  // getter
  long long get_dt();

  // setter
  void set_dt(const long long dt);

private:
  DISALLOW_COPY_AND_ASSIGN(Velocity);

  long long dt_ = 0;
  long long time_ = 0;
  long long time_prev_ = 0;
};
}  // namespace tpp

#endif  // __VELOCITY_H__
