#ifndef __EGO_PARAM_H__
#define __EGO_PARAM_H__

#include "tpp.h"
#include "simple_kalman.h"
#include "utils.h"

namespace tpp
{
class EgoParam
{
public:
  EgoParam()
  {
  }

  ~EgoParam()
  {
  }

  SimpleKalman kf_;
  void update(const float ego_param_input);

private:
  bool is_kf_ = false;
  double t_ = 0.;       // seconds
  double t_prev_ = 0.;  // seconds
  double dt_ = 0.;      // seconds
};
}  // namespace tpp

#endif  // __EGO_PARAM_H__