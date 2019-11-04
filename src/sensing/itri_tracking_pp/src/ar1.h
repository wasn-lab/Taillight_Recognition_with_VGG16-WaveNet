#ifndef __AR1_H__
#define __AR1_H__

#include "tpp.h"
#include <iostream>  // std::cout

namespace tpp
{
class AR1
{
public:
  AR1()
  {
  }
  ~AR1()
  {
  }

  int compute_params(const std::vector<long double>& xs, long double& beta0, long double& beta1);

private:
  DISALLOW_COPY_AND_ASSIGN(AR1);
};
}  // namespace tpp

#endif  // __AR1_H__