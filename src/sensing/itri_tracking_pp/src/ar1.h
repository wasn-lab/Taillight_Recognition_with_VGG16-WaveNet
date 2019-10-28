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

  long double determinant(const long double a, const long double b, const long double c, const long double d);

  int inverse2(const long double a, const long double b, const long double c, const long double d,  //
               long double& A, long double& B, long double& C, long double& D);

  int inverse_A_square(const long double sum, const long double sum_square,  //
                       long double& A, long double& B, long double& C, long double& D);
};
}  // namespace tpp

#endif  // __AR1_H__