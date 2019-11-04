#include "ar1.h"

namespace tpp
{
int AR1::compute_params(const std::vector<long double>& xs, long double& beta0, long double& beta1)
{
  int err = 0;

  long double sum0 = 0.;  // xs[1]...xs[-2]
  for (unsigned i = 1; i < (xs.size() - 1); i++)
  {
    sum0 += xs[i];
  }

  long double sum1 = sum0 + xs[0];  // xs[0]...xs[-2]

  long double sum_square1 = 0.;  // xs[0]^2...xs[-2]^2

  for (unsigned i = 0; i < (xs.size() - 1); i++)
  {
    sum_square1 += std::pow(xs[i], 2);
  }

  Eigen::Matrix2d m;
  m << 3., sum1, sum1, sum_square1;
  Eigen::Matrix3d inverse;

  bool is_invertible = false;
  double determinant = 0.;

  m.computeInverseAndDetWithCheck(inverse, determinant, is_invertible);

  beta0 = 0.;
  beta1 = 0.;

  if (is_invertible)
  {
    long double sum2 = sum0 + xs.back();  // xs[1]...xs[-1]

    long double sum_square2 = 0.;  // xs[0]xs[1]...xs[-2]xs[-1]

    for (unsigned i = 1; i < xs.size(); i++)
    {
      sum_square2 += (xs[i - 1] * xs[i]);
    }

    beta0 = (long double)(inverse(0, 0) * sum2 + inverse(0, 1) * sum_square2);
    beta1 = (long double)(inverse(1, 0) * sum2 + inverse(1, 1) * sum_square2);

#if DEBUG
    std::cout << "determinant = " << determinant << std::endl;
    std::cout << "m is invertible. inverse = " << std::endl << inverse << std::endl;

    std::cout << "sum1 = " << sum1 << std::endl;
    std::cout << "sum_square1 = " << sum_square1 << std::endl;

    std::cout << "sum2 = " << sum2 << std::endl;
    std::cout << "sum_square2 = " << sum_square2 << std::endl;
#endif
  }
  else
  {
    std::cout << "m is not invertible." << std::endl;
    err = 2;
  }

#if DEBUG
  std::cout << "beta0 = " << beta0 << std::endl;
  std::cout << "beta1 = " << beta1 << std::endl;
#endif

  return err;
}
}  // namespace tpp