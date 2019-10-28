#include "ar1.h"

namespace tpp
{
long double AR1::determinant(const long double a, const long double b, const long double c, const long double d)
{
  return a * d - b * c;
}

int AR1::inverse2(const long double a, const long double b, const long double c, const long double d,  //
                  long double& A, long double& B, long double& C, long double& D)
{
  long double det = determinant(a, b, c, d);

  if (det == 0)
  {
    LOG_INFO << "Inverse matrix for ar1_params() is NOT invertable!" << std::endl;
    return 2;
  }

  A = d / det;
  B = -b / det;
  C = -c / det;
  D = a / det;

  return 0;
}

int AR1::inverse_A_square(const long double sum, const long double sum_square,  //
                          long double& A, long double& B, long double& C, long double& D)
{
  return inverse2(3, sum, sum, sum_square, A, B, C, D);
}

int AR1::compute_params(const std::vector<long double>& xs, long double& beta0, long double& beta1)
{
  // Inverse matrix
  // [A B]
  // [C D]
  long double A = 0;
  long double B = 0;
  long double C = 0;
  long double D = 0;

  long double sum0 = 0;

  for (unsigned i = 1; i < (xs.size() - 1); i++)
  {
    sum0 += xs[i];
  }
  long double sum1 = sum0 + xs[0];

  long double sum_square1 = 0;

  for (unsigned i = 0; i < (xs.size() - 1); i++)
  {
    sum_square1 += std::pow(xs[i], 2);
  }

  // compute elements A B C D of inverse matrix
  int err = inverse_A_square(sum1, sum_square1, A, B, C, D);

  if (err > 0)
  {
    return err;
  }

  long double sum2 = sum0 + xs.back();

  long double sum_square2 = 0;

  for (unsigned i = 1; i < xs.size(); i++)
  {
    sum_square2 += (xs[i - 1] * xs[i]);
  }

  beta0 = A * sum2 + B * sum_square2;
  beta1 = C * sum2 + D * sum_square2;

#if DEBUG
  std::cout << "sum1 = " << sum1 << std::endl;
  std::cout << "sum_square1 " << sum_square1 << std::endl;
  std::cout << "sum2 " << sum2 << std::endl;
  std::cout << "sum_square2 " << sum_square2 << std::endl;
#endif

  return 0;
}
}  // namespace tpp