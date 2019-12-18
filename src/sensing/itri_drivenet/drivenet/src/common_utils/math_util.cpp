#include "drivenet/math_util.h"

float truncateDecimalPrecision(const float num, int decimal_place)
{
  int multiplier = 0; 
  float num_rounding = num;
  multiplier = pow(10, decimal_place);
  num_rounding = (int)(num_rounding * multiplier + 0.5) / (multiplier * 1.0);
  return num_rounding;
}