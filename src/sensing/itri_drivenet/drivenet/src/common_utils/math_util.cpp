#include "drivenet/math_util.h"

void rounding(float &num, int index)
{
  int multiplier = 0;
  multiplier = pow(10, index);
  num = (int)(num * multiplier + 0.5) / (multiplier * 1.0);
}