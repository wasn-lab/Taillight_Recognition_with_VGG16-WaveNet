#include "drivenet/math_util.h"
#include <iostream>

float truncateDecimalPrecision(const float num, int decimal_place)
{
  int multiplier = 0;
  float num_rounding = num;
  multiplier = pow(10, decimal_place);
  num_rounding = (int)(num_rounding * multiplier + 0.5) / (multiplier * 1.0);
  return num_rounding;
}

float AbsoluteToRelativeDistance(std::vector<float> left_point, std::vector<float> right_point)
{
  float distance = 0;
  std::vector<float> centerPoint(2);
  centerPoint[0] = (left_point[0] + right_point[0]) / 2;
  centerPoint[1] = (left_point[1] + right_point[1]) / 2;
  distance = sqrt(pow(centerPoint[0], 2) + pow(centerPoint[1], 2));
  return distance;
}