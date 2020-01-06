#ifndef MATHUTIL_H_
#define MATHUTIL_H_

#include <cmath>
#include <vector>

float truncateDecimalPrecision(const float num, int decimal_place);
float AbsoluteToRelativeDistance(std::vector<float> left_point, std::vector<float> right_point);

#endif /*MATHUTIL_H_*/