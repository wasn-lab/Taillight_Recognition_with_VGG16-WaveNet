#ifndef __UTILS_H__
#define __UTILS_H__

#include <ctime>
#include <cmath>
#include <msgs/PointXYZ.h>
#include "track2d_base.h"

namespace track2d
{
float divide(const float dividend, const float divisor);

template <typename T>
void assign_value_cannot_zero(T& out, const T in)
{
  T epsilon = (T)0.00001;
  out = (in == (T)0) ? epsilon : in;
}

void increase_uint(unsigned int& x);

double clock_to_milliseconds(const clock_t num_ticks);

float squared_euclidean_distance(const float x1, const float y1, const float x2, const float y2);

float euclidean_distance(const float a, const float b);

float euclidean_distance3(const float a, const float b, const float c);
}  // namespace track2d

#endif  // __UTILS_H__