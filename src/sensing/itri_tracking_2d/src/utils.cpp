#include "utils.h"

namespace tpp
{
float divide(const float dividend, const float divisor)
{
  if (divisor == 0)
  {
    throw std::runtime_error("Math error: Attempted to divide by Zero\n");
  }

  return (dividend / divisor);
}

void increase_uint(unsigned int& x)
{
  if (x == 4294967295)
  {
    x = 1;
  }
  else
  {
    x++;
  }
}

double clock_to_milliseconds(clock_t num_ticks)
{
  // units/(units/time) => time (seconds) * 1000 = milliseconds
  return (num_ticks / (double)CLOCKS_PER_SEC) * 1000.0;
}

float squared_euclidean_distance(const float x1, const float y1, const float x2, const float y2)
{
  return std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2);
}

float euclidean_distance(const float a, const float b)
{
  return std::sqrt(std::pow(a, 2) + std::pow(b, 2));
}

float euclidean_distance3(const float a, const float b, const float c)
{
  return std::sqrt(std::pow(a, 2) + std::pow(b, 2) + std::pow(c, 2));
}

// km/h to m/s
float kmph_to_mps(const float kmph)
{
  return (0.277778f * kmph);
}

// m/s to km/h
float mps_to_kmph(const float mps)
{
  return (3.6f * mps);
}

void set_PoseRPY32(PoseRPY32& out, const PoseRPY32 in)
{
  out.x = in.x;
  out.y = in.y;
  out.z = in.z;

  out.roll = in.roll;
  out.pitch = in.pitch;
  out.yaw = in.yaw;
}

void set_MyPoint32(MyPoint32& out, const MyPoint32 in)
{
  out.x = in.x;
  out.y = in.y;
  out.z = in.z;
}

void swap_MyPoint32(MyPoint32& A, MyPoint32& B)
{
  MyPoint32 tmp;

  tmp.x = A.x;
  tmp.y = A.y;
  tmp.z = A.z;

  A.x = B.x;
  A.y = B.y;
  A.z = B.z;

  B.x = tmp.x;
  B.y = tmp.y;
  B.z = tmp.z;
}

void convert_MyPoint32_to_Point(geometry_msgs::Point& out, const MyPoint32 in)
{
  out.x = (double)in.x;
  out.y = (double)in.y;
  out.z = (double)in.z;
}

MyPoint32 add_two_MyPoint32s(const MyPoint32 A, const MyPoint32 B)
{
  MyPoint32 C;
  C.x = A.x + B.x;
  C.y = A.y + B.y;
  C.z = A.z + B.z;
  return C;
}
}  // namespace tpp
