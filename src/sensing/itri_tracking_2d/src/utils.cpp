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
}  // namespace tpp
