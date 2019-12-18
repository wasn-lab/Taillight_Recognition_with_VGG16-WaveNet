#include "drivenet/type_conversion.h"

std::string floatToString_with_RealPrecision_with_RealPrecision(float value)
{
  std::ostringstream ostream;
  ostream << value;
  std::string value_str_(ostream.str());

  return value_str_;
}