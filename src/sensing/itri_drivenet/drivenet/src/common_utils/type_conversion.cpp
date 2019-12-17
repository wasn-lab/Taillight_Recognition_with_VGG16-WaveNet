#include "drivenet/type_conversion.h"

std::string floatToString(float value)
{
  std::ostringstream ostream;
  ostream << value;
  std::string value_str_(ostream.str());

  return value_str_;
}