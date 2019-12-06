#include "drivenet/object_label_util.h"

namespace DriveNet
{
int translate_label(int label)
{
  if (label == static_cast<int>(DriveNet::net_type_id::person))
  {
    return static_cast<int>(DriveNet::common_type_id::person);
  }
  else if (label == static_cast<int>(DriveNet::net_type_id::bicycle))
  {
    return static_cast<int>(DriveNet::common_type_id::bicycle);
  }
  else if (label == static_cast<int>(DriveNet::net_type_id::car))
  {
    return static_cast<int>(DriveNet::common_type_id::car);
  }
  else if (label == static_cast<int>(DriveNet::net_type_id::motorbike))
  {
    return static_cast<int>(DriveNet::common_type_id::motorbike);
  }
  else if (label == static_cast<int>(DriveNet::net_type_id::bus))
  {
    return static_cast<int>(DriveNet::common_type_id::bus);
  }
  else if (label == static_cast<int>(DriveNet::net_type_id::truck))
  {
    return static_cast<int>(DriveNet::common_type_id::truck);
  }
  else
  {
    return static_cast<int>(DriveNet::common_type_id::other);
  }
}
cv::Scalar get_labelColor(std::vector<cv::Scalar> colors, int label_id)
{
  cv::Scalar class_color;
  if (label_id == static_cast<int>(DriveNet::net_type_id::person))
    class_color = colors[0];
  else if (label_id == static_cast<int>(DriveNet::net_type_id::bicycle) ||
           label_id == static_cast<int>(DriveNet::net_type_id::motorbike))
    class_color = colors[1];
  else if (label_id == static_cast<int>(DriveNet::net_type_id::car) ||
           label_id == static_cast<int>(DriveNet::net_type_id::bus) ||
           label_id == static_cast<int>(DriveNet::net_type_id::truck))
    class_color = colors[2];
  else
    class_color = colors[3];
  return class_color;
}
};