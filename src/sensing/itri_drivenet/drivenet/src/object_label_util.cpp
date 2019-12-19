#include "drivenet/object_label_util.h"

namespace DriveNet
{
const std::vector<cv::Scalar> g_label_colors = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                                    cv::Scalar(125, 125, 125)};

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
cv::Scalar get_labelColor(int label_id)
{
  cv::Scalar class_color;
  if (label_id == static_cast<int>(DriveNet::net_type_id::person))
    class_color = g_label_colors[0];
  else if (label_id == static_cast<int>(DriveNet::net_type_id::bicycle) ||
           label_id == static_cast<int>(DriveNet::net_type_id::motorbike))
    class_color = g_label_colors[1];
  else if (label_id == static_cast<int>(DriveNet::net_type_id::car) ||
           label_id == static_cast<int>(DriveNet::net_type_id::bus) ||
           label_id == static_cast<int>(DriveNet::net_type_id::truck))
    class_color = g_label_colors[2];
  else
    class_color = g_label_colors[3];
  return class_color;
}

cv::Scalar get_commonLabelColor(int label_id)
{
  cv::Scalar class_color;
  if (label_id == static_cast<int>(DriveNet::common_type_id::person))
    class_color = g_label_colors[0];
  else if (label_id == static_cast<int>(DriveNet::common_type_id::bicycle) ||
           label_id == static_cast<int>(DriveNet::common_type_id::motorbike))
    class_color = g_label_colors[1];
  else if (label_id == static_cast<int>(DriveNet::common_type_id::car) ||
           label_id == static_cast<int>(DriveNet::common_type_id::bus) ||
           label_id == static_cast<int>(DriveNet::common_type_id::truck))
    class_color = g_label_colors[2];
  else
    class_color = g_label_colors[3];
  return class_color;
}
};