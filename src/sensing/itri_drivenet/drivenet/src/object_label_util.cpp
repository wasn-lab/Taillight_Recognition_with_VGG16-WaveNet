#include "drivenet/object_label_util.h"

namespace DriveNet
{
const int g_image_w = camera::image_width;
const int g_image_h = camera::image_height;
const int g_raw_image_w = camera::raw_image_width;
const int g_raw_image_h = camera::raw_image_height;
const float g_scaling_ratio_w = (float)g_image_w / (float)g_raw_image_w;
const float g_scaling_ratio_h = (float)g_image_h / (float)g_raw_image_h;

int translate_label(int label)
{
  if (label == static_cast<int>(DriveNet::yolo_class_id::person))
  {
    return static_cast<int>(DriveNet::common_type_id::person);
  }
  else if (label == static_cast<int>(DriveNet::yolo_class_id::bicycle))
  {
    return static_cast<int>(DriveNet::common_type_id::bicycle);
  }
  else if (label == static_cast<int>(DriveNet::yolo_class_id::car))
  {
    return static_cast<int>(DriveNet::common_type_id::car);
  }
  else if (label == static_cast<int>(DriveNet::yolo_class_id::motorbike))
  {
    return static_cast<int>(DriveNet::common_type_id::motorbike);
  }
  else if (label == static_cast<int>(DriveNet::yolo_class_id::bus))
  {
    return static_cast<int>(DriveNet::common_type_id::bus);
  }
  else if (label == static_cast<int>(DriveNet::yolo_class_id::truck))
  {
    return static_cast<int>(DriveNet::common_type_id::truck);
  }
  else
  {
    return static_cast<int>(DriveNet::common_type_id::other);
  }
}
cv::Scalar get_label_color(int label_id)
{
  cv::Scalar class_color;
  if (label_id == static_cast<int>(DriveNet::yolo_class_id::person))
  {
    class_color = CvColor::red_;
  }
  else if (label_id == static_cast<int>(DriveNet::yolo_class_id::bicycle) ||
           label_id == static_cast<int>(DriveNet::yolo_class_id::motorbike))
  {
    class_color = CvColor::green_;
  }
  else if (label_id == static_cast<int>(DriveNet::yolo_class_id::car) ||
           label_id == static_cast<int>(DriveNet::yolo_class_id::bus) ||
           label_id == static_cast<int>(DriveNet::yolo_class_id::truck))
  {
    class_color = CvColor::blue_;
  }
  else
  {
    class_color = CvColor::gray_;
  }
  return class_color;
}

cv::Scalar get_common_label_color(int label_id)
{
  cv::Scalar class_color;
  if (label_id == static_cast<int>(DriveNet::common_type_id::person))
  {
    class_color = CvColor::red_;
  }
  else if (label_id == static_cast<int>(DriveNet::common_type_id::bicycle) ||
           label_id == static_cast<int>(DriveNet::common_type_id::motorbike))
  {
    class_color = CvColor::green_;
  }
  else if (label_id == static_cast<int>(DriveNet::common_type_id::car) ||
           label_id == static_cast<int>(DriveNet::common_type_id::bus) ||
           label_id == static_cast<int>(DriveNet::common_type_id::truck))
  {
    class_color = CvColor::blue_;
  }
  else
  {
    class_color = CvColor::gray_;
  }
  return class_color;
}
std::string get_common_label_string(int label_id)
{
  std::string class_name = "";
  if (label_id == static_cast<int>(DriveNet::common_type_id::person))
  {
    class_name = "P";
  }
  else if (label_id == static_cast<int>(DriveNet::common_type_id::bicycle))
  {
    class_name = "B";
  }
  else if (label_id == static_cast<int>(DriveNet::common_type_id::motorbike))
  {
    class_name = "M";
  }
  else if (label_id == static_cast<int>(DriveNet::common_type_id::car))
  {
    class_name = "C";
  }
  else if (label_id == static_cast<int>(DriveNet::common_type_id::bus))
  {
    class_name = "B";
  }
  else if (label_id == static_cast<int>(DriveNet::common_type_id::truck))
  {
    class_name = "T";
  }
  else
  {
    class_name = "";
  }
  return class_name;
}

void transferPixelScaling(PixelPosition& positions)
{
  positions.u = int(positions.u * g_scaling_ratio_w);
  positions.v = int(positions.v * g_scaling_ratio_h);
}

void transferPixelScaling(std::vector<PixelPosition>& pixel_positions)
{
  for (auto& positions : pixel_positions)
  {
    positions.u = int(positions.u * g_scaling_ratio_w);
    positions.v = int(positions.v * g_scaling_ratio_h);
  }
}

object_box getDefaultObjectBox(int label_id)
{
  object_box bbox{};
  if (label_id == static_cast<int>(DriveNet::common_type_id::person))
  {
    bbox.width = 0.6;
    bbox.height = 1.8;
    bbox.length = 0.33;
  }
  else if (label_id == static_cast<int>(DriveNet::common_type_id::bicycle) ||
           label_id == static_cast<int>(DriveNet::common_type_id::motorbike))
  {
    bbox.width = 0.6;
    bbox.height = 1.8;
    bbox.length = 2.5;
  }
  else if (label_id == static_cast<int>(DriveNet::common_type_id::car) ||
           label_id == static_cast<int>(DriveNet::common_type_id::truck))
  {
    bbox.width = 2;
    bbox.height = 1.5;
    bbox.length = 5;
  }
  else if (label_id == static_cast<int>(DriveNet::common_type_id::bus))
  {
    bbox.width = 2.5;
    bbox.height = 2;
    bbox.length = 7;
  }
  else
  {
    bbox.width = 0.6;
    bbox.height = 1.8;
    bbox.length = 0.33;
  }
  return bbox;
}
}  // namespace DriveNet
