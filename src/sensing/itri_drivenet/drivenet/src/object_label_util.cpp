#include "drivenet/object_label_util.h"

namespace DriveNet
{
int image_w_ = camera::image_width;
int image_h_ = camera::image_height;
int raw_image_w_ = camera::raw_image_width;
int raw_image_h_ = camera::raw_image_height;
float scaling_ratio_w_ = (float)image_w_ / (float)raw_image_w_;
float scaling_ratio_h_ = (float)image_h_ / (float)raw_image_h_;

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
cv::Scalar get_label_color(int label_id)
{
  cv::Scalar class_color;
  if (label_id == static_cast<int>(DriveNet::net_type_id::person))
  {
    class_color = Color::red_;
  }
  else if (label_id == static_cast<int>(DriveNet::net_type_id::bicycle) ||
           label_id == static_cast<int>(DriveNet::net_type_id::motorbike))
  {
    class_color = Color::green_;
  }
  else if (label_id == static_cast<int>(DriveNet::net_type_id::car) ||
           label_id == static_cast<int>(DriveNet::net_type_id::bus) ||
           label_id == static_cast<int>(DriveNet::net_type_id::truck))
  {
    class_color = Color::blue_;
  }
  else
  {
    class_color = Color::gray_;
  }
  return class_color;
}

cv::Scalar get_common_label_color(int label_id)
{
  cv::Scalar class_color;
  if (label_id == static_cast<int>(DriveNet::common_type_id::person))
  {
    class_color = Color::red_;
  }
  else if (label_id == static_cast<int>(DriveNet::common_type_id::bicycle) ||
           label_id == static_cast<int>(DriveNet::common_type_id::motorbike))
  {
    class_color = Color::green_;
  }
  else if (label_id == static_cast<int>(DriveNet::common_type_id::car) ||
           label_id == static_cast<int>(DriveNet::common_type_id::bus) ||
           label_id == static_cast<int>(DriveNet::common_type_id::truck))
  {
    class_color = Color::blue_;
  }
  else
  {
    class_color = Color::gray_;
  }
  return class_color;
}

void transferPixelScaling(std::vector<PixelPosition>& pixel_positions)
{
  for(auto& positions: pixel_positions)
  {
    positions.u = int(positions.u * scaling_ratio_w_);
    positions.v = int(positions.v * scaling_ratio_h_);
  }
}

} // namespace DriveNet
