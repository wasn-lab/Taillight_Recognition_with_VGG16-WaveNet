#ifndef OBJECTLABELUTIL_H_
#define OBJECTLABELUTIL_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "detected_object_class_id.h"
#include "image_preprocessing.h"
#include "camera_params.h"

namespace DriveNet
{
enum class net_type_id
{
  begin = 0,
  person = begin,  // 0
  bicycle,         // 1
  car,             // 2
  motorbike,       // 3
  others1,         // 4
  bus,             // 5
  others2,         // 6
  truck            // 7
};

enum class common_type_id
{
  other = sensor_msgs_itri::DetectedObjectClassId::Unknown,       // 0
  person = sensor_msgs_itri::DetectedObjectClassId::Person,       // 1
  bicycle = sensor_msgs_itri::DetectedObjectClassId::Bicycle,     // 2
  motorbike = sensor_msgs_itri::DetectedObjectClassId::Motobike,  // 3
  car = sensor_msgs_itri::DetectedObjectClassId::Car,             // 4
  bus = sensor_msgs_itri::DetectedObjectClassId::Bus,             // 5
  truck = sensor_msgs_itri::DetectedObjectClassId::Truck          // 6
};

///         front view
///          *-------*
///         /|      /|  height
///        *-|----*  |
///      --|-*----|- *
/// length |/     | /
///     ---*------*
///        \     /
///         width

struct object_box
{
  float width;
  float height;
  float length;
};

int translate_label(int label);
void transferPixelScaling(PixelPosition& positions);
void transferPixelScaling(std::vector<PixelPosition>& pixel_positions);
cv::Scalar get_label_color(int label_id);
cv::Scalar get_common_label_color(int label_id);
object_box getDefaultObjectBox(int label_id);

}  // namespace DriveNet
#endif /*OBJECTLABELUTIL_H_*/
