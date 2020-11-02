#ifndef BOX_FUSION_H_
#define BOX_FUSION_H_

/// util
#include "car_model.h"
#include <camera_params.h>
#include "drivenet/image_preprocessing.h"
#include <msgs/DetectedObjectArray.h>
#include <msgs/DetectedObject.h>

struct CheckArea
{
  cv::Point LeftLinePoint1;
  cv::Point LeftLinePoint2;
  cv::Point RightLinePoint1;
  cv::Point RightLinePoint2;
};

class Boxfusion
{
private:
  int image_w_ = camera::image_width;
  int image_h_ = camera::image_height;
  CheckArea front_bottom, left_back;
  int pixelthres = 40;

  static constexpr int FB_left_top_x = 0;
  static constexpr int FB_left_top_y = 821;
  static constexpr int FB_right_bottom_x = 938;
  static constexpr int FB_right_bottom_y = 1207;

  static constexpr int LB_left_top_x = 1115;
  static constexpr int LB_left_top_y = 135;
  static constexpr int LB_right_bottom_x = 1832;
  static constexpr int LB_right_bottom_y = 340;

public:
  Boxfusion();
  ~Boxfusion();
  std::vector<msgs::DetectedObjectArray> boxfuse(std::vector<msgs::DetectedObjectArray> ori_object_arrs,
                                                 int camera_id_1, int camera_id_2);
  msgs::DetectedObjectArray fusetwocamera(msgs::DetectedObjectArray obj1, msgs::DetectedObjectArray obj2);
  int CheckPointInArea(CheckArea area, int object_x1, int object_y2);
  bool pointcompare(DriveNet::PixelPosition front_bottom, DriveNet::PixelPosition projected);
};

#endif
