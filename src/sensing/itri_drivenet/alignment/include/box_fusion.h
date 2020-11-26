#ifndef BOX_FUSION_H_
#define BOX_FUSION_H_

/// util
#include "car_model.h"
#include <camera_params.h>
#include "drivenet/image_preprocessing.h"
#include <msgs/DetectedObjectArray.h>
#include <msgs/DetectedObject.h>
#include <msgs/BoxPoint.h>

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
  int pixelthres_ = 40;
  float iou_threshold_ = 0;

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
  std::vector<msgs::DetectedObjectArray> box_fuse(std::vector<msgs::DetectedObjectArray> ori_object_arrs,
                                                 int camera_id_1, int camera_id_2);
  msgs::DetectedObjectArray fuse_two_camera(msgs::DetectedObjectArray obj1, msgs::DetectedObjectArray obj2);
  int check_point_in_area(CheckArea area, int object_x1, int object_y2);
  bool point_compare(DriveNet::PixelPosition front_bottom, DriveNet::PixelPosition projected);
  std::vector<msgs::DetectedObject> multi_cambox_fuse(std::vector<msgs::DetectedObject>& input_obj_arrs);
  float iou_compare_with_heading(msgs::DetectedObject& obj1, msgs::DetectedObject& obj2);
  msgs::BoxPoint redefine_bounding_box(msgs::BoxPoint origin_box);
  
  
  
};

#endif
