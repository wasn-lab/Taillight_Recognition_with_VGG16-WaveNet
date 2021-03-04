/*
   CREATER: ICL U300
   DATE: Oct, 2019
 */
#ifndef __CAMERA_PARAMS_C1_H__
#define __CAMERA_PARAMS_C1_H__

#include "car_model.h"
#if CAR_MODEL_IS_C1
#include <cmath>
#include <string>

namespace camera
{
// C1 car: 5*FOV60 + 2*FOV120 + 1*FOV30
enum id
{
  begin = 0,
  front_bottom_60 = begin,  // 0
  front_top_far_30,         // 1
  front_top_close_120,      // 2  
  right_front_60,           // 3
  right_back_60,            // 4
  left_front_60,            // 5  
  left_back_60,             // 6
  back_top_120,             // 7
  num_ids                   // 8                      
};

static_assert(id::begin == 0, "The first camera id is 0");
static_assert(id::front_bottom_60 == 0, "The camera id 0 is also front_bottom_60");
static_assert(id::num_ids == 8, "The number of ids is 8");

extern const std::string names[id::num_ids];
extern const std::string topics[id::num_ids];
extern const std::string topics_obj[id::num_ids];
extern const bool distortion[id::num_ids];
extern const std::string detect_result;
extern const std::string detect_result_occupancy_grid;

// TODO: fill in the following parameters.
//This is for Gstreamer
constexpr int raw_image_width = 1280;
constexpr int raw_image_height = 720;
constexpr int raw_image_rows = raw_image_height;
constexpr int raw_image_cols = raw_image_width;
constexpr int num_raw_image_pixels = raw_image_width * raw_image_height;
constexpr int num_raw_image_bytes = raw_image_width * raw_image_height * 3;
constexpr int yolov3_image_width = 608;
constexpr int yolov3_image_height = 608;
constexpr int yolov3_image_rows = 608;
constexpr int yolov3_image_cols = 608;
constexpr int yolov3_image_center_x = 608 >> 1;
constexpr int yolov3_image_center_y = 608 >> 1;
constexpr int num_yolov3_image_pixels = yolov3_image_width * yolov3_image_height;
constexpr int num_yolov3_image_bytes = yolov3_image_width * yolov3_image_height * 3;
constexpr int num_yolov3_bytes_per_row_u8 = yolov3_image_width * 3;
constexpr int num_yolov3_bytes_per_row_f32 = yolov3_image_width * 3 * sizeof(float);
constexpr int image_width = 608;
constexpr int image_height = 342; //This is for 1280x720 resolution
constexpr int image_rows = raw_image_height;
constexpr int image_cols = raw_image_width;
constexpr int num_image_pixels = image_width * image_height;
constexpr int num_image_bytes = image_width * image_height * 3;

// Parameters for resizing 1280x720 to 608x608(yolov3 default size)
constexpr double image_ratio_on_yolov3 = 608.0 / raw_image_width;
constexpr double inv_image_ratio_on_yolov3 = raw_image_width / 608.0;
constexpr int yolov3_letterbox_visible_height = image_ratio_on_yolov3 * raw_image_height;  // 382
constexpr int left_border = std::abs(raw_image_width * image_ratio_on_yolov3 - yolov3_image_width) / 2;
constexpr int right_border = left_border;
constexpr int top_border = std::abs(raw_image_height * image_ratio_on_yolov3 - yolov3_image_height) / 2;
static_assert(top_border == 133, "top border should be 133");
constexpr int bottom_border = top_border;
// NPP library has different rounding from opencv.
constexpr int npp_top_border = top_border;
constexpr int npp_bottom_border = npp_top_border;
static_assert(yolov3_letterbox_visible_height + npp_top_border + npp_top_border == yolov3_image_height,
              "visible height + border should be 608");
}  // namespace camera
#endif  // CAR_MODEL_IS_C1
#endif  // __CAMERA_PARAMS_C1_H__
