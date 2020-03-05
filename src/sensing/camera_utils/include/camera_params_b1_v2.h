/*
   CREATER: ICL U300
   DATE: Oct, 2019
 */
#ifndef __CAMERA_PARAMS_B1_V2_H__
#define __CAMERA_PARAMS_B1_V2_H__

#include "car_model.h"
#ifdef CAR_MODEL_IS_B1_V2
#include <cmath>
#include <string>

namespace camera
{
// B1 car: 5*FOV60 + 2*FOV120 + 1*FOV30
enum id
{
  begin = 0,
  front_bottom_60 = begin,  // 0
  front_top_far_30,         // 1
  _dummy0,                  // 2  // NOLINT
  _dummy1,                  // 3  // NOLINT
  front_top_close_120,      // 4
  right_front_60,           // 5
  right_back_60,            // 6
  _dummy2,                  // 7  // NOLINT
  left_front_60,            // 8
  left_back_60,             // 9
  back_top_120,             // 10
  _dummy3,                  // 11  // NOLINT
  num_ids                   // 12
};

static_assert(id::begin == 0, "The first camera id is 0");
static_assert(id::front_bottom_60 == 0, "The camera id 0 is also front_bottom_60");
static_assert(id::num_ids == 12, "The number of ids is 12");

extern const std::string names[id::num_ids];
extern const std::string topics[id::num_ids];
extern const std::string topics_obj[id::num_ids];
extern const bool distortion[id::num_ids];
extern const std::string detect_result;
extern const std::string detect_result_polygon;
extern const std::string detect_result_occupancy_grid;

// TODO: fill in the following parameters.
constexpr int raw_image_width = 1920;
constexpr int raw_image_height = 1208;
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
constexpr int image_height = 384;
constexpr int image_rows = raw_image_height;
constexpr int image_cols = raw_image_width;
constexpr int num_image_pixels = image_width * image_height;
constexpr int num_image_bytes = image_width * image_height * 3;

// Parameters for resizing 1920x1208 to 608x608(yolov3 default size)
constexpr double image_ratio_on_yolov3 = 608.0 / raw_image_width;
constexpr double inv_image_ratio_on_yolov3 = raw_image_width / 608.0;
constexpr int yolov3_letterbox_visible_height = image_ratio_on_yolov3 * raw_image_height;  // 382
constexpr int left_border = std::abs(raw_image_width * image_ratio_on_yolov3 - yolov3_image_width) / 2;
constexpr int right_border = left_border;
constexpr int top_border = std::abs(raw_image_height * image_ratio_on_yolov3 - yolov3_image_height) / 2;
constexpr int bottom_border = top_border + 1;
// NPP library has different rounding from opencv.
constexpr int npp_top_border = 1 + top_border;
constexpr int npp_bottom_border = npp_top_border;
static_assert(yolov3_letterbox_visible_height + npp_top_border + npp_top_border == yolov3_image_height,
              "visible height + border should be 608");

// When input is 608x384
constexpr int top_border_608x384 = (yolov3_image_height - 384) / 2;
constexpr int bottom_border_608x384 = top_border_608x384;
}  // namespace camera
#endif  // CAR_MODEL_IS_B1_V2
#endif  // __CAMERA_PARAMS_B1_V2_H__
