/*
   CREATER: ICL U300
   DATE: August, 2019
 */

#ifndef __CAMERA_PARAMS_H__
#define __CAMERA_PARAMS_H__
#include <cmath>
#include <string>

namespace camera
{
enum id
{
  begin = 0,
  left_60 = begin,  // 0
  front_60,         // 1
  right_60,         // 2
  _dummy1,          // 3
  left_120,         // 4
  front_120,        // 5
  right_120,        // 6
  _dummy2,          // 7
  left_30,          // 8
  front_30,         // 9
  right_30,         // 10
  _dummy3,          // 11
  num_ids           // 12
};

static_assert(id::begin == 0, "The first camera id is 1");
static_assert(id::left_60 == 0, "The camera id 0 is also left_60");
static_assert(id::num_ids == 12, "The number of ids is 12");

extern const std::string names[id::num_ids];
extern const std::string topics[id::num_ids];

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

};      // namespace
#endif  // __CAMERA_PARAMS_H__
