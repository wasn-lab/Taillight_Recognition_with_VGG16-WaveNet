/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */

#ifndef __PARKNET_CAMERA_H__
#define __PARKNET_CAMERA_H__
#include <string>
#include "parknet.h"
#include "camera_params.h"

namespace parknet
{
enum camera
{
  left_120_e,
  front_120_e,
  right_120_e,
  num_cams_e
};

static_assert(left_120_e == 0, "");
static_assert(num_cams_e == 3, "");

constexpr int expected_fps = 30;
constexpr float displayed_image_scale = 0.25;
constexpr unsigned int all_cameras_done_detection = (1 << camera::num_cams_e) - 1;
static_assert(all_cameras_done_detection == 7, "");

extern const std::string camera_names[num_cams_e];
extern const ::camera::id camera_id_mapping[num_cams_e];
};  // namespace parknet

#endif  // __PARKNET_CAMERA_H__
