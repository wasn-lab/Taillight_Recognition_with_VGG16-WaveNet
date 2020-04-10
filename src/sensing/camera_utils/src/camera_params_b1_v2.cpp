/*
CREATER: ICL U300
DATE: Oct, 2019
*/
#include "camera_params.h"

#if CAR_MODEL_IS_B1_V2 || CAR_MODEL_IS_OMNIBUS
namespace camera
{
const std::string names[id::num_ids] = {
  // New b1
  "front_bottom_60",
  "front_top_far_30",
  "front_bottom_60_crop",
  "",
  "front_top_close_120",
  "right_front_60",
  "right_back_60",
  "",
  "left_front_60",
  "left_back_60",
  "back_top_120",
  "",
};

const std::string topics[id::num_ids] = {
  "/cam/front_bottom_60",
  "/cam/front_top_far_30",
  "/cam/front_bottom_60_crop",
  "",
  "/cam/front_top_close_120",
  "/cam/right_front_60",
  "/cam/right_back_60",
  "",
  "/cam/left_front_60",
  "/cam/left_back_60",
  "/cam/back_top_120",
  "",
};

const std::string topics_obj[id::num_ids] = {
  "/cam_obj/front_bottom_60",
  "/cam_obj/front_top_far_30",
  "/cam_obj/front_bottom_60_crop",
  "",
  "/cam_obj/front_top_close_120",
  "/cam_obj/right_front_60",
  "/cam_obj/right_back_60",
  "",
  "/cam_obj/left_front_60",
  "/cam_obj/left_back_60",
  "/cam_obj/back_top_120",
  "",
};

const bool distortion[id::num_ids] = {
  false,  // front_bottom_60
  false,  // front_top_far_30
  false,  // front_bottom_60_crop
  false,  //
  true,   // front_top_close_120
  false,  // right_front_60
  false,  // right_back_60
  false,  //
  false,  // left_front_60
  false,  // left_back_60
  true,   // back_top_120
  false,  //
};

const std::string detect_result = "/CameraDetection";
const std::string detect_result_polygon = "/CameraDetection/polygon";
const std::string detect_result_occupancy_grid = "/CameraDetection/occupancy_grid";

}  // namespace camera

#endif  // CAR_MODEL_IS_B1
