/*
CREATER: ICL U300
DATE: Oct, 2019
*/
#include "camera_params.h"

#if CAR_MODEL_IS_B1
namespace camera
{
const std::string names[id::num_ids] = {
  "F_right", "F_center", "F_left", "", "F_top", "R_front", "R_rear", "", "L_front", "L_rear", "B_top", "",
};

const std::string topics[id::num_ids] = {
  "/cam/F_right", "/cam/F_center", "/cam/F_left", "", "/cam/F_top", "/cam/R_front", "/cam/R_rear", "",
  "/cam/L_front", "/cam/L_rear",   "/cam/B_top",  "",
};

const std::string topics_obj[id::num_ids] = {
  "/CamObjFrontRight", "/CamObjFrontCenter", "/CamObjFrontLeft", "",
  "/CamObjFrontTop",   "/CamObjRightFront",  "/CamObjRightBack", "",
  "/CamObjLeftFront",  "/CamObjLeftBack",    "/CamObjBackTop",   "",
};

const std::string detect_result = "/CameraDetection";
const std::string detect_result_polygon = "/CameraDetection/polygon";
const std::string detect_result_occupancy_grid = "/CameraDetection/occupancy_grid";

const bool distortion[id::num_ids] = {
  false,  // F_right
  false,  // F_center
  false,  // F_left
  false,  //
  true,   // F_top
  true,   // R_front
  true,   // R_rear
  false,  //
  true,   // L_front
  true,   // L_rear
  true,   // B_top
  false,  //
};

};  // namespace

#endif  // CAR_MODEL_IS_B1
