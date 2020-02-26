/*
CREATER: ICL U300
DATE: Oct, 2019
*/
#include "camera_params.h"

#if CAR_MODEL_IS_B1_V2
namespace camera
{
const std::string names[id::num_ids] = {
  // New b1
  "F_center", "F_top_far", "", "",
  "F_top_close", "R_front", "R_rear", "",
  "L_front", "L_rear", "B_top", "",
};

const std::string topics[id::num_ids] = {
  "/cam/F_center",
  "/cam/F_top_far",
  "",
  "",
  "/cam/F_top_close",
  "/cam/R_front",
  "/cam/R_rear",
  "",
  "/cam/L_front",
  "/cam/L_rear",
  "/cam/B_top",
  "",
};

const std::string topics_obj[id::num_ids] = {
  "/CamObjFrontCenter",
  "/CamObjFrontTopFar",
  "",
  "",
  "/CamObjFrontTopClose",
  "/CamObjRightFront",
  "/CamObjRightBack",
  "",
  "/CamObjLeftFront",
  "/CamObjLeftBack",
  "/CamObjBackTop",
  "",
};

const int distortion[id::num_ids] = {
  0, // F_center
  0, // F_top_far
  0, // 
  0, //
  1, // F_top_close
  0, // R_front
  0, // R_rear
  0, //
  0, // L_front
  0, // L_rear
  1, // B_top
  0, //
};

const std::string detect_result = "/CameraDetection";
const std::string detect_result_polygon = "/CameraDetection/polygon";
const std::string detect_result_occupancy_grid = "/CameraDetection/occupancy_grid";

};  // namespace

#endif  // CAR_MODEL_IS_B1
