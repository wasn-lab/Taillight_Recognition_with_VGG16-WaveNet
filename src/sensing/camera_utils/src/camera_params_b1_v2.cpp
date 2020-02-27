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
  "F_center", "F_top_far", "", "", "F_top_close", "R_front", "R_back", "", "L_front", "L_back", "B_top", "",
};

const std::string topics[id::num_ids] = {
  "/cam/F_center", "/cam/F_top_far", "",           "", "/cam/F_top_close", "/cam/R_front", "/cam/R_back", "",
  "/cam/L_front",  "/cam/L_back",    "/cam/B_top", "",
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

const bool distortion[id::num_ids] = {
  false,  // F_center
  false,  // F_top_far
  false,  //
  false,  //
  true,  // F_top_close
  false,  // R_front
  false,  // R_rear
  false,  //
  false,  // L_front
  false,  // L_rear
  true,  // B_top
  false,  //
};

const std::string detect_result = "/CameraDetection";
const std::string detect_result_polygon = "/CameraDetection/polygon";
const std::string detect_result_occupancy_grid = "/CameraDetection/occupancy_grid";

};  // namespace

#endif  // CAR_MODEL_IS_B1
