/*
CREATER: ICL U300
DATE: Oct, 2019
*/
#include "camera_params.h"

#if CAR_MODEL_IS_B1
namespace camera
{
const std::string names[id::num_ids] = {
  "left_60", "front_60", "right_60", "",
  "left_120", "front_120", "right_120", "",
  "__left_120", "__front_120", "__right_120", "",
};

const std::string topics[id::num_ids] = {
  "/gmsl_camera/left_60",
  "/gmsl_camera/front_60",
  "/gmsl_camera/right_60",
  "",
  "/gmsl_camera/left_120",
  "/gmsl_camera/front_120",
  "/gmsl_camera/right_120",
  "",
  "/gmsl_camera/__left_120",
  "/gmsl_camera/__front_120",
  "/gmsl_camera/__right_120",
  "",
};

};  // namespace

#endif  // CAR_MODEL_IS_B1
