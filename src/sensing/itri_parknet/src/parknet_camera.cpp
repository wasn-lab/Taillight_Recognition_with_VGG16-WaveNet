/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */
#include "parknet_camera.h"

namespace parknet
{
const std::string camera_names[num_cams_e] = {
  "left_120", "front_120", "right_120",
};
#if CAR_MODEL_IS_HINO
const ::camera::id camera_id_mapping[num_cams_e] = { ::camera::id::left_120, ::camera::id::front_120,
                                                     ::camera::id::right_120 };
#elif CAR_MODEL_IS_B1 || CAR_MODEL_IS_OMNIBUS
const ::camera::id camera_id_mapping[num_cams_e] = { ::camera::id::top_right_front_120, ::camera::id::top_front_120,
                                                     ::camera::id::top_right_rear_120 };
#else
#error "unreachable"
#endif
};  // namespace parknet
