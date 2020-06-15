#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "tpp_args_parser.h"

namespace tpp
{
// 1: /LidarDetection
// 2: /RadarDetection
// 3: /CamObjFrontCenter
// otherwise: /SensorFusion
DEFINE_int32(in_source, 0, "TPP input source");
DEFINE_bool(ego_speed, true, "TPP use ego speed");

int get_in_source()
{
  return int(FLAGS_in_source);
}

bool get_ego_speed()
{
  return FLAGS_ego_speed;
}

};  // namespace tpp