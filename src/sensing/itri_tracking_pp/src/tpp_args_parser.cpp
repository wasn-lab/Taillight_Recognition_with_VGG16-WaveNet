#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "tpp_args_parser.h"

namespace tpp
{
// 1: /LidarDetection
// 2: /RadarDetection
// 3: /DetectedObjectArray/cam60_1
// otherwise: /SensorFusion
DEFINE_int32(in_source, 0, "TPP input source");
DEFINE_bool(ego_speed, true, "TPP use ego speed");
DEFINE_bool(draw_pp, false, "TPP draw pp on test canvas");

int get_in_source()
{
  return int(FLAGS_in_source);
}

bool get_ego_speed()
{
  return FLAGS_ego_speed;
}

bool get_draw_pp()
{
  return FLAGS_draw_pp;
}

};  // namespace tpp