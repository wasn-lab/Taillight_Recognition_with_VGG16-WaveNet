#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "pcd_to_pc2_args_parser.h"

namespace pcd_to_pc2
{
DEFINE_string(topic, "/LidarFrontTop/Raw", "Output topic name");
DEFINE_string(pcd, "", "pcd file path");
DEFINE_string(frame_id, "lidar", "frame id in point cloud message");
DEFINE_int32(fps, 20, "number of messages per second");

std::string get_output_topic()
{
  return FLAGS_topic;
}

std::string get_frame_id()
{
  return FLAGS_frame_id;
}

std::string get_pcd_path()
{
  return FLAGS_pcd;
}

uint32_t get_fps()
{
  return FLAGS_fps;
}

};  // namespace pcd_to_pc2
