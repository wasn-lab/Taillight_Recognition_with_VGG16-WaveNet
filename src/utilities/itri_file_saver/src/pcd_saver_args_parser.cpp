#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "pcd_saver_args_parser.h"

namespace pcd_saver
{
DEFINE_string(pcd_topic, "/LidarAll", "Save point cloud at this topic");
DEFINE_bool(save_as_ascii, true, "Save point cloud in ascii format");

std::string get_pcd_topic()
{
  return FLAGS_pcd_topic;
}

bool save_as_ascii()
{
  return FLAGS_save_as_ascii;
}

};  // namespace pcd_saver
