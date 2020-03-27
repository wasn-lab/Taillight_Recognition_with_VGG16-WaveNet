#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "pcd_saver_args_parser.h"

namespace pcd_saver
{
DEFINE_string(pcd_topic, "/LidarAll", "Save point cloud at this topic");

std::string get_pcd_topic()
{
  return FLAGS_pcd_topic;
}

};  // namespace pcd_saver
