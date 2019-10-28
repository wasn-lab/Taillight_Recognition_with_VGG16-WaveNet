#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <assert.h>
#include "parknet_args_parser.h"
#include "parknet.h"

namespace alignment_args_parser
{
DEFINE_int32(pcd_nums, 2380, "Number of pcds to be used.");
DEFINE_int32(cam_sn, 5, "Camera serial number");
DEFINE_string(output_filename, "out.json", "Output file name");

int get_pcd_nums()
{
  assert(FLAGS_pcd_nums > 0);
  return FLAGS_pcd_nums;
}

int get_cam_sn()
{
  assert(FLAGS_cam_sn >= 1);
  assert(FLAGS_cam_sn <= 9);
  return FLAGS_cam_sn;
}

std::string get_output_filename()
{
  return FLAGS_output_filename;
}

};  // namespace
