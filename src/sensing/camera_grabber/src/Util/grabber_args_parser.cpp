#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "grabber_args_parser.h"

namespace SensingSubSystem
{
DEFINE_string(mode, "a", "Specify the mode in [a, b]");
DEFINE_int32(expected_fps, 30, "Expected frames per seconds");
DEFINE_bool(feed_608, true, "Resize image size from 1920x1208 to 608x608");

std::string get_mode()
{
  return FLAGS_mode;
}

int get_expected_fps()
{
  return FLAGS_expected_fps;
}

bool should_feed_608()
{
  return FLAGS_feed_608;
}

};  // namespace
