#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <ros/package.h>
#include "deeplab_args_parser.h"

namespace deeplab
{
DEFINE_string(image_topic, "/cam/front_bottom_60", "Input image topic");
DEFINE_string(pb_file, "",
              "The pb file that is exported by deeplab model. If not set, then the default pb will be used.");
DEFINE_bool(profiling_mode, false, "In profiling mode, node runs for a few seconds and then shut down.");

std::string get_image_topic()
{
  return FLAGS_image_topic;
}

std::string get_pb_file()
{
  if (FLAGS_pb_file.size() == 0)
  {
    return ros::package::getPath("deeplab") + "/weights/frozen_inference_graph.pb";
  }
  return FLAGS_image_topic;
}

bool in_profiling_mode()
{
  return FLAGS_profiling_mode;
}

};  // namespace deeplab
