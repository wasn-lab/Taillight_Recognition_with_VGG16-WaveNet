#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "image_saver_args_parser.h"

namespace image_saver
{
DEFINE_string(image_topic, "/cam/front_bottom_60", "Save images at this topic");
DEFINE_string(output_dir, ".", "Where to save images.");

std::string get_image_topic()
{
  return FLAGS_image_topic;
}

std::string get_output_dir()
{
  return FLAGS_output_dir;
}

};  // namespace image_saver
