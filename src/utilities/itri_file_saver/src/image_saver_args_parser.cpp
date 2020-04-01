#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "image_saver_args_parser.h"

namespace image_saver
{
DEFINE_string(image_topic, "/cam/F_center", "Save images at this topic");

std::string get_image_topic()
{
  return FLAGS_image_topic;
}

};  // namespace image_saver
