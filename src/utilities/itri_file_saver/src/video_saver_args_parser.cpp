#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "video_saver_args_parser.h"

namespace video_saver
{
DEFINE_string(image_topic, "/cam/F_center", "Save images into video at this topic");
DEFINE_string(output_filename, "out.avi", "Video output file");
DEFINE_double(frame_rate, 20.0, "Video frame rate");

std::string get_image_topic()
{
  return FLAGS_image_topic;
}

std::string get_output_filename()
{
  return FLAGS_output_filename;
}

double get_frame_rate()
{
  return FLAGS_frame_rate;
}
};  // namespace video_saver
