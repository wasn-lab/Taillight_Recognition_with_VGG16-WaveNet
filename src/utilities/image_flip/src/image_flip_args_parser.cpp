#include <cstdlib>
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "image_flip_args_parser.h"

namespace image_flip
{
DEFINE_string(input_topic, "", "Input topic name");
DEFINE_string(output_topic, "", "Output topic name");

std::string get_input_topic()
{
  return FLAGS_input_topic;
}

std::string get_output_topic()
{
  return FLAGS_output_topic;
}
};  // namespace image_flip
