#include <cstdlib>
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "image_compressor_args_parser.h"

namespace image_compressor
{
DEFINE_string(input_topic, "", "Input topic name");
DEFINE_string(output_topic, "", "Output topic name");
DEFINE_bool(use_png, false, "Use png compression (10x slower and 8x bigger than jpg, but lossless");
DEFINE_int32(quality, 85, "Set quality (jpg: 1~100, png:1~10)");

std::string get_input_topic()
{
  return FLAGS_input_topic;
}

std::string get_output_topic()
{
  return FLAGS_output_topic;
}

int32_t get_quality()
{
  return FLAGS_quality;
}

compression_format get_compression_format()
{
  return static_cast<compression_format>(FLAGS_use_png);
}

};  // namespace image_compressor
