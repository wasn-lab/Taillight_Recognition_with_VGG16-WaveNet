#include <cstdlib>
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "image_compressor_args_parser.h"

namespace image_compressor
{
DEFINE_string(input_topic, "", "Input topic name");
DEFINE_string(output_topic, "", "Output topic name");
DEFINE_bool(verify, false, "Verify if the compressed data can be restored back");
DEFINE_bool(verbose, false, "Verbose mode: print more logs for debuggin.");
DEFINE_bool(use_threading, false, "Use threads to do compression.");
DEFINE_bool(use_png, false, "Use png compression (10x slower and 8x bigger than jpg, but lossless");

std::string get_input_topic()
{
  return FLAGS_input_topic;
}

std::string get_output_topic()
{
  return FLAGS_output_topic;
}

bool should_verify_decompressed_data()
{
  return FLAGS_verify;
}

bool use_threading()
{
  return FLAGS_use_threading;
}

bool is_verbose()
{
  return FLAGS_verbose;
}

void set_verbose(bool mode)
{
  FLAGS_verbose = mode;
}

compression_format get_compression_format()
{
  return static_cast<compression_format>(FLAGS_use_png);
}

void set_use_threading(bool mode)
{
  FLAGS_use_threading = mode;
}

};  // namespace image_compressor
