#include <cstdlib>
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "pc2_args_parser.h"

namespace pc2_compressor
{
DEFINE_string(input_topic, "", "Input topic name");
DEFINE_string(output_topic, "", "Output topic name");
DEFINE_string(compression_format, "lzf", "Compression format [none, lzf, snappy, zlib]");
DEFINE_bool(verify, false, "Verify if the compressed data can be restored back");
DEFINE_bool(verbose, false, "Verbose mode: print more logs for debuggin.");
DEFINE_bool(use_threading, true, "Use threads to do compression.");

std::string get_input_topic()
{
  return FLAGS_input_topic;
}

std::string get_output_topic()
{
  return FLAGS_output_topic;
}

std::string get_compression_format()
{
  return FLAGS_compression_format;
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

};  // namespace pc2_compressor
