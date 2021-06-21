#pragma once

#include <string>
#include "image_compressor.h"

namespace image_compressor
{
// Getters
std::string get_input_topic();
std::string get_output_topic();
bool should_verify_decompressed_data();
bool is_verbose();
void set_verbose(bool mode);
compression_format get_compression_format();

};  // namespace image_compressor
