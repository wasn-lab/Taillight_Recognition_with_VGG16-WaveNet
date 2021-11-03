#pragma once

#include <string>
#include "image_compressor.h"

namespace image_compressor
{
// Getters
std::string get_input_topic();
std::string get_output_topic();
compression_format get_compression_format();
int32_t get_quality();

};  // namespace image_compressor
