#pragma once

#include <string>

namespace pc2_compressor
{
// Getters
std::string get_input_topic();
std::string get_output_topic();
bool should_verify_decompressed_data();
bool use_threading();
bool is_verbose();
void set_verbose(bool mode);
std::string get_compression_format();

};  // namespace pc2_compressor
