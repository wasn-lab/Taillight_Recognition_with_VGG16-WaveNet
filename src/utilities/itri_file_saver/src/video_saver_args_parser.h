#pragma once
#include <string>
namespace video_saver
{
// Getters

std::string get_image_topic();
std::string get_output_filename();
int get_frame_width();
int get_frame_height();
double get_frame_rate();

};  // namespace video_saver
