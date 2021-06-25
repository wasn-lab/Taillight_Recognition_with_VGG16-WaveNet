#pragma once
#include <string>

namespace xwin_grabber
{
std::string get_window_title();

std::string get_output_topic();
bool should_publish_raw_image();
};  // namespace xwin_grabber
