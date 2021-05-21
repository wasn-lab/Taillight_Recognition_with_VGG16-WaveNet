#include <cstdlib>
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "xwin_grabber_args_parser.h"

namespace xwin_grabber
{
DEFINE_string(window_title, "rviz", "Grabber the contents of the window with the given title");
DEFINE_string(output_topic, "/xwin_grabber", "Output topic name");

std::string get_window_title()
{
  return FLAGS_window_title;
}

std::string get_output_topic()
{
  return FLAGS_output_topic;
}
};  // namespace xwin_grabber
