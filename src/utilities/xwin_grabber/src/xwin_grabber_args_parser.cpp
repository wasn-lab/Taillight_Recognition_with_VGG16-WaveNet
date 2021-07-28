#include <cstdlib>
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "xwin_grabber_args_parser.h"

namespace xwin_grabber
{
DEFINE_string(window_title, "rviz", "Grabber the contents of the window with the given title");
DEFINE_int32(publish_raw_image, 0, "Publish raw images");

std::string get_window_title()
{
  return FLAGS_window_title;
}

std::string get_output_topic()
{
  return "/xwin_grabber/" + FLAGS_window_title;
}

bool should_publish_raw_image()
{
  return bool(FLAGS_publish_raw_image);
}
};  // namespace xwin_grabber
