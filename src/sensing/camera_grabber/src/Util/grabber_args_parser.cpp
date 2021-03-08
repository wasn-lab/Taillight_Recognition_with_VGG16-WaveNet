#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "grabber_args_parser.h"
#include <iostream>

namespace SensingSubSystem
{
DEFINE_string(mode, "a", "Specify the mode in [a, b], default: a");
DEFINE_int32(expected_fps, 30, "Expected frames per seconds, default: 30");
DEFINE_bool(feed_608, true, "Resize image size from 1920x1208 to 608x608, default: true");
DEFINE_bool(do_resize, true, "Resize image size, default: true");
DEFINE_bool(do_crop, true, "Crop image, default: true");
DEFINE_string(password, "nvidia", "Specify the sudo password for ar0231 camera driver , default: nivida");
DEFINE_bool(car_driver, true, "true :car camera driver; false: laboratory camera driver, default: true");
DEFINE_bool(motion_vector, false, "true :enable Mv extractor; false: disable Mv extractor, default: false");

std::string get_mode()
{
  return FLAGS_mode;
}

int get_expected_fps()
{
  return FLAGS_expected_fps;
}

bool should_feed_608()
{
  return FLAGS_feed_608;
}

bool do_resize()
{
  return FLAGS_do_resize;
}

bool do_crop()
{
  return FLAGS_do_crop;
}

std::string get_password()
{
  return FLAGS_password;
}

bool car_driver()
{
  return FLAGS_car_driver;
}

bool motion_vector()
{
  return FLAGS_motion_vector;
}

};  // namespace
