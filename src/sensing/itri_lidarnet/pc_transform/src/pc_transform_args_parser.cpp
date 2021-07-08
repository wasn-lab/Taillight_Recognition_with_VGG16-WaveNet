/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "pc_transform_args_parser.h"

namespace pc_transform
{
DEFINE_string(input_topic, "", "Input topic name");
DEFINE_string(output_topic, "", "Output topic name");
DEFINE_string(transform_param_name, "", "ROS param that defines transform parameters");

std::string get_input_topic()
{
  return FLAGS_input_topic;
}

std::string get_output_topic()
{
  return FLAGS_output_topic;
}

std::string get_transform_param_name()
{
  return FLAGS_transform_param_name;
}

};  // namespace pc_transform
