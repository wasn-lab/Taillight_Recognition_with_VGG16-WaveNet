/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <cstdlib>
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "args_parser.h"

namespace msg_replay
{
DEFINE_string(input_topic, "", "Input topic name");
DEFINE_string(output_topic, "", "Output topic name");

std::string get_input_topic()
{
  return FLAGS_input_topic;
}

std::string get_output_topic()
{
  return FLAGS_output_topic;
}

};  // namespace msg_replay
