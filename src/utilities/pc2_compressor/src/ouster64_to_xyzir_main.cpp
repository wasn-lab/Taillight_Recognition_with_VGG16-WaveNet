/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <ros/ros.h>
#include <glog/logging.h>
#include "ouster64_to_xyzir_node.h"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "ouster64_to_xyzir_node", ros::init_options::AnonymousName);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  pc2_compressor::Ouster64ToXYZIRNode node;
  node.run();

  return 0;
}
