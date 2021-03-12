/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <ros/ros.h>
#include <glog/logging.h>
#include "pcd_to_pc2_node.h"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "pcd_to_pc2_node", ros::init_options::AnonymousName);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  pcd_to_pc2::PCDToPc2Node node;
  node.run();

  return 0;
}
