/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include "msg_replay_node.h"
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <ros/ros.h>
#include <glog/logging.h>

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "msg_replay_node", ros::init_options::AnonymousName);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  msg_replay::MsgReplayNode node;
  node.run();

  return 0;
}
