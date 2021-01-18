#include "pc2_compressor_node.h"
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <ros/ros.h>
#include <glog/logging.h>

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "pc2_compressor_node", ros::init_options::AnonymousName);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  pc2_compressor::PC2CompressorNode node;
  node.run();

  return 0;
}
