#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <ros/ros.h>
#include <glog/logging.h>
#include "image_compressor_node.h"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "image_compressor_node", ros::init_options::AnonymousName);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  image_compressor::ImageCompressorNode node;
  node.run();

  return 0;
}
