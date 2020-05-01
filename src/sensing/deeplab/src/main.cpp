#include "deeplab_node.h"
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <glog/logging.h>
#include <ros/ros.h>

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "deeplab");
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  deeplab::DeeplabNode deeplab_node;

  deeplab_node.run(argc, argv);

  return 0;
}
