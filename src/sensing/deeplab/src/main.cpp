#include "deeplab_node.h"
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <glog/logging.h>
#include <ros/ros.h>
#include <opencv2/core/utility.hpp>

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "deeplab");
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  cv::setNumThreads(0);  // Avoid excessive CPU load
  deeplab::DeeplabNode deeplab_node;

  deeplab_node.run(argc, argv);

  return 0;
}
