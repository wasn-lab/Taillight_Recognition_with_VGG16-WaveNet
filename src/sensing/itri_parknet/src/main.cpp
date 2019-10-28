/*
   ICL U300
   Feb, 2019
 */

#include "parknet.h"
#include "parknet_node.h"
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <ros/ros.h>
#include <glog/logging.h>
#include <opencv2/core/utility.hpp>

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "pslot_detector");
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  cv::setNumThreads(0);
  ParknetNode app;
  app.run(argc, argv);

  return 0;
}
