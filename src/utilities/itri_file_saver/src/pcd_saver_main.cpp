#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <ros/ros.h>
#include <glog/logging.h>
#include "pcd_saver_node.h"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "pcd_saver");
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  PCDSaverNode app;
  app.run();

  return 0;
}
