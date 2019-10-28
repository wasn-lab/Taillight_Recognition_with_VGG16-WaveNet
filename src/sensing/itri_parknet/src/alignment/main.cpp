/*
   ICL U300
   Feb, 2019
 */

#include <memory>
#include "parknet.h"
#include "parknet_alignment_node.h"
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <ros/ros.h>
#include <glog/logging.h>

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "gound_points_mapping");
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  std::unique_ptr<ParknetAlignmentNode> app(new ParknetAlignmentNode());
  app->run(argc, argv);

  return 0;
}
