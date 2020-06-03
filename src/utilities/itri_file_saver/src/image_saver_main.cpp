#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <ros/ros.h>
#include <glog/logging.h>
#include <opencv2/core/utility.hpp>
#include "image_saver_node.h"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "file_saver", ros::init_options::AnonymousName);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  LOG(INFO) << "OpenCV Version: " << CV_VERSION;
  image_saver::ImageSaverNode app;
  app.run();

  return 0;
}
