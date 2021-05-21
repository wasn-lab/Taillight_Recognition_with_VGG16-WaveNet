#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <ros/ros.h>
#include <glog/logging.h>            // for COMPACT_GOOGLE_LOG_INFO, LOG
#include <opencv2/core/utility.hpp>  // for setNumThreads
#include <ostream>                   // for operator<<
#include <string>                    // for string
#include "xwin_grabber.h"            // for XWinGrabber
#include "xwin_grabber_args_parser.h"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "xwin_grabber_node", ros::init_options::AnonymousName);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  cv::setNumThreads(0);
  auto grabber = xwin_grabber::XWinGrabber(xwin_grabber::get_window_title());
  return grabber.run();
}
