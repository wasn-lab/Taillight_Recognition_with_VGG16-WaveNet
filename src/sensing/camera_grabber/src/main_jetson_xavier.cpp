#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <glog/logging.h>
#include "JetsonXavierGrabber.h"
#include "Util/ProgramArguments.hpp"
#include "grabber_args_parser.h"

int main(int argc, char** argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  cv::setNumThreads(0);

  // do resize
  const bool do_resize = SensingSubSystem::do_resize();

  // Gstreamer
  ros::init(argc, argv, "camera_grabber");
  std::cout << "Running Camera grabber (gstreamer)" << std::endl;
  SensingSubSystem::JetsonXavierGrabber app;
  if (app.initializeModulesGst(do_resize) == false)
  {
    std::cout << "initializeModulesGst fail" << std::endl;
    return -1;
  }

  return app.runPerceptionGst();
}
