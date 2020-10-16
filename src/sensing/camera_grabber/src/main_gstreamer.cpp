#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <glog/logging.h>
#include "TegraCGrabber.h"
#include "Util/ProgramArguments.hpp"
#include "grabber_args_parser.h"

int main(int argc, char** argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  cv::setNumThreads(0);

  // mode selection
  auto mode = SensingSubSystem::get_mode();

  // do resize
  const bool do_resize = SensingSubSystem::do_resize();
 

  if (mode == "c")
  {
    //Gstreamer
    ros::init(argc, argv, "camera_c_grabber");
    printf("Running Camera c grabber (gstreamer)\n");
    SensingSubSystem::TegraCGrabber app;
    if(app.initializeModulesGst(do_resize) == false)
    {
      //std::cout <<"initializeModulesGst fail" << std::endl;
      return -1;
    }
    return app.runPerceptionGst();
  }
  else
  {
    LOG(WARNING) << "Unknown or unsupported mode: " << mode;
    return -1;
  }
   
}
