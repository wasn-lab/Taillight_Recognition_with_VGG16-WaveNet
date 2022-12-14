#include "track2d_node.h"
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#if USE_GLOG == 1
#include <glog/logging.h>
#endif

int main(int argc, char** argv)
{
  ros::init(argc, argv, "itri_tracking_2d", ros::init_options::AnonymousName);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
#if USE_GLOG == 1
  google::InstallFailureSignalHandler();
#endif
  track2d::Track2DNode app;
  app.run();

  return 0;
}  // namespace track2d