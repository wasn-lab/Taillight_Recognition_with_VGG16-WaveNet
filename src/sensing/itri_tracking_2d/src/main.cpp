#include "track2d_node.h"
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#if USE_GLOG
#include <glog/logging.h>
#endif

int main(int argc, char** argv)
{
  ros::init(argc, argv, "itri_tracking_2d", ros::init_options::NoSigintHandler);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
#if USE_GLOG
  google::InstallFailureSignalHandler();
#endif
  track2d::Track2DNode app;
  app.run();

  return 0;
}  // namespace track2d