#include "daf_node.h"
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#if USE_GLOG == 1
#include <glog/logging.h>
#endif

int main(int argc, char** argv)
{
  ros::init(argc, argv, "drivable_area_filter", ros::init_options::NoSigintHandler);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
#if USE_GLOG == 1
  google::InstallFailureSignalHandler();
#endif
  daf::DAFNode app;
  app.run();
}  // namespace daf