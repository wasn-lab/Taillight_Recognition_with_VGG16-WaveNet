#include <gtest/gtest.h>
#include <glog/logging.h>
//#include <gflags/gflags.h>
//#include <gflags/gflags_gflags.h>

//#include <ros/ros.h>

// Run all the tests that were declared with TEST()
int main(int argc, char** argv)
{
//  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
