/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <pcl/console/print.h>

//#include <ros/ros.h>

// Run all the tests that were declared with TEST()
int main(int argc, char** argv)
{
  //  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  //  pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
  return RUN_ALL_TESTS();
}
