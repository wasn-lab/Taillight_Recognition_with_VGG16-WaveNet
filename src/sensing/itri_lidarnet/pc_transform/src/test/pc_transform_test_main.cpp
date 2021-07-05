
#include <ros/ros.h>
#include "gtest/gtest.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "pc_transform_test_node", ros::init_options::AnonymousName);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
