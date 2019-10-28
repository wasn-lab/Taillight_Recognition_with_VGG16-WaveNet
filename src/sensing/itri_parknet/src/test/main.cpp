#include <gtest/gtest.h>
#include <ros/ros.h>

// Run all the tests that were declared with TEST()
int main(int argc, char** argv)
{
  ros::init(argc, argv, "parknet_node_test");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
