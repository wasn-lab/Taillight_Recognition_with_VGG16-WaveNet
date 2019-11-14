#include <iostream>
#include <gtest/gtest.h>
#include <opencv2/core/version.hpp>
#include <pcl/pcl_config.h>

TEST(VersionChecker, test_ver)
{
  std::cout << "OpenCV Version: " << CV_VERSION << "\n";
  std::cout << "PCL Version: " << PCL_VERSION_PRETTY << "\n";
  EXPECT_EQ(CV_VERSION_MAJOR, 3);
  EXPECT_EQ(CV_VERSION_MINOR, 4);
  EXPECT_EQ(CV_VERSION_REVISION, 5);
  EXPECT_EQ(PCL_MAJOR_VERSION, 1);
  EXPECT_EQ(PCL_MINOR_VERSION, 9);
}

