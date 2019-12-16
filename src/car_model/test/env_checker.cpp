#include <iostream>
#include <gtest/gtest.h>
#include <opencv2/core/version.hpp>
#include <pcl/pcl_config.h>

TEST(VersionChecker, test_ver)
{
  std::cout << "OpenCV Version: " << CV_VERSION << "\n";
  std::cout << "PCL Version: " << PCL_VERSION_PRETTY << "\n";
}

