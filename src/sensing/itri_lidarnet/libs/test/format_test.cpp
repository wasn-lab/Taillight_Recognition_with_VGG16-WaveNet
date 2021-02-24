/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <cstdlib>
#include <memory>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "pcl/point_types.h"

constexpr int num_perf_loops = 100;

TEST(lidar_test, test_gen_rand_cloud)
{
  EXPECT_EQ(1+3, 4);
}

