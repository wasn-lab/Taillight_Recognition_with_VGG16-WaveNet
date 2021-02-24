/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <cstdlib>
#include <memory>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "pcl/point_types.h"
#include "gen_msg.h"
#include "pointcloud_format_conversion.h"
#include "lidar_hardware.h"

constexpr int num_perf_loops = 100;

TEST(format_test, test_gen_rand_lidar_msg)
{
  auto ptr = gen_rand_lidar_msg();
}

TEST(format_test, perf_SensorMsgs_to_XYZIR)
{
  auto ptr = gen_rand_lidar_msg();
  for(int i=0; i<num_perf_loops; i++){
    auto ret = SensorMsgs_to_XYZIR(*ptr, lidar::Hardware::Ouster);
  }
}

