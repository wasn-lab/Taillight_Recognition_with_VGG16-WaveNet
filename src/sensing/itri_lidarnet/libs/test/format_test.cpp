/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <cstdlib>
#include <memory>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "pcl/point_types.h"
#include "point_os1.h"
#include "lidar_hardware.h"
#include "pointcloud_format_conversion.h"
#include "gen_msg.h"

constexpr int num_perf_loops = 100;

TEST(format_test, init_test_env)
{
  auto msg_ptr = get_msg_ptr();
}

TEST(format_test, perf_SensorMsgs_to_XYZIR)
{
  auto msg_ptr = get_msg_ptr();
  for(int i=0; i<num_perf_loops; i++){
    auto ret = SensorMsgs_to_XYZIR(*msg_ptr, lidar::Hardware::Ouster);
  }
}
