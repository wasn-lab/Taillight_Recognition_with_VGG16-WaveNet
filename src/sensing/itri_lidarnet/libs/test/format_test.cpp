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

constexpr int g_num_perf_loops = 100;

TEST(format_test, init_test_env)
{
  auto pc2_msg_ptr = get_msg_ptr();
  EXPECT_TRUE(pc2_msg_ptr.get() != nullptr);
  EXPECT_EQ(pc2_msg_ptr->width, 1024);
  EXPECT_EQ(pc2_msg_ptr->height, 64);
  EXPECT_EQ(pc2_msg_ptr->fields.size(), 9);
  EXPECT_TRUE(pc2_msg_ptr->is_dense);

  auto cloud_ptr = get_cloud_ptr();
  EXPECT_TRUE(cloud_ptr.get() != nullptr);
  EXPECT_EQ(cloud_ptr->width, 1024);
  EXPECT_EQ(cloud_ptr->height, 64);
  EXPECT_EQ(cloud_ptr->points.size(), 1024 * 64);
  EXPECT_TRUE(cloud_ptr->is_dense);
}

TEST(format_test, test_SensorMsgs_to_XYZIR)
{
  auto msg_ptr = get_msg_ptr();
  auto cloud_ptr = get_cloud_ptr();
  auto res = SensorMsgs_to_XYZIR(*msg_ptr, lidar::Hardware::Ouster);
  EXPECT_EQ(res.points.size(), cloud_ptr->points.size());
  for (size_t idx = 0; idx < res.points.size(); idx++)
  {
    EXPECT_EQ(res[idx].x, cloud_ptr->points[idx].x);
    EXPECT_EQ(res[idx].y, cloud_ptr->points[idx].y);
    EXPECT_EQ(res[idx].z, cloud_ptr->points[idx].z);
    EXPECT_EQ(res[idx].intensity, cloud_ptr->points[idx].intensity);
    EXPECT_EQ(res[idx].ring, cloud_ptr->points[idx].ring);
  }
}

TEST(format_test, perf_SensorMsgs_to_XYZIR)
{
  auto msg_ptr = get_msg_ptr();
  for (int i = 0; i < g_num_perf_loops; i++)
  {
    auto ret = SensorMsgs_to_XYZIR(*msg_ptr, lidar::Hardware::Ouster);
  }
}
