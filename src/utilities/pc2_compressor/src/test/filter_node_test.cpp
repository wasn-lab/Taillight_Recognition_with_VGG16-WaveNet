/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <cstdlib>
#include <memory>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "pc2_compressor.h"
#include "pcl/point_types.h"
#include "pc2_compressor_test_utils.h"
#include "pc2_compression_format.h"
#include "filter_node.h"

TEST(PC2FilterTest, test_warmup)
{
  gen_rand_cloud();
}

TEST(PC2FilterTest, test_filter_ouster64_pc2)
{
  auto msg = pc2_compressor::filter_ouster64_pc2(g_org_ros_pc2_ptr);
  EXPECT_EQ(msg->fields.size(), 5);
  EXPECT_EQ(msg->fields[0].name, "x");
  EXPECT_EQ(msg->fields[1].name, "y");
  EXPECT_EQ(msg->fields[2].name, "z");
  EXPECT_EQ(msg->fields[3].name, "intensity");
  EXPECT_EQ(msg->fields[4].name, "ring");

  int32_t org_size = pc2_compressor::size_of_msg(g_org_ros_pc2_ptr);
  int32_t result_size = pc2_compressor::size_of_msg(msg);
  double ratio = double(result_size) / org_size;

  LOG(INFO) << "Filtered size: " << result_size << ", org size: " << org_size << ", ratio: " << ratio;

}

