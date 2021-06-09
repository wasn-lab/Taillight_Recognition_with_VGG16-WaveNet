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
#include "pc2_args_parser.h"
#include "point_os1.h"
#include "point_xyzir.h"

constexpr int num_perf_loops = 100;

static bool check_if_lossless(const sensor_msgs::PointCloud2ConstPtr& decmpr_msg_ptr)
{
  EXPECT_EQ(decmpr_msg_ptr->width, static_cast<size_t>(1024));
  EXPECT_EQ(decmpr_msg_ptr->height, static_cast<size_t>(64));
  EXPECT_EQ(decmpr_msg_ptr->is_dense, true);

  pcl::PointCloud<ouster_ros::OS1::PointOS1> decmpr_cloud;
  pcl::PCLPointCloud2 decmpr_pcl_pc2;

  pcl_conversions::toPCL(*decmpr_msg_ptr, decmpr_pcl_pc2);

  pcl::fromPCLPointCloud2(decmpr_pcl_pc2, decmpr_cloud);
  EXPECT_EQ(decmpr_cloud.points.size(), g_cloud.points.size());

  for (size_t i = 0; i < decmpr_cloud.points.size(); ++i)
  {
    EXPECT_EQ(decmpr_cloud.points[i].x, g_cloud.points[i].x);
    EXPECT_EQ(decmpr_cloud.points[i].y, g_cloud.points[i].y);
    EXPECT_EQ(decmpr_cloud.points[i].z, g_cloud.points[i].z);
    EXPECT_EQ(decmpr_cloud.points[i].intensity, g_cloud.points[i].intensity);
  }
  return true;
}

TEST(PC2CompressorTest, test_gen_rand_cloud)
{
  gen_rand_cloud();
  EXPECT_EQ(g_cloud.points.size(), static_cast<size_t>(65536));
  EXPECT_EQ(sizeof(g_cloud.points[0]), static_cast<size_t>(48));
  EXPECT_EQ(sizeof(g_cloud.points[0].x), static_cast<size_t>(4));
  EXPECT_EQ(sizeof(g_cloud.points[0].y), static_cast<size_t>(4));
  EXPECT_EQ(sizeof(g_cloud.points[0].z), static_cast<size_t>(4));
  EXPECT_EQ(sizeof(g_cloud.points[0].intensity), static_cast<size_t>(4));
  EXPECT_EQ(g_org_ros_pc2_ptr->fields.size(), 9);
  EXPECT_EQ(g_org_ros_pc2_ptr->fields[0].name, "x");
  EXPECT_EQ(g_org_ros_pc2_ptr->fields[1].name, "y");
  EXPECT_EQ(g_org_ros_pc2_ptr->fields[2].name, "z");
  EXPECT_EQ(g_org_ros_pc2_ptr->fields[3].name, "intensity");
}

TEST(PC2CompressorTest, test_default_cmpr_decmpr)
{
  gen_rand_cloud();

  auto cmpr_msg_ptr = pc2_compressor::compress_msg(g_org_ros_pc2_ptr);
  EXPECT_TRUE(cmpr_msg_ptr->data.size() > 0);

  EXPECT_EQ(cmpr_msg_ptr->header, g_org_ros_pc2_ptr->header);
  EXPECT_EQ(cmpr_msg_ptr->header.seq, 31415);

  auto decmpr_msg_ptr = pc2_compressor::decompress_msg(cmpr_msg_ptr);
  EXPECT_TRUE(check_if_lossless(decmpr_msg_ptr));
  EXPECT_EQ(decmpr_msg_ptr->header, g_org_ros_pc2_ptr->header);
  EXPECT_EQ(decmpr_msg_ptr->header.seq, 31415);
  EXPECT_TRUE(pc2_compressor::is_equal_pc2(decmpr_msg_ptr, g_org_ros_pc2_ptr));
}

TEST(PC2CompressorTest, test_lzf_cmpr_decmpr)
{
  gen_rand_cloud();

  pc2_compressor::set_verbose(true);
  auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::lzf);
  EXPECT_EQ(cmpr_msg->compression_format, pc2_compressor::compression_format::lzf);
  auto decmpr_msg_ptr = pc2_compressor::decompress_msg(cmpr_msg);
  pc2_compressor::set_verbose(false);
  EXPECT_TRUE(check_if_lossless(decmpr_msg_ptr));
  EXPECT_TRUE(pc2_compressor::is_equal_pc2(decmpr_msg_ptr, g_org_ros_pc2_ptr));
}

TEST(PC2CompressorTest, test_snappy_cmpr_decmpr)
{
  gen_rand_cloud();
  pc2_compressor::set_verbose(true);
  auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::snappy);
  EXPECT_EQ(cmpr_msg->compression_format, pc2_compressor::compression_format::snappy);
  auto decmpr_msg_ptr = pc2_compressor::decompress_msg(cmpr_msg);
  EXPECT_TRUE(check_if_lossless(decmpr_msg_ptr));
  EXPECT_TRUE(pc2_compressor::is_equal_pc2(decmpr_msg_ptr, g_org_ros_pc2_ptr));
}

TEST(PC2CompressorTest, test_none_cmpr_decmpr)
{
  gen_rand_cloud();
  auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::none);
  auto decmpr_msg_ptr = pc2_compressor::decompress_msg(cmpr_msg);
  EXPECT_TRUE(check_if_lossless(decmpr_msg_ptr));
  EXPECT_TRUE(pc2_compressor::is_equal_pc2(decmpr_msg_ptr, g_org_ros_pc2_ptr));
}

TEST(PC2CompressorTest, test_zlib_cmpr_decmpr)
{
  gen_rand_cloud();
  pc2_compressor::set_verbose(true);
  auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::zlib);
  EXPECT_EQ(cmpr_msg->compression_format, pc2_compressor::compression_format::zlib);
  auto decmpr_msg_ptr = pc2_compressor::decompress_msg(cmpr_msg);
  pc2_compressor::set_verbose(false);
  EXPECT_TRUE(check_if_lossless(decmpr_msg_ptr));
  EXPECT_TRUE(pc2_compressor::is_equal_pc2(decmpr_msg_ptr, g_org_ros_pc2_ptr));
}

TEST(PC2CompressorTest, test_ouster64_to_xyzir)
{
  gen_rand_cloud();
  auto msg = pc2_compressor::ouster64_to_xyzir(g_org_ros_pc2_ptr);
  EXPECT_EQ(msg->fields.size(), 5U);
  EXPECT_EQ(msg->fields[0].name, "x");
  EXPECT_EQ(msg->fields[1].name, "y");
  EXPECT_EQ(msg->fields[2].name, "z");
  EXPECT_EQ(msg->fields[3].name, "intensity");
  EXPECT_EQ(msg->fields[4].name, "ring");
  EXPECT_EQ(msg->header, g_org_ros_pc2_ptr->header);
  EXPECT_EQ(msg->width, 1024U);
  EXPECT_EQ(msg->height, 64U);
  EXPECT_EQ(msg->is_bigendian, g_org_ros_pc2_ptr->is_bigendian);
  EXPECT_EQ(msg->is_dense, g_org_ros_pc2_ptr->is_dense);
  EXPECT_EQ(msg->header, g_org_ros_pc2_ptr->header);

  int32_t org_size = pc2_compressor::size_of_msg(g_org_ros_pc2_ptr);
  int32_t result_size = pc2_compressor::size_of_msg(msg);
  double ratio = double(result_size) / org_size;

  LOG(INFO) << "Filtered size: " << result_size << ", org size: " << org_size << ", ratio: " << ratio;

  // Check x, y, z, intensity
  pcl::PCLPointCloud2 before_pc2;
  pcl::PointCloud<ouster_ros::OS1::PointXYZIR>::Ptr before_cloud(new pcl::PointCloud<ouster_ros::OS1::PointXYZIR>);
  pcl_conversions::toPCL(*g_org_ros_pc2_ptr, before_pc2);
  pcl::fromPCLPointCloud2(before_pc2, *before_cloud);

  pcl::PCLPointCloud2 after_pc2;
  pcl_conversions::toPCL(*msg, after_pc2);
  pcl::PointCloud<ouster_ros::OS1::PointXYZIR>::Ptr after_cloud(new pcl::PointCloud<ouster_ros::OS1::PointXYZIR>);
  pcl::fromPCLPointCloud2(after_pc2, *after_cloud);
  EXPECT_EQ(after_cloud->size(), 65536U);
  for(int i=0; i<after_cloud->size(); i++)
  {
    EXPECT_EQ(before_cloud->points[i].x, after_cloud->points[i].x);
    EXPECT_EQ(before_cloud->points[i].y, after_cloud->points[i].y);
    EXPECT_EQ(before_cloud->points[i].z, after_cloud->points[i].z);
    EXPECT_EQ(before_cloud->points[i].intensity, after_cloud->points[i].intensity);
    EXPECT_EQ(before_cloud->points[i].ring, after_cloud->points[i].ring);
    EXPECT_EQ(before_cloud->points[i].ring, i % 64);
  }
}


TEST(PC2CompressorTest, test_sizeof_msg)
{
  gen_rand_cloud();
  EXPECT_EQ(pc2_compressor::size_of_msg(g_org_ros_pc2_ptr), 1900730);
}

TEST(PC2CompressorTest, test_lzf_cmpr_perf)
{
  gen_rand_cloud();
  for (int i = 0; i < num_perf_loops; i++)
  {
    auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::lzf);
  }
}

TEST(PC2CompressorTest, test_lzf_decmpr_perf)
{
  gen_rand_cloud();
  auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::lzf);
  for (int i = 0; i < num_perf_loops; i++)
  {
    auto decmpr_msg_ptr = pc2_compressor::decompress_msg(cmpr_msg);
  }
}

TEST(PC2CompressorTest, test_snappy_cmpr_perf)
{
  gen_rand_cloud();
  for (int i = 0; i < num_perf_loops; i++)
  {
    auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::snappy);
  }
}

TEST(PC2CompressorTest, test_snappy_decmpr_perf)
{
  gen_rand_cloud();
  auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::snappy);
  for (int i = 0; i < num_perf_loops; i++)
  {
    auto decmpr_msg_ptr = pc2_compressor::decompress_msg(cmpr_msg);
  }
}

TEST(PC2CompressorTest, test_zlib_cmpr_perf)
{
  gen_rand_cloud();
  for (int i = 0; i < num_perf_loops; i++)
  {
    auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::zlib);
  }
}

TEST(PC2CompressorTest, test_zlib_decmpr_perf)
{
  gen_rand_cloud();
  auto cmpr_msg = pc2_compressor::compress_msg(g_org_ros_pc2_ptr, pc2_compressor::compression_format::zlib);
  for (int i = 0; i < num_perf_loops; i++)
  {
    auto decmpr_msg_ptr = pc2_compressor::decompress_msg(cmpr_msg);
  }
}

TEST(PC2CompressorTest, test_ouster64_to_xyzir_perf)
{
  gen_rand_cloud();
  for (int i = 0; i < num_perf_loops; i++)
  {
    auto msg = pc2_compressor::ouster64_to_xyzir(g_org_ros_pc2_ptr);
  }
}
