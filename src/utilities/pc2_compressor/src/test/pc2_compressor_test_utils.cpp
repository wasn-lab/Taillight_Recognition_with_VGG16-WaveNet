/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <cstdlib>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <pcl/point_types.h>
#include "pc2_compressor.h"
#include "pc2_compressor_def.h"
#include "point_os1.h"

pcl::PointCloud<ouster_ros::OS1::PointOS1> g_cloud;
pcl::PCLPointCloud2 g_pcl_pc2;
sensor_msgs::PointCloud2Ptr g_org_ros_pc2_ptr;

void gen_rand_cloud()
{
  if (!g_cloud.empty())
  {
    return;
  }
  LOG(INFO) << "init global point cloud";
  const std::string ouster64 = std::string{PC2_COMPRESSOR_TEST_DIR} + "/ouster64.pcd";
  pcl::io::loadPCDFile(ouster64, g_pcl_pc2);
  pcl::fromPCLPointCloud2(g_pcl_pc2, g_cloud);

  g_org_ros_pc2_ptr.reset(new sensor_msgs::PointCloud2);
  pcl_conversions::fromPCL(g_pcl_pc2, *g_org_ros_pc2_ptr);

  g_org_ros_pc2_ptr->header.seq = 31415;
  g_org_ros_pc2_ptr->header.stamp.sec = 1608000995;
  g_org_ros_pc2_ptr->header.stamp.nsec = 10007;
  g_org_ros_pc2_ptr->header.frame_id = "test";
}
