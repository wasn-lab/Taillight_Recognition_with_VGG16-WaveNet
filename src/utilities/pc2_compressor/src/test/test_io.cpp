/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <gtest/gtest.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_traits.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/console/print.h>
#include <pcl/io/auto_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/ascii_io.h>
#include <pcl/io/obj_io.h>
#include <fstream>
#include <locale>
#include <stdexcept>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/serialization.h>
#include <glog/logging.h>

using namespace pcl;
using namespace pcl::io;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(PCL, IO)
{
  pcl::PCLPointCloud2 cloud_blob;
  PointCloud<PointXYZI> cloud;

  cloud.width = 640;
  cloud.height = 480;
  cloud.points.resize(cloud.width * cloud.height);
  cloud.is_dense = true;

  srand(static_cast<unsigned int>(time(nullptr)));
  std::size_t nr_p = cloud.points.size();
  // Randomly create a new point cloud
  for (std::size_t i = 0; i < nr_p; ++i)
  {
    cloud[i].x = static_cast<float>(1024 * rand() / (RAND_MAX + 1.0));
    cloud[i].y = static_cast<float>(1024 * rand() / (RAND_MAX + 1.0));
    cloud[i].z = static_cast<float>(1024 * rand() / (RAND_MAX + 1.0));
    cloud.points[i].intensity = static_cast<float>(i);
  }
  // Convert from data type to blob
  toPCLPointCloud2(cloud, cloud_blob);
  EXPECT_EQ(cloud_blob.fields.size(), 4);

  pcl::PCLPointCloud2 pc2;
  int res = pcl::io::savePCDFile("test_pcl_io.pcd", cloud_blob, Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(),
                                 true);
  EXPECT_EQ(res, 0);
  res = loadPCDFile("test_pcl_io.pcd", pc2);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(pc2.fields.size(), 4);
}

TEST(PCL, IO2)
{
  pcl::PCLPointCloud2 cloud_blob;
  PointCloud<PointXYZI> cloud;

  cloud.width = 640;
  cloud.height = 480;
  cloud.points.resize(cloud.width * cloud.height);
  cloud.is_dense = true;

  srand(static_cast<unsigned int>(time(nullptr)));
  std::size_t nr_p = cloud.points.size();
  // Randomly create a new point cloud
  for (std::size_t i = 0; i < nr_p; ++i)
  {
    cloud[i].x = static_cast<float>(1024 * rand() / (RAND_MAX + 1.0));
    cloud[i].y = static_cast<float>(1024 * rand() / (RAND_MAX + 1.0));
    cloud[i].z = static_cast<float>(1024 * rand() / (RAND_MAX + 1.0));
    cloud.points[i].intensity = static_cast<float>(i);
  }
  // Convert from data type to blob
  toPCLPointCloud2(cloud, cloud_blob);
  EXPECT_EQ(cloud_blob.fields.size(), 4);

  pcl::PCLPointCloud2 pc2;
  int res = pcl::io::savePCDFile("test_pcl_io.pcd", cloud_blob, Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(),
                                 true);
  EXPECT_EQ(res, 0);
  res = loadPCDFile("test_pcl_io.pcd", pc2);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(pc2.fields.size(), 4);
  ros::serialization::Serializer<pcl::PCLPointCloud2> serializer;
  uint8_t* data = new uint8_t[1024 * 1024 * 16];
  ros::serialization::OStream stream(data, 1024 * 1024 * 16);
  LOG(INFO) << "before stream len: " << stream.getLength();
  serializer.write(stream, cloud_blob);
  LOG(INFO) << "after stream len: " << stream.getLength();
  delete[] data;
}
