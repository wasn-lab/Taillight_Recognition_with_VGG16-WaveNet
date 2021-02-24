/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <glog/logging.h>
#include <pcl_conversions/pcl_conversions.h>
#include "point_os1.h"
#include "gen_msg.h"

sensor_msgs::PointCloud2Ptr g_pc2_msg_ptr;

sensor_msgs::PointCloud2Ptr __gen_rand_lidar_msg()
{
  pcl::PointCloud<ouster_ros::OS1::PointOS1> cloud;
  cloud.width = 1024;
  cloud.height = 64;
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
    cloud[i].intensity = static_cast<float>(1 + (rand() & 0x2ff));
    cloud[i].t = rand() & 0x2ff;
    cloud[i].reflectivity = rand() & 0x2ff;
    cloud[i].ring = i % 64;
    cloud[i].noise = static_cast<uint16_t>(rand() % 65536);
    cloud[i].range = static_cast<uint32_t>(rand());
  }

  pcl::PCLPointCloud2 cloud_blob;
  pcl::toPCLPointCloud2(cloud, cloud_blob);

  g_pc2_msg_ptr.reset(new sensor_msgs::PointCloud2);
  pcl_conversions::fromPCL(cloud_blob, *g_pc2_msg_ptr);

  return g_pc2_msg_ptr;
}

sensor_msgs::PointCloud2ConstPtr gen_rand_lidar_msg()
{
  if (!g_pc2_msg_ptr.get()) {
    g_pc2_msg_ptr = __gen_rand_lidar_msg();
    CHECK(g_pc2_msg_ptr.get() != nullptr);
    CHECK(g_pc2_msg_ptr->width == 1024);
    CHECK(g_pc2_msg_ptr->height == 64);
    CHECK(g_pc2_msg_ptr->fields.size() == 9);
    CHECK(g_pc2_msg_ptr->is_dense);
  }
  return g_pc2_msg_ptr;
}
