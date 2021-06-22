/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include "pc_transform_utils.h"

namespace pc_transform
{
pcl::PointCloud<pcl::PointXYZI>::Ptr pc2_msg_to_xyzi(const sensor_msgs::PointCloud2ConstPtr& msg_ptr)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr target_cloud{ new pcl::PointCloud<pcl::PointXYZI> };
  // Get the field structure of this point cloud
  int point_bytes = msg_ptr->point_step;
  int offset_x = 0;
  int offset_y = 0;
  int offset_z = 0;
  int offset_intensity = 0;
  for (const auto& field : msg_ptr->fields)
  {
    if (field.name == "x")
    {
      offset_x = field.offset;
    }
    else if (field.name == "y")
    {
      offset_y = field.offset;
    }
    else if (field.name == "z")
    {
      offset_z = field.offset;
    }
    else if (field.name == "intensity")
    {
      offset_intensity = field.offset;
    }
  }

  // populate point cloud object
  target_cloud->points.resize(msg_ptr->width * msg_ptr->height);
  for (size_t p = 0, bound = msg_ptr->width * msg_ptr->height, point_offset = 0; p < bound;
       ++p, point_offset += point_bytes)
  {
    const auto base_addr = &msg_ptr->data[0] + point_offset;
    target_cloud->points[p].x = *(float*)(base_addr + offset_x);
    target_cloud->points[p].y = *(float*)(base_addr + offset_y);
    target_cloud->points[p].z = *(float*)(base_addr + offset_z);
    target_cloud->points[p].intensity = *(float*)(base_addr + offset_intensity);
  }
  pcl_conversions::toPCL(msg_ptr->header, target_cloud->header);
  return target_cloud;
}

uint32_t checksum_of(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr)
{
  const uint32_t p = 16777619;
  auto hash = static_cast<uint32_t>(2166136261);
  for (int i = cloud_ptr->points.size() - 1; i >= 0; i--)
  {
    hash = (hash ^ (static_cast<uint32_t>(cloud_ptr->points[i].x))) * p;
    hash = (hash ^ (static_cast<uint32_t>(cloud_ptr->points[i].y))) * p;
    hash = (hash ^ (static_cast<uint32_t>(cloud_ptr->points[i].z))) * p;
    hash = (hash ^ (static_cast<uint32_t>(cloud_ptr->points[i].intensity))) * p;
    hash += hash << 13u;
    hash ^= hash >> 7u;
    hash += hash << 3u;
    hash ^= hash >> 17u;
    hash += hash << 5u;
  }
  return hash;
}

uint32_t checksum_of(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  const uint32_t p = 16777619;
  auto hash = static_cast<uint32_t>(2166136261);
  hash = (hash ^ (static_cast<uint32_t>(msg->header.seq))) * p;
  hash = (hash ^ (static_cast<uint32_t>(msg->header.stamp.sec))) * p;
  hash = (hash ^ (static_cast<uint32_t>(msg->header.stamp.nsec))) * p;
  for (const unsigned char ch : msg->data)
  {
    hash = (hash ^ (static_cast<uint32_t>(ch))) * p;
    hash += hash << 13u;
    hash ^= hash >> 7u;
    hash += hash << 3u;
    hash ^= hash >> 17u;
    hash += hash << 5u;
  }
  return hash;
}

};  // namespace pc_transform
