/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include "glog/logging.h"
#include "snappy.h"
#include "pc2_compressor.h"
#include "pc2_args_parser.h"
#include "itri_pcd_reader.h"
#include "itri_pcd_writer.h"

#define NO_UNUSED_VAR_CHECK(x) ((void)(x))

namespace pc2_compressor
{
static msgs::CompressedPointCloud2ConstPtr __compress(const sensor_msgs::PointCloud2ConstPtr& in_msg, const int32_t fmt)
{
  ITRIPCDWriter writer;
  pcl::PCLPointCloud2 pc2;
  pcl_conversions::toPCL(*in_msg, pc2);

  std::ostringstream oss;
  int res = writer.writeBinaryCompressed(oss, pc2, fmt);
  assert(res == 0);
  NO_UNUSED_VAR_CHECK(res);

  msgs::CompressedPointCloud2Ptr cmpr_msg{ new msgs::CompressedPointCloud2 };
  cmpr_msg->data = oss.str();
  cmpr_msg->header = in_msg->header;
  cmpr_msg->compression_format = fmt;

  if (is_verbose())
  {
    auto org_size = size_of_msg(in_msg);
    auto cmpr_size = size_of_msg(cmpr_msg);
    LOG(INFO) << "Compression ratio: " << double(cmpr_size) / org_size << " (" << cmpr_size << "/" << org_size << ")"
              << ", " << describe(in_msg);
  }

  return cmpr_msg;
}

static sensor_msgs::PointCloud2ConstPtr __decompress(const msgs::CompressedPointCloud2ConstPtr& cmpr_msg)
{
  int pcd_version = -1;
  int data_type = -1;
  unsigned int data_idx = 0;
  Eigen::Vector4f origin;
  Eigen::Quaternionf orientation;

  std::istringstream iss(cmpr_msg->data, std::ios::binary);
  ITRIPCDReader reader;
  pcl::PCLPointCloud2 pcl_pc2;
  int res = reader.readHeader(iss, pcl_pc2, origin, orientation, pcd_version, data_type, data_idx);
  NO_UNUSED_VAR_CHECK(res);
  assert(res == 0);
  assert(data_type == 2);  // Expect data_type is compressed.

  const auto data = reinterpret_cast<const unsigned char*>(cmpr_msg->data.data());
  const int32_t fmt = cmpr_msg->compression_format;
  res = reader.readBodyCompressed(data, pcl_pc2, fmt, data_idx);
  NO_UNUSED_VAR_CHECK(res);
  assert(res == 0);

  sensor_msgs::PointCloud2Ptr decmpr_msg(new sensor_msgs::PointCloud2);
  pcl_conversions::moveFromPCL(pcl_pc2, *decmpr_msg);
  decmpr_msg->header = cmpr_msg->header;
  return decmpr_msg;
}

msgs::CompressedPointCloud2ConstPtr compress_msg(const sensor_msgs::PointCloud2ConstPtr& in_msg, const int32_t fmt)
{
  CHECK(fmt >= 0 && fmt < compression_format::nums) << "unsupported format " << fmt;
  return __compress(in_msg, fmt);
}

sensor_msgs::PointCloud2ConstPtr decompress_msg(const msgs::CompressedPointCloud2ConstPtr& cmpr_msg)
{
  return __decompress(cmpr_msg);
}

bool is_equal_pc2(const sensor_msgs::PointCloud2ConstPtr& a, const sensor_msgs::PointCloud2ConstPtr& b)
{
  if (a->header != b->header)
  {
    LOG(INFO) << "inconstent header";
    return false;
  }
  sensor_msgs::PointCloud pc_a, pc_b;

  sensor_msgs::convertPointCloud2ToPointCloud(*a, pc_a);
  sensor_msgs::convertPointCloud2ToPointCloud(*b, pc_b);

  if (pc_a.points.size() != pc_b.points.size())
  {
    LOG(INFO) << "Inconsitent points size: pc_a.points.size()=" << pc_a.points.size()
              << " pc_b.points.size()=" << pc_b.points.size();
    return false;
  }

  // pc_a.points is of type geometry_msgs/Point32[], each of which is 3d points
  for (int i = 0, np = pc_a.points.size(); i < np; i++)
  {
    if (pc_a.points[i].x != pc_b.points[i].x || pc_a.points[i].y != pc_b.points[i].y ||
        pc_a.points[i].z != pc_b.points[i].z)
    {
      LOG(INFO) << "inconsitent point[ " << i << "]: " << pc_a.points[i] << " v.s. " << pc_b.points[i];
      return false;
    }
  }

  if (pc_a.channels.size() != pc_b.channels.size())
  {
    LOG(INFO) << "Inconsitent channels size: pc_a channel=" << pc_a.channels.size()
              << " pc_b channels=" << pc_b.channels.size();
    LOG(INFO) << "pc_a channels:";
    for (auto& channel : pc_a.channels)
    {
      LOG(INFO) << channel.name;
    }

    LOG(INFO) << "pc_b channels:";
    for (auto& channel : pc_b.channels)
    {
      LOG(INFO) << channel.name;
    }
    return false;
  }

  for (auto& channel_a : pc_a.channels)
  {
    for (auto& channel_b : pc_b.channels)
    {
      if (channel_a.name != channel_b.name)
      {
        continue;
      }
      for (int k = 0, nvalues = channel_a.values.size(); k < nvalues; k++)
      {
        if (channel_a.values[k] != channel_b.values[k])
        {
          LOG(INFO) << "channel " << channel_a.name << " value differ at " << k << ":" << channel_a.values[k]
                    << " v.s. " << channel_b.values[k];
          return false;
        }
      }
    }
  }
  return true;
}

uint64_t size_of_msg(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  uint64_t res = sizeof(msg->header);
  res += sizeof(msg->height);
  res += sizeof(msg->width);
  res += sizeof(msg->is_bigendian);
  res += sizeof(msg->point_step);
  res += sizeof(msg->row_step);
  res += sizeof(msg->is_dense);
  res += sizeof(uint8_t) * msg->data.size();
  for (auto& field : msg->fields)
  {
    res += field.name.size();
    res += sizeof(field.offset);
    res += sizeof(field.datatype);
    res += sizeof(field.count);
  }
  return res;
}

uint64_t size_of_msg(const msgs::CompressedPointCloud2ConstPtr& msg)
{
  return sizeof(msg->header) + msg->data.size();
}

std::string describe(const sensor_msgs::PointCloud2ConstPtr& in_msg)
{
  std::string field_names;
  for (const auto& field : in_msg->fields)
  {
    field_names += field.name + "(datatype: " + std::to_string(field.datatype) + ") ";
  }
  if (!field_names.empty())
  {
    field_names.pop_back();
  }
  auto num_points = in_msg->height * in_msg->width;
  auto is_bigendian = in_msg->is_bigendian ? "true" : "false";
  auto is_dense = in_msg->is_dense ? "true" : "false";
  return "#points: " + std::to_string(num_points) + ", is_bigendian: " + is_bigendian +
         ", point_step: " + std::to_string(in_msg->point_step) + ", row_step: " + std::to_string(in_msg->row_step) +
         ", is_dense: " + is_dense + ", point cloud fields: " + field_names;
}

};  // namespace pc2_compressor
