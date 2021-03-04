/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#pragma once

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

namespace pc2_compressor
{
class ITRIPCDWriter : public pcl::PCDWriter
{
public:
  ITRIPCDWriter() = default;
  ~ITRIPCDWriter() = default;

  int writeBinaryCompressed(std::ostream& os, const pcl::PCLPointCloud2& cloud, const int32_t fmt,
                            const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                            const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity());
};
};  // namespace pc2_compressor
