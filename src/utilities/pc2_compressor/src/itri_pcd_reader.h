#pragma once

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

namespace pc2_compressor
{
class ITRIPCDReader : public pcl::PCDReader
{
public:
  ITRIPCDReader() = default;
  ~ITRIPCDReader() = default;

  int readBodyCompressed(const unsigned char* data, pcl::PCLPointCloud2& cloud, const int32_t fmt,
                         const uint32_t data_idx);
};
};  // namespace pc2_compressor
