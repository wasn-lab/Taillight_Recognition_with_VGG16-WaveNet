#include "glog/logging.h"
#include "pc2_compressor.h"
#include <pcl/io/pcd_io.h>
#include <pcl_ros/transforms.h>

#define NO_UNUSED_VAR_CHECK(x) ((void)(x))

namespace pc2_compressor
{
std::string compress(const sensor_msgs::PointCloud2ConstPtr& in_pcd_message)
{
  pcl::PCDWriter writer;
  pcl::PCLPointCloud2 pc2;
  pcl_conversions::toPCL(*in_pcd_message, pc2);

  std::ostringstream oss;
  int res = writer.writeBinaryCompressed(oss, pc2);
  assert(res == 0);
  NO_UNUSED_VAR_CHECK(res);
  return oss.str();
}

sensor_msgs::PointCloud2Ptr decompress(const std::string& pcd_str)
{
  int pcd_version = -1;
  int data_type = -1;
  unsigned int data_idx = 0;
  Eigen::Vector4f origin;
  Eigen::Quaternionf orientation;

  std::istringstream iss(pcd_str, std::ios::binary);
  pcl::PCDReader reader;
  pcl::PCLPointCloud2 pcl_pc2;
  int res = reader.readHeader(iss, pcl_pc2, origin, orientation, pcd_version, data_type, data_idx);
  assert(res == 0);
  assert(data_type == 2);  // Expect data_type is compressed.

  const unsigned char* data = reinterpret_cast<const unsigned char*>(pcd_str.data());
  res = reader.readBodyBinary(data, pcl_pc2, pcd_version, /* compressed = */ true, data_idx);
  assert(res == 0);

  sensor_msgs::PointCloud2Ptr ros_pc2_ptr(new sensor_msgs::PointCloud2);
  pcl_conversions::moveFromPCL(pcl_pc2, *ros_pc2_ptr);
  return ros_pc2_ptr;
}

};  // namespace pc2_compressor