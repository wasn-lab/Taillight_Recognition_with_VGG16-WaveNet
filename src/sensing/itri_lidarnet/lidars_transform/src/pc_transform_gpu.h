#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>

namespace pc_transform
{
template <typename PointT>
class PCTransformGPU
{
public:
  PCTransformGPU();
  ~PCTransformGPU();
  int set_transform_matrix(const float tx, const float ty, const float tz, const float rx, const float ry,
                           const float rz);

  bool transform(pcl::PointCloud<PointT>& cloud);
  bool transform(sensor_msgs::PointCloud2Ptr msg);

private:
  Eigen::Affine3f transform_matrix_;
  float tm_elements_[16];  // 4x4 matrix, column-major
  float* tm_elements_gpu_ = nullptr;
  PointT* cloud_gpu_ = nullptr;
  size_t cloud_gpu_size_ = 0;
  int num_cuda_threads_;

  void free_cloud_gpu_if_necessary();
  PCTransformGPU(PCTransformGPU& other) = delete;
  PCTransformGPU operator=(const PCTransformGPU& other) = delete;
};
};  // namespace pc_transform
