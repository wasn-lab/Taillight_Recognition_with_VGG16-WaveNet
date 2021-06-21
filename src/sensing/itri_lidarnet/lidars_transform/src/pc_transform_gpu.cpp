#include <glog/stl_logging.h>
#include <glog/logging.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include "pc_transform_gpu.h"

namespace pc_transform
{
template <typename PointT>
cudaError_t cudaTransformPoints(int threads, PointT* cloud_gpu_, int number_of_points, float* d_matrix);

template <typename PointT>
PCTransformGPU<PointT>::PCTransformGPU(): num_cuda_threads_(0)
{
  cudaError_t err = ::cudaSuccess;
  err = cudaSetDevice(0);
  CHECK(err == ::cudaSuccess) << "Initialize CUDA failed";

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  num_cuda_threads_ = prop.maxThreadsPerBlock;
  LOG(INFO) << "cuda device version: " << prop.major << "." << prop.minor << ", device name: " << prop.name
            << ", set num_cuda_threads_ to " << num_cuda_threads_;
  CHECK(num_cuda_threads_ > 0) << "Cuda threads is 0!";

  err = cudaMalloc((void**)&tm_elements_gpu_, 16 * sizeof(float));
  CHECK(err == cudaSuccess) << "Cannot allocate gpu memory for tm_elements_gpu_";
}

template <typename PointT>
PCTransformGPU<PointT>::~PCTransformGPU()
{
  if (tm_elements_gpu_ != nullptr)
  {
    auto ret = cudaFree(tm_elements_gpu_);
    CHECK(ret == ::cudaSuccess);
    tm_elements_gpu_ = nullptr;
  }
  free_cloud_gpu_if_necessary();
}

template <typename PointT>
void PCTransformGPU<PointT>::free_cloud_gpu_if_necessary()
{
  if (cloud_gpu_ != nullptr)
  {
    auto ret = cudaFree(cloud_gpu_);
    CHECK(ret == ::cudaSuccess);
    cloud_gpu_ = nullptr;
  }
}

template <typename PointT>
int PCTransformGPU<PointT>::set_transform_matrix(const float tx, const float ty, const float tz, const float rx,
                                                   const float ry, const float rz)
{
  transform_matrix_ = Eigen::Affine3f::Identity();
  transform_matrix_.translation() << tx, ty, tz;
  transform_matrix_.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));  // The angle of rotation in radians
  transform_matrix_.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
  transform_matrix_.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));

  const auto m4x4 = transform_matrix_.matrix();
  tm_elements_[0] = m4x4(0, 0);
  tm_elements_[1] = m4x4(1, 0);
  tm_elements_[2] = m4x4(2, 0);
  tm_elements_[3] = m4x4(3, 0);

  tm_elements_[4] = m4x4(0, 1);
  tm_elements_[5] = m4x4(1, 1);
  tm_elements_[6] = m4x4(2, 1);
  tm_elements_[7] = m4x4(3, 1);

  tm_elements_[8] = m4x4(0, 2);
  tm_elements_[9] = m4x4(1, 2);
  tm_elements_[10] = m4x4(2, 2);
  tm_elements_[11] = m4x4(3, 2);

  tm_elements_[12] = m4x4(0, 3);
  tm_elements_[13] = m4x4(1, 3);
  tm_elements_[14] = m4x4(2, 3);
  tm_elements_[15] = m4x4(3, 3);

  auto err = cudaMemcpy(tm_elements_gpu_, tm_elements_, 16 * sizeof(float), cudaMemcpyHostToDevice);
  CHECK(err == ::cudaSuccess) << "Cannot memcpy tm_elements_ -> tm_elements_gpu_";

  return 0;
}

template <typename PointT>
bool PCTransformGPU<PointT>::transform(pcl::PointCloud<PointT>& cloud)
{
  if (cloud_gpu_ == nullptr || cloud_gpu_size_ != cloud.points.size() * sizeof(PointT))
  {
    free_cloud_gpu_if_necessary();
    cloud_gpu_size_ = cloud.points.size() * sizeof(PointT);
    auto err = cudaMalloc((void**)&cloud_gpu_, cloud_gpu_size_);
    if (err == ::cudaSuccess)
    {
      LOG(INFO) << "Allocate " << cloud_gpu_size_ << " bytes in cuda";
    }
    else
    {
      LOG(FATAL) << "Cannot allocate " << cloud_gpu_size_ << " bytes for cloud_gpu_size_";
      return false;
    }
  }

  if (cudaMemcpy(cloud_gpu_, cloud.points.data(), cloud_gpu_size_, cudaMemcpyHostToDevice) != ::cudaSuccess)
  {
    return false;
  }

  if (cudaTransformPoints<PointT>(num_cuda_threads_, cloud_gpu_, cloud.points.size(), tm_elements_gpu_) !=
      ::cudaSuccess)
  {
    LOG(INFO) << "Cannot run cudaTransformPoints.";
    return false;
  }

  if (cudaMemcpy(cloud.points.data(), cloud_gpu_, cloud.points.size() * sizeof(PointT), cudaMemcpyDeviceToHost) !=
      ::cudaSuccess)
  {
    return false;
  }

  return true;
}


template <typename PointT>
bool PCTransformGPU<PointT>::transform(sensor_msgs::PointCloud2Ptr msg)
{
  pcl::PointCloud<PointT> cloud;
  auto header = msg->header;
  pcl::fromROSMsg(*msg, cloud);
  bool ret = transform(cloud);
  pcl::toROSMsg(cloud, *msg);
  msg->header = header;
  return ret;
}

template class PCTransformGPU<pcl::PointXYZI>;
};  // namespace pc_transform
