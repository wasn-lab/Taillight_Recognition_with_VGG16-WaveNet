#include "VoxelGrid_CUDA.h"

using namespace std;
using namespace pcl;

bool VoxelGrid_CUDA::hasInitialCUDA = false;
int VoxelGrid_CUDA::maxThreadsNumber = 0;

VoxelGrid_CUDA::VoxelGrid_CUDA()
{
  if (!hasInitialCUDA)
  {
    cudaError_t err = ::cudaSuccess;
    err = cudaSetDevice(0);
    if (err != ::cudaSuccess)
    {
      return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (prop.major == 2)
    {
      maxThreadsNumber = prop.maxThreadsPerBlock / 2;
    }
    else if (prop.major > 2)
    {
      maxThreadsNumber = prop.maxThreadsPerBlock;
    }
    else
    {
      maxThreadsNumber = 0;
    }

    hasInitialCUDA = true;
  }
}

VoxelGrid_CUDA::~VoxelGrid_CUDA()
{
}

void VoxelGrid_CUDA::coutMemoryStatus()
{
  size_t free_byte;
  size_t total_byte;

  cudaError_t err = cudaMemGetInfo(&free_byte, &total_byte);

  if (err != ::cudaSuccess)
  {
    std::cout << "Error: cudaMemGetInfo fails: " << cudaGetErrorString(err) << std::endl;
    return;
  }
  double free_db = (double)free_byte;
  double total_db = (double)total_byte;
  double used_db = total_db - free_db;

  std::cout << "GPU memory usage: used = " << used_db / 1024.0 / 1024.0 << "(MB), free = " << free_db / 1024.0 / 1024.0
            << "(MB), total = " << total_db / 1024.0 / 1024.0 << "(MB)" << std::endl;
}

template <typename PointT>
bool VoxelGrid_CUDA::run(typename pcl::PointCloud<PointT>& point_cloud, float resolution)
{
  cudaError_t err;
  gridParameters rgd_params;
  PointT* d_point_cloud;
  hashElement* d_hashTable = NULL;
  bucket* d_buckets = NULL;
  bool* d_markers;
  bool* h_markers;

  if (maxThreadsNumber == 0)
  {
    std::cout << "[voxel_grid] Error: max thread equals 0" << std::endl;
    return false;
  }

  err = cudaMalloc((void**)&d_point_cloud, point_cloud.points.size() * sizeof(PointT));
  if (err != ::cudaSuccess)
  {
    std::cout << cudaGetErrorName(err) << std::endl;
    std::cout << "[voxel_grid] Error: Failed to allocate memory for cuda" << std::endl;
    return false;
  }

  err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size() * sizeof(PointT),
                   cudaMemcpyHostToDevice);
  if (err != ::cudaSuccess)
  {
    std::cout << "[voxel_grid] Error: Failed to copy memory for cuda" << std::endl;
    return false;
  }

  err = cudaCalculateGridParams<PointT>(d_point_cloud, point_cloud.points.size(), resolution, resolution, resolution,
                                        rgd_params);
  if (err != ::cudaSuccess)
  {
    std::cout << "[voxel_grid] Error: Failed to calculate grid parameters" << std::endl;
    return false;
  }

  // std::cout << "regular grid parameters:" << std::endl;
  // std::cout << "bounding_box_min_X: " << rgd_params.bounding_box_min_X << std::endl;
  // std::cout << "bounding_box_min_Y: " << rgd_params.bounding_box_min_Y << std::endl;
  // std::cout << "bounding_box_min_Z: " << rgd_params.bounding_box_min_Z << std::endl;
  // std::cout << "bounding_box_max_X: " << rgd_params.bounding_box_max_X << std::endl;
  // std::cout << "bounding_box_max_Y: " << rgd_params.bounding_box_max_Y << std::endl;
  // std::cout << "bounding_box_max_Z: " << rgd_params.bounding_box_max_Z << std::endl;
  // std::cout << "number_of_buckets_X: " << rgd_params.number_of_buckets_X << std::endl;
  // std::cout << "number_of_buckets_Y: " << rgd_params.number_of_buckets_Y << std::endl;
  // std::cout << "number_of_buckets_Z: " << rgd_params.number_of_buckets_Z << std::endl;
  // std::cout << "resolution_X: " << rgd_params.resolution_X << std::endl;
  // std::cout << "resolution_Y: " << rgd_params.resolution_Y << std::endl;
  // std::cout << "resolution_Z: " << rgd_params.resolution_Z << std::endl;

  err = cudaMalloc((void**)&d_hashTable, point_cloud.points.size() * sizeof(hashElement));
  if (err != ::cudaSuccess)
  {
    std::cout << "[voxel_grid] Error: Failed to allocate hash table memory for cuda" << std::endl;
    return false;
  }

  err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets * sizeof(bucket));
  if (err != ::cudaSuccess)
  {
    std::cout << "[voxel_grid] Error: Failed to allocate buckects for cuda" << std::endl;
    return false;
  }

  err = cudaCalculateGrid<PointT>(maxThreadsNumber, d_point_cloud, d_buckets, d_hashTable, point_cloud.points.size(),
                                  rgd_params);
  if (err != ::cudaSuccess)
  {
    std::cout << "[voxel_grid] Error: Failed to calculate grid" << std::endl;
    return false;
  }

  err = cudaMalloc((void**)&d_markers, point_cloud.points.size() * sizeof(bool));
  if (err != ::cudaSuccess)
  {
    std::cout << "[voxel_grid] Error: Failed to allocate marker memory for cuda" << std::endl;
    return false;
  }

  err = cudaDownSample(maxThreadsNumber, d_markers, d_hashTable, d_buckets, rgd_params, point_cloud.points.size());
  if (err != ::cudaSuccess)
  {
    std::cout << "[voxel_grid] Error: Failed to downsampling" << std::endl;
    return false;
  }

  h_markers = (bool*)malloc(point_cloud.points.size() * sizeof(bool));

  err = cudaMemcpy(h_markers, d_markers, point_cloud.points.size() * sizeof(bool), cudaMemcpyDeviceToHost);
  if (err != ::cudaSuccess)
  {
    std::cout << "[voxel_grid] Error: Failed to copy marker memory for cuda" << std::endl;
    return false;
  }

  pcl::PointCloud<PointT> downsampled_point_cloud;
  for (size_t i = 0; i < point_cloud.points.size(); i++)
  {
    if (h_markers[i])
    {
      downsampled_point_cloud.push_back(point_cloud[i]);
    }
  }

  // std::cout << "Number of points before down-sampling: " << point_cloud.size() << std::endl;
  point_cloud = downsampled_point_cloud;
  // std::cout << "Number of points after down-sampling: " << point_cloud.size() << std::endl;

  // std::cout << "Before cudaFree" << std::endl;
  // coutMemoryStatus ();

  free(h_markers);

  err = cudaFree(d_point_cloud);
  d_point_cloud = NULL;
  if (err != ::cudaSuccess)
  {
    return false;
  }

  err = cudaFree(d_hashTable);
  d_hashTable = NULL;
  if (err != ::cudaSuccess)
  {
    return false;
  }

  err = cudaFree(d_buckets);
  d_buckets = NULL;
  if (err != ::cudaSuccess)
  {
    return false;
  }

  err = cudaFree(d_markers);
  d_markers = NULL;
  if (err != ::cudaSuccess)
  {
    return false;
  }

  // std::cout << "After cudaFree" << std::endl;
  // coutMemoryStatus ();

  return true;
}

template bool VoxelGrid_CUDA::run(pcl::PointCloud<PointXYZ>& point_cloud, float resolution);

template bool VoxelGrid_CUDA::run(pcl::PointCloud<PointXYZI>& point_cloud, float resolution);

template bool VoxelGrid_CUDA::run(pcl::PointCloud<PointXYZIL>& point_cloud, float resolution);

template <typename PointT>
PointCloud<PointT> VoxelGrid_CUDA::compute(typename PointCloud<PointT>::Ptr input, float resolution)
{
  PointCloud<PointT> out_cloud;
  out_cloud = *input;

  if (!VoxelGrid_CUDA::run<PointT>(out_cloud, resolution))
  {
    cudaDeviceReset();
    std::cout << "[voxel_grid] NOT SUCCESFULL" << std::endl;
  }

  return out_cloud;
}

template PointCloud<PointXYZ> VoxelGrid_CUDA::compute(pcl::PointCloud<PointXYZ>::Ptr input, float resolution);

template PointCloud<PointXYZI> VoxelGrid_CUDA::compute(pcl::PointCloud<PointXYZI>::Ptr input, float resolution);

template PointCloud<PointXYZIL> VoxelGrid_CUDA::compute(pcl::PointCloud<PointXYZIL>::Ptr input, float resolution);
