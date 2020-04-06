#include "VoxelFilter_CUDA.h"

bool VoxelFilter_CUDA::hasInitialCUDA = false;

VoxelFilter_CUDA::VoxelFilter_CUDA()
{
  if (!hasInitialCUDA)
  {
    cudaError_t err = ::cudaSuccess;
    err = cudaSetDevice(0);
    if (err != ::cudaSuccess)
    {
      return;
    }
    hasInitialCUDA = true;
  }
}

VoxelFilter_CUDA::~VoxelFilter_CUDA()
{
}

int VoxelFilter_CUDA::getNumberOfAvailableThreads()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  int threads = 0;
  if (prop.major == 2)
  {
    threads = prop.maxThreadsPerBlock / 2;
  }
  else if (prop.major > 2)
  {
    threads = prop.maxThreadsPerBlock;
  }
  else
  {
    return 0;
  }

  return threads;
}

bool VoxelFilter_CUDA::removeNoiseNaive(pcl::PointCloud<pcl::PointXYZ>& point_cloud, float resolution,
                                        int number_of_points_in_bucket_threshold)
{
  cudaError_t err = ::cudaSuccess;
  err = cudaSetDevice(0);
  if (err != ::cudaSuccess)
  {
    return false;
  }

  gridParameters rgd_params;
  pcl::PointXYZ* d_point_cloud;
  hashElement* d_hashTable = NULL;
  bucket* d_buckets = NULL;
  bool* d_markers;
  bool* h_markers;
  int threads = getNumberOfAvailableThreads();

  // std::cout << "CUDA code will use " << threads << " device threads" << std::endl;
  if (threads == 0)
  {
    return false;
  }

  err = cudaMalloc((void**)&d_point_cloud, point_cloud.points.size() * sizeof(pcl::PointXYZ));
  if (err != ::cudaSuccess)
  {
    return false;
  }

  err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size() * sizeof(pcl::PointXYZ),
                   cudaMemcpyHostToDevice);
  if (err != ::cudaSuccess)
  {
    return false;
  }

  err =
      cudaCalculateGridParams(d_point_cloud, point_cloud.points.size(), resolution, resolution, resolution, rgd_params);
  if (err != ::cudaSuccess)
  {
    return false;
  }

  //  std::cout << "regular grid parameters:" << std::endl;
  //  std::cout << "bounding_box_min_X: " << rgd_params.bounding_box_min_X << std::endl;
  //  std::cout << "bounding_box_min_Y: " << rgd_params.bounding_box_min_Y << std::endl;
  //  std::cout << "bounding_box_min_Z: " << rgd_params.bounding_box_min_Z << std::endl;
  //  std::cout << "bounding_box_max_X: " << rgd_params.bounding_box_max_X << std::endl;
  //  std::cout << "bounding_box_max_Y: " << rgd_params.bounding_box_max_Y << std::endl;
  //  std::cout << "bounding_box_max_Z: " << rgd_params.bounding_box_max_Z << std::endl;
  //  std::cout << "number_of_buckets_X: " << rgd_params.number_of_buckets_X << std::endl;
  //  std::cout << "number_of_buckets_Y: " << rgd_params.number_of_buckets_Y << std::endl;
  //  std::cout << "number_of_buckets_Z: " << rgd_params.number_of_buckets_Z << std::endl;
  //  std::cout << "resolution_X: " << rgd_params.resolution_X << std::endl;
  //  std::cout << "resolution_Y: " << rgd_params.resolution_Y << std::endl;
  //  std::cout << "resolution_Z: " << rgd_params.resolution_Z << std::endl;

  err = cudaMalloc((void**)&d_hashTable, point_cloud.points.size() * sizeof(hashElement));
  if (err != ::cudaSuccess)
  {
    return false;
  }

  err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets * sizeof(bucket));
  if (err != ::cudaSuccess)
  {
    return false;
  }

  err = cudaCalculateGrid(threads, d_point_cloud, d_buckets, d_hashTable, point_cloud.points.size(), rgd_params);
  if (err != ::cudaSuccess)
  {
    return false;
  }

  err = cudaMalloc((void**)&d_markers, point_cloud.points.size() * sizeof(bool));
  if (err != ::cudaSuccess)
  {
    return false;
  }

  err = cudaRemoveNoiseNaive(threads, d_markers, d_point_cloud, d_hashTable, d_buckets, rgd_params,
                             point_cloud.points.size(), number_of_points_in_bucket_threshold);
  if (err != ::cudaSuccess)
  {
    return false;
  }

  h_markers = (bool*)malloc(point_cloud.points.size() * sizeof(bool));

  err = cudaMemcpy(h_markers, d_markers, point_cloud.points.size() * sizeof(bool), cudaMemcpyDeviceToHost);
  if (err != ::cudaSuccess)
  {
    return false;
  }

  pcl::PointCloud<pcl::PointXYZ> filtered_point_cloud;
  for (size_t i = 0; i < point_cloud.points.size(); i++)
  {
    if (h_markers[i])
    {
      filtered_point_cloud.push_back(point_cloud[i]);
    }
  }

  // std::cout << "Number of points before filtering: " << point_cloud.size () << std::endl;

  point_cloud = filtered_point_cloud;
  // std::cout << "Number of points after filtering: " << point_cloud.size () << std::endl;

  // std::cout << "Before cudaFree" << std::endl;

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

  return true;
}

PointCloud<PointXYZ> VoxelFilter_CUDA::compute(PointCloud<PointXYZ>::Ptr input, float resolution,
                                               int number_of_points_in_bucket_threshold)
{
  PointCloud<PointXYZ> out_cloud;
  out_cloud = *input;

  if (!VoxelFilter_CUDA::removeNoiseNaive(out_cloud, resolution, number_of_points_in_bucket_threshold))
  {
    cudaDeviceReset();
    std::cout << "removeNoiseNaive NOT SUCCESFULL" << std::endl;
  }
  return out_cloud;
}
