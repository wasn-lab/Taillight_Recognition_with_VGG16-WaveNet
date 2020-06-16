#ifndef __VOXELFILTER_CUDA__
#define __VOXELFILTER_CUDA__

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace std;
using namespace pcl;

#include "../VoxelGrid/VoxelGrid_CUDA.cuh"

class VoxelFilter_CUDA
{
public:
  VoxelFilter_CUDA();
  ~VoxelFilter_CUDA();

  int getNumberOfAvailableThreads();

  bool removeNoiseNaive(pcl::PointCloud<pcl::PointXYZ>& point_cloud, float resolution,
                        int number_of_points_in_bucket_threshold);

  pcl::PointCloud<pcl::PointXYZ> compute(pcl::PointCloud<pcl::PointXYZ>::Ptr input, float resolution,
                                         int number_of_points_in_bucket_threshold);

private:
  static bool hasInitialCUDA;
};

#endif
