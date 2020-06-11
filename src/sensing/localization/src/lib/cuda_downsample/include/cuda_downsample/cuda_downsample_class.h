#ifndef __CUDA_DOWNSAMPLE_CLASS__
#define __CUDA_DOWNSAMPLE_CLASS__

#include "cuda_downsample.h"
#include <pcl/point_cloud.h>

class CudaDownSample
{
public:
  CudaDownSample();
  ~CudaDownSample();

  void warmUpGPU();
  int getNumberOfAvailableThreads();
  void coutMemoryStatus();

  bool downsampling(pcl::PointCloud<pcl::PointXYZI>& point_cloud, float resolution);
  pcl::PointCloud<pcl::PointXYZI> compute(pcl::PointCloud<pcl::PointXYZI>::Ptr input, float resolution);
};

#endif
