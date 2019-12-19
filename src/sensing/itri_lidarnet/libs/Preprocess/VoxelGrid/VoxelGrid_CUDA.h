#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace pcl;

#include "VoxelGrid_CUDA.cuh"

class VoxelGrid_CUDA
{
  public:
    VoxelGrid_CUDA ();
    ~VoxelGrid_CUDA ();

    void
    coutMemoryStatus ();
    bool
    downsampling (pcl::PointCloud<pcl::PointXYZ> &point_cloud,
                  float resolution);

    PointCloud<PointXYZ>
    compute (PointCloud<PointXYZ>::Ptr input,
             float resolution);
  private:
    static bool hasInitialCUDA;
    static int  maxThreadsNumber;

};

#endif
