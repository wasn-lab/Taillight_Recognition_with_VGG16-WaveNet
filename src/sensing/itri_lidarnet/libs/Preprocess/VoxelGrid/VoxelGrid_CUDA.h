#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace pcl;

#include "VoxelGrid_CUDA.cuh"
#include "../../UserDefine.h"

class VoxelGrid_CUDA
{
  public:
    VoxelGrid_CUDA ();
    ~VoxelGrid_CUDA ();

    void
    coutMemoryStatus ();

    template <typename PointT>
    bool
    run (typename pcl::PointCloud<PointT> &point_cloud,
         float resolution);

    template <typename PointT>
    PointCloud<PointT>
    compute (typename pcl::PointCloud<PointT>::Ptr input,
             float resolution);

  private:
    static bool hasInitialCUDA;
    static int  maxThreadsNumber;

};

#endif
