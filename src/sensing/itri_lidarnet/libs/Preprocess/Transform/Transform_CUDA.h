#ifndef __TRANSFORM_CUDA__
#define __TRANSFORM_CUDA__

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace pcl;

template <typename PointT>
cudaError_t
cudaTransformPoints (int threads,
                     PointT *d_point_cloud,
                     int number_of_points,
                     float *d_matrix);

cudaError_t
cudaRemovePointsInsideSphere (int threads,
                              pcl::PointXYZ *d_point_cloud,
                              bool *d_markers,
                              int number_of_points,
                              float sphere_radius);

class Transform_CUDA
{
  public:
    Transform_CUDA ();
    ~Transform_CUDA ();

    bool
    removePointsInsideSphere (pcl::PointCloud<pcl::PointXYZ> &point_cloud);

    template <typename PointT>
    bool
    run (typename pcl::PointCloud<PointT> &point_cloud,
         Eigen::Affine3f matrix);

    template <typename PointT>
    PointCloud<PointT>
    compute (typename PointCloud<PointT>::Ptr input,
                             float tx,
                             float ty,
                             float tz,
                             float rx,
                             float ry,
                             float rz);

    template <typename PointT>
    PointCloud<PointT>
    compute (typename PointCloud<PointT>::Ptr input,
                             Eigen::Affine3f mr);


  private:
    static bool hasInitialCUDA;
    static int  maxThreadsNumber;

};

#endif
