#ifndef __TRANSFORM_CUDA__
#define __TRANSFORM_CUDA__

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace pcl;


cudaError_t
cudaTransformPoints (int threads,
                     pcl::PointXYZ *d_point_cloud,
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

    void
    warmUpGPU ();
    int
    getNumberOfAvailableThreads ();

    bool
    removePointsInsideSphere (pcl::PointCloud<pcl::PointXYZ> &point_cloud);

    bool
    run (pcl::PointCloud<pcl::PointXYZ> &point_cloud,
         Eigen::Affine3f matrix);

    PointCloud<PointXYZI>
    compute (PointCloud<PointXYZI>::Ptr input,
                             float tx,
                             float ty,
                             float tz,
                             float rx,
                             float ry,
                             float rz);

    PointCloud<PointXYZ>
    compute (PointCloud<PointXYZ>::Ptr input,
             float tx,
             float ty,
             float tz,
             float rx,
             float ry,
             float rz);

    PointCloud<PointXYZI>
    compute (PointCloud<PointXYZI>::Ptr input,
             Eigen::Affine3f mr);

    PointCloud<PointXYZ>
    compute (PointCloud<PointXYZ>::Ptr input,
             Eigen::Affine3f mr);

  private:
    static bool hasInitialCUDA;

};

#endif
