#include "Transform_CUDA.h"

bool Transform_CUDA::hasInitialCUDA = false;
int Transform_CUDA::maxThreadsNumber = 0;

Transform_CUDA::Transform_CUDA ()
{
  if (!hasInitialCUDA)
  {
    cudaError_t err = ::cudaSuccess;
    err = cudaSetDevice (0);
    if (err != ::cudaSuccess){
      return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties (&prop, 0);

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

Transform_CUDA::~Transform_CUDA ()
{

}

bool
Transform_CUDA::removePointsInsideSphere (pcl::PointCloud<pcl::PointXYZ> &point_cloud)
{
  float sphere_radius = 1.0f;
  pcl::PointXYZ * d_point_cloud;
  bool* d_markers;
  bool* h_markers;

  cudaError_t err = ::cudaSuccess;
  err = cudaSetDevice (0);
  if (err != ::cudaSuccess)
    return false;

  err = cudaMalloc ((void**) &d_point_cloud, point_cloud.points.size () * sizeof(pcl::PointXYZ));
  if (err != ::cudaSuccess)
    return false;

  err = cudaMemcpy (d_point_cloud, point_cloud.points.data (), point_cloud.points.size () * sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice);
  if (err != ::cudaSuccess)
    return false;

  err = cudaMalloc ((void**) &d_markers, point_cloud.points.size () * sizeof(bool));
  if (err != ::cudaSuccess)
    return false;

  err = cudaRemovePointsInsideSphere (maxThreadsNumber, d_point_cloud, d_markers, point_cloud.points.size (), sphere_radius);
  if (err != ::cudaSuccess)
    return false;

  h_markers = (bool *) malloc (point_cloud.points.size () * sizeof(bool));

  err = cudaMemcpy (h_markers, d_markers, point_cloud.points.size () * sizeof(bool), cudaMemcpyDeviceToHost);
  if (err != ::cudaSuccess)
    return false;

  pcl::PointCloud<pcl::PointXYZ> new_point_cloud;
  for (size_t i = 0; i < point_cloud.points.size (); i++)
  {
    if (h_markers[i])
      new_point_cloud.push_back (point_cloud[i]);
  }

  std::cout << "Number of points before removing points: " << point_cloud.size () << std::endl;
  point_cloud = new_point_cloud;
  std::cout << "Number of points after removing points: " << point_cloud.size () << std::endl;

  free (h_markers);

  err = cudaFree (d_markers);
  d_markers = NULL;
  if (err != ::cudaSuccess)
    return false;

  err = cudaFree (d_point_cloud);
  d_point_cloud = NULL;
  if (err != ::cudaSuccess)
    return false;

  return true;
}

template <typename PointT>
bool
Transform_CUDA::run (typename pcl::PointCloud<PointT> &point_cloud,
                     Eigen::Affine3f matrix)
{
  PointT *d_point_cloud;
  float *d_m;
  float h_m[16];

  h_m[0] = matrix.matrix () (0, 0);
  h_m[1] = matrix.matrix () (1, 0);
  h_m[2] = matrix.matrix () (2, 0);
  h_m[3] = matrix.matrix () (3, 0);

  h_m[4] = matrix.matrix () (0, 1);
  h_m[5] = matrix.matrix () (1, 1);
  h_m[6] = matrix.matrix () (2, 1);
  h_m[7] = matrix.matrix () (3, 1);

  h_m[8] = matrix.matrix () (0, 2);
  h_m[9] = matrix.matrix () (1, 2);
  h_m[10] = matrix.matrix () (2, 2);
  h_m[11] = matrix.matrix () (3, 2);

  h_m[12] = matrix.matrix () (0, 3);
  h_m[13] = matrix.matrix () (1, 3);
  h_m[14] = matrix.matrix () (2, 3);
  h_m[15] = matrix.matrix () (3, 3);

  if (cudaMalloc ((void**) &d_m, 16 * sizeof(float)) != ::cudaSuccess)
    return false;

  if (cudaMemcpy (d_m, h_m, 16 * sizeof(float), cudaMemcpyHostToDevice) != ::cudaSuccess)
    return false;

  if (cudaMalloc ((void**) &d_point_cloud, point_cloud.points.size () * sizeof(PointT)) != ::cudaSuccess)
    return false;

  if (cudaMemcpy (d_point_cloud, point_cloud.points.data (), point_cloud.points.size () * sizeof(PointT), cudaMemcpyHostToDevice) != ::cudaSuccess)
    return false;

  if (cudaTransformPoints<PointT> (maxThreadsNumber, d_point_cloud, point_cloud.points.size (), d_m) != ::cudaSuccess)
    return false;

  if (cudaMemcpy (point_cloud.points.data (), d_point_cloud, point_cloud.points.size () * sizeof(PointT), cudaMemcpyDeviceToHost) != ::cudaSuccess)
    return false;

  if (cudaFree (d_m) != ::cudaSuccess)
    return false;

  if (cudaFree (d_point_cloud) != ::cudaSuccess)
    return false;

  d_point_cloud = NULL;

  return true;
}

template
bool
Transform_CUDA::run (pcl::PointCloud<PointXYZ> &point_cloud,
                     Eigen::Affine3f matrix);

template
bool
Transform_CUDA::run (pcl::PointCloud<PointXYZI> &point_cloud,
                     Eigen::Affine3f matrix);


template <typename PointT>
PointCloud<PointT>
Transform_CUDA::compute (typename PointCloud<PointT>::Ptr input,
                         float tx,
                         float ty,
                         float tz,
                         float rx,
                         float ry,
                         float rz)
{

  PointCloud<PointT> out_cloud;
  out_cloud = *input;

  Eigen::Affine3f mr = Eigen::Affine3f::Identity ();

  mr.translation () << tx, ty, tz;
  mr.rotate (Eigen::AngleAxisf (rx, Eigen::Vector3f::UnitX ())); // The angle of rotation in radians
  mr.rotate (Eigen::AngleAxisf (ry, Eigen::Vector3f::UnitY ()));
  mr.rotate (Eigen::AngleAxisf (rz, Eigen::Vector3f::UnitZ ()));

  //pcl::transformPointCloud (*input, out_cloud, mr); // no cuda

  if (!Transform_CUDA::run<PointT> (out_cloud, mr))
  {
    std::cout << "Problem with transform" << std::endl;
    cudaDeviceReset ();
  }

  return out_cloud;
}

template
PointCloud<PointXYZ>
Transform_CUDA::compute (typename PointCloud<PointXYZ>::Ptr input,
                         float tx,
                         float ty,
                         float tz,
                         float rx,
                         float ry,
                         float rz);

template
PointCloud<PointXYZI>
Transform_CUDA::compute (typename PointCloud<PointXYZI>::Ptr input,
                         float tx,
                         float ty,
                         float tz,
                         float rx,
                         float ry,
                         float rz);

template <typename PointT>
PointCloud<PointT>
Transform_CUDA::compute (typename PointCloud<PointT>::Ptr input,
                         Eigen::Affine3f mr)
{

  PointCloud<PointT> out_cloud;
  out_cloud = *input;

  if (!Transform_CUDA::run (out_cloud, mr))
  {
    std::cout << "Problem with transform" << std::endl;
    cudaDeviceReset ();
  }

  return out_cloud;
}

template
PointCloud<PointXYZ>
Transform_CUDA::compute (typename PointCloud<PointXYZ>::Ptr input,
                         Eigen::Affine3f mr);

template
PointCloud<PointXYZI>
Transform_CUDA::compute (typename PointCloud<PointXYZI>::Ptr input,
                         Eigen::Affine3f mr);
