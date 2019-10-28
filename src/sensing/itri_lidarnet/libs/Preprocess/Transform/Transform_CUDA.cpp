#include "Transform_CUDA.h"

bool Transform_CUDA::hasInitialCUDA = false;

Transform_CUDA::Transform_CUDA ()
{
  if (!hasInitialCUDA)
  {
    cudaError_t err = ::cudaSuccess;
    err = cudaSetDevice (0);
    if (err != ::cudaSuccess){
      return;
    }
    hasInitialCUDA = true;
  }
}

Transform_CUDA::~Transform_CUDA ()
{

}

int
Transform_CUDA::getNumberOfAvailableThreads ()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties (&prop, 0);

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

  err = cudaRemovePointsInsideSphere (getNumberOfAvailableThreads (), d_point_cloud, d_markers, point_cloud.points.size (), sphere_radius);
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

bool
Transform_CUDA::run (pcl::PointCloud<pcl::PointXYZ> &point_cloud,
                     Eigen::Affine3f matrix)
{
  int threads;
  pcl::PointXYZ * d_point_cloud;

  float h_m[16];
  float *d_m;

  cudaError_t err = ::cudaSuccess;
  err = cudaSetDevice (0);
  if (err != ::cudaSuccess)
    return false;

  threads = getNumberOfAvailableThreads ();

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

  err = cudaMalloc ((void**) &d_m, 16 * sizeof(float));
  if (err != ::cudaSuccess)
    return false;

  err = cudaMemcpy (d_m, h_m, 16 * sizeof(float), cudaMemcpyHostToDevice);
  if (err != ::cudaSuccess)
    return false;

  err = cudaMalloc ((void**) &d_point_cloud, point_cloud.points.size () * sizeof(pcl::PointXYZ));
  if (err != ::cudaSuccess)
    return false;

  err = cudaMemcpy (d_point_cloud, point_cloud.points.data (), point_cloud.points.size () * sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice);
  if (err != ::cudaSuccess)
    return false;

  err = cudaTransformPoints (threads, d_point_cloud, point_cloud.points.size (), d_m);
  if (err != ::cudaSuccess)
    return false;

  err = cudaMemcpy (point_cloud.points.data (), d_point_cloud, point_cloud.points.size () * sizeof(pcl::PointXYZ), cudaMemcpyDeviceToHost);
  if (err != ::cudaSuccess)
    return false;

  err = cudaFree (d_m);
  if (err != ::cudaSuccess)
    return false;

  err = cudaFree (d_point_cloud);
  d_point_cloud = NULL;
  if (err != ::cudaSuccess)
    return false;

  return true;
}

PointCloud<PointXYZI>
Transform_CUDA::compute (PointCloud<PointXYZI>::Ptr input,
                         float tx,
                         float ty,
                         float tz,
                         float rx,
                         float ry,
                         float rz)
{

  PointCloud<PointXYZ> buff;
  pcl::copyPointCloud (*input, buff);

  Eigen::Affine3f mr = Eigen::Affine3f::Identity ();

  mr.translation () << tx, ty, tz;

  // The angle of rotation in radians
  mr.rotate (Eigen::AngleAxisf (rx, Eigen::Vector3f::UnitX ()));
  mr.rotate (Eigen::AngleAxisf (ry, Eigen::Vector3f::UnitY ()));
  mr.rotate (Eigen::AngleAxisf (rz, Eigen::Vector3f::UnitZ ()));

  //pcl::transformPointCloud (*input, buff, mr); // no cuda

  if (!Transform_CUDA::run (buff, mr))
  {
    std::cout << "Problem with transform" << std::endl;
    cudaDeviceReset ();
  }

  PointCloud<PointXYZI> out_cloud;
  pcl::copyPointCloud (buff, out_cloud);

  for (size_t i = 0; i < out_cloud.points.size (); ++i)
  {
    out_cloud.points[i].intensity = input->points[i].intensity;
  }

  return out_cloud;
}

PointCloud<PointXYZ>
Transform_CUDA::compute (PointCloud<PointXYZ>::Ptr input,
                         float tx,
                         float ty,
                         float tz,
                         float rx,
                         float ry,
                         float rz)
{

  PointCloud<PointXYZ> out_cloud;
  out_cloud = *input;

  Eigen::Affine3f mr = Eigen::Affine3f::Identity ();

  mr.translation () << tx, ty, tz;

  // The angle of rotation in radians
  mr.rotate (Eigen::AngleAxisf (rx, Eigen::Vector3f::UnitX ()));
  mr.rotate (Eigen::AngleAxisf (ry, Eigen::Vector3f::UnitY ()));
  mr.rotate (Eigen::AngleAxisf (rz, Eigen::Vector3f::UnitZ ()));

  //pcl::transformPointCloud (*input, out_cloud, mr); // no cuda

  if (!Transform_CUDA::run (out_cloud, mr))
  {
    std::cout << "Problem with transform" << std::endl;
    cudaDeviceReset ();
  }

  return out_cloud;
}

PointCloud<PointXYZI>
Transform_CUDA::compute (PointCloud<PointXYZI>::Ptr input,
                         Eigen::Affine3f mr)
{
  PointCloud<PointXYZ> buff;
  pcl::copyPointCloud (*input, buff);

  if (!Transform_CUDA::run (buff, mr))
  {
    std::cout << "Problem with transform" << std::endl;
    cudaDeviceReset ();
  }

  PointCloud<PointXYZI> out_cloud;
  pcl::copyPointCloud (buff, out_cloud);

  for (size_t i = 0; i < out_cloud.points.size (); ++i)
  {
    out_cloud.points[i].intensity = input->points[i].intensity;
  }

  return out_cloud;
}

PointCloud<PointXYZ>
Transform_CUDA::compute (PointCloud<PointXYZ>::Ptr input,
                         Eigen::Affine3f mr)
{

  PointCloud<PointXYZ> out_cloud;
  out_cloud = *input;

  if (!Transform_CUDA::run (out_cloud, mr))
  {
    std::cout << "Problem with transform" << std::endl;
    cudaDeviceReset ();
  }

  return out_cloud;
}
