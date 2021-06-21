#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

template <typename PointT>
__global__ void kernel_cudaTransformPoints(PointT* d_point_cloud, const int number_of_points, float* d_matrix)
{
  int ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < number_of_points)
  {
    float vSrcVector[3] = { d_point_cloud[ind].x, d_point_cloud[ind].y, d_point_cloud[ind].z };
    d_point_cloud[ind].x = d_matrix[0] * vSrcVector[0] + d_matrix[4] * vSrcVector[1] + d_matrix[8] * vSrcVector[2] + d_matrix[12];
    d_point_cloud[ind].y = d_matrix[1] * vSrcVector[0] + d_matrix[5] * vSrcVector[1] + d_matrix[9] * vSrcVector[2] + d_matrix[13];
    d_point_cloud[ind].z = d_matrix[2] * vSrcVector[0] + d_matrix[6] * vSrcVector[1] + d_matrix[10] * vSrcVector[2] + d_matrix[14];

  }
}

template __global__ void kernel_cudaTransformPoints(pcl::PointXYZ* d_point_cloud, const int number_of_points,
                                                    float* d_matrix);

template __global__ void kernel_cudaTransformPoints(pcl::PointXYZI* d_point_cloud, const int number_of_points,
                                                    float* d_matrix);

template <typename PointT>
cudaError_t cudaTransformPoints(const int threads, PointT* d_point_cloud, const int number_of_points, float* d_matrix)
{
  kernel_cudaTransformPoints<<<number_of_points / threads + 1, threads>>>(d_point_cloud, number_of_points, d_matrix);

  cudaDeviceSynchronize();
  return cudaGetLastError();
}

template cudaError_t cudaTransformPoints(const int threads, pcl::PointXYZ* d_point_cloud, const int number_of_points,
                                         float* d_matrix);

template cudaError_t cudaTransformPoints(const int threads, pcl::PointXYZI* d_point_cloud, const int number_of_points,
                                         float* d_matrix);

__global__ void kernel_cudaRemovePointsInsideSphere(pcl::PointXYZ* d_point_cloud, bool* d_markers, int number_of_points,
                                                    float sphere_radius)
{
  int ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < number_of_points)
  {
    float x = d_point_cloud[ind].x;
    float y = d_point_cloud[ind].y;
    float z = d_point_cloud[ind].z;

    float distance = (x * x + y * y + z * z);

    if (distance < sphere_radius * sphere_radius)
    {
      d_markers[ind] = false;
    }
    else
    {
      d_markers[ind] = true;
    }
  }
}

cudaError_t cudaRemovePointsInsideSphere(int threads, pcl::PointXYZ* d_point_cloud, bool* d_markers,
                                         int number_of_points, float sphere_radius)
{
  kernel_cudaRemovePointsInsideSphere<<<number_of_points / threads + 1, threads>>>(d_point_cloud, d_markers,
                                                                                   number_of_points, sphere_radius);

  cudaDeviceSynchronize();
  return cudaGetLastError();
}

