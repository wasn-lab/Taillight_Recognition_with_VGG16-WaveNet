#include <cassert>
#include <gtest/gtest.h>
#include "glog/stl_logging.h"
#include "glog/logging.h"
#include <cstdio>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "pc_transform_cpu.h"
#include "pc_transform_gpu.h"

using namespace pc_transform;

constexpr int cloud_width = 1024;
constexpr int cloud_height = 64;
constexpr int nloop = 100;
constexpr double tx = 0.323;
constexpr double ty = 1.171;
constexpr double tz = -1.942;
constexpr double rx = 0.054;  // radians
constexpr double ry = 0.015;
constexpr double rz = 1.798;

static auto gen_random_cloud(int width=cloud_width, int height=cloud_height)
{
  srand(time(nullptr));
  pcl::PointCloud<pcl::PointXYZI> cloud;
  cloud.width = width;
  cloud.height = height;
  cloud.points.resize(cloud.width * cloud.height);
  cloud.is_dense = true;
  srand(static_cast<unsigned int>(time(nullptr)));

  auto base = RAND_MAX + 1.0;
  for (int i = cloud.points.size() - 1; i >= 0; i--)
  {
    cloud[i].x = static_cast<float>(1024 * (rand() / base));
    cloud[i].y = static_cast<float>(1024 * (rand() / base));
    cloud[i].z = static_cast<float>(1024 * (rand() / base));
    cloud.points[i].intensity = static_cast<float>(i);
  }
  return cloud;
}

static auto get_affine3f()
{
  Eigen::Affine3f a3f = Eigen::Affine3f::Identity();
  a3f.translation() << tx, ty, tz;
  a3f.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));
  a3f.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
  a3f.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));
  return a3f;
}


TEST(kk, test_transform_gpu_eq_cpu_varing_clouds)
{
  PCTransformGPU<pcl::PointXYZI> obj;
  obj.set_transform_matrix(tx, ty, tz, rx, ry, rz);
  for (int k = 0; k < 4; k++)
  {
    int width = 1024 * (1 + rand() % 8);
    int height = 64 * (1 + rand() % 8);
    auto cloud = gen_random_cloud(width, height);
    auto a3f = get_affine3f();
    //  pcl::PointCloud<pcl::PointXYZI>::Ptr input{ new pcl::PointCloud<pcl::PointXYZI> };
    //  *input = cloud;
    auto result_cpu = cloud;
    auto result_gpu = cloud;

    pc_transform_by_cpu(result_cpu, a3f);

    obj.transform(result_gpu);
    EXPECT_EQ(result_cpu.points.size(), result_gpu.points.size());
    EXPECT_EQ(result_cpu.points.size(), width * height);

    for (int i = result_cpu.points.size() - 1; i >= 0; i--)
    {
      EXPECT_TRUE(abs(result_cpu.points[i].x - result_gpu.points[i].x) <= 0.001);
      EXPECT_TRUE(abs(result_cpu.points[i].y - result_gpu.points[i].y) <= 0.001);
      EXPECT_TRUE(abs(result_cpu.points[i].z - result_gpu.points[i].z) <= 0.001);
      EXPECT_EQ(result_cpu.points[i].intensity, result_gpu.points[i].intensity);
    }
  }
}

TEST(kk, test_transform_cpu_perf)
{
  auto cloud = gen_random_cloud();
  auto a3f = get_affine3f();
  for(int i=0; i<nloop; i++)
  {
    pc_transform_by_cpu(cloud, a3f);
  }
}

TEST(kk, test_transform_gpu_perf)
{
  auto cloud = gen_random_cloud();
  PCTransformGPU<pcl::PointXYZI> obj;
  obj.set_transform_matrix(tx, ty, tz, rx, ry, rz);

  for(int i=0; i<nloop; i++)
  {
    obj.transform(cloud);
  }
}
