#ifndef USEAPPROXMVBB_H_
#define USEAPPROXMVBB_H_

#include <iostream>
#include <cstdlib>
#include <cstring>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h> //getMinMax3D
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <Eigen/Eigenvalues>

#include "Core/ComputeApproxMVBB.hpp"
// for computing ConvexHull2D
#include "Core/ConvexHull2D.hpp"

class UseApproxMVBB
{
  public:
    UseApproxMVBB ();
    virtual
    ~UseApproxMVBB ();

    void
    setInputCloud (const pcl::PointCloud<pcl::PointXYZ> input);

    void
    Compute (std::vector<pcl::PointXYZ> &out_cube,
             pcl::PointXYZ &out_centroid,
             pcl::PointXYZ &out_minPoint,
             pcl::PointXYZ &out_maxPoint,
             pcl::PointCloud<pcl::PointXYZ> &out_ch
             );

  private:
    pcl::PointCloud<pcl::PointXYZ> cloud_3d;
};

#endif
