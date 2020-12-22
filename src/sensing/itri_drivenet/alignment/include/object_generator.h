#ifndef OBJECT_GENERATOR_H_
#define OBJECT_GENERATOR_H_

/// ros
#include <msgs/BoxPoint.h>

/// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/// util
#include "point_preprocessing.h"
#include "UserDefine.h"  // CLUSTER_INFO struct

struct OrientedBBox
{
  std::vector<pcl::PointXYZ> obb_vertex;
  pcl::PointXYZ obb_center;
  float obb_dx;
  float obb_dy;
  float obb_dz;
  float obb_orient;  // when z = 0
};

class ObjectGenerator
{
private:
public:
  pcl::PointCloud<pcl::PointXYZ> pointsToPolygon(const pcl::PointCloud<pcl::PointXYZI>& cloud);
  OrientedBBox clusterToOrientedBBox(const CLUSTER_INFO& cluster_vector);
  msgs::BoxPoint clusterToBoxPoint(const CLUSTER_INFO& cluster_vector);
  void rotation2D(double xy_input[2], double xy_output[2], double theta);
  void setBoundingBox(CLUSTER_INFO& cluster_info, double pt0[2], double pt3[2], double pt4[2], double pt7[2]);
  void pointsToLShapeBBox(const pcl::PointCloud<pcl::PointXYZI>& cloud, const int class_id, OrientedBBox& obb, msgs::BoxPoint& box_point);
  msgs::BoxPoint minMax3dToBBox(MinMax3D& cube);
};

#endif
