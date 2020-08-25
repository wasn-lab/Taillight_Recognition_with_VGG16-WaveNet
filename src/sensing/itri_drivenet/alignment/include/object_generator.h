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

class ObjectGenerator
{
private:
public:
  pcl::PointCloud<pcl::PointXYZ> pointsToPolygon(const pcl::PointCloud<pcl::PointXYZI>& cloud);
  msgs::BoxPoint clusterToBoxPoint(const CLUSTER_INFO& cluster_vector);
  void rotation2D(double xy_input[2], double xy_output[2], double theta);
  void setBoundingBox(CLUSTER_INFO& cluster_info, double pt0[2], double pt3[2], double pt4[2], double pt7[2]);
  msgs::BoxPoint pointsToLShapeBBox(const pcl::PointCloud<pcl::PointXYZI>& cloud, const int class_id);
  msgs::BoxPoint minMax3dToBBox(MinMax3D& cube);
};

#endif
