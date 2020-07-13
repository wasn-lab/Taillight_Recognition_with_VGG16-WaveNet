#ifndef OBJECT_GENERATOR_H_
#define OBJECT_GENERATOR_H_

/// ros
#include <msgs/BoxPoint.h>

/// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/// util
#include "UseApproxMVBB.h"
#include "point_preprocessing.h"

class ObjectGenerator
{
private:

public:
  pcl::PointCloud<pcl::PointXYZ> pointsToPolygon(pcl::PointCloud<pcl::PointXYZI>& cloud);
  msgs::BoxPoint minMax3dToBBox(MinMax3D& cube);
};

#endif
