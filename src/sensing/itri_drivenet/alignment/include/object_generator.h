#ifndef OBJECT_GENERATOR_H_
#define OBJECT_GENERATOR_H_

/// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/// util
#include "UseApproxMVBB.h"

class ObjectGenerator
{
private:
  UseApproxMVBB approxMVBB_;

public:
  pcl::PointCloud<pcl::PointXYZ> pointsToPolygon(pcl::PointCloud<pcl::PointXYZI>& cloud);
};

#endif
