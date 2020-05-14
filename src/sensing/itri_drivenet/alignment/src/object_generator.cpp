#include "object_generator.h"

pcl::PointCloud<pcl::PointXYZ> ObjectGenerator::bbox_to_polygon(pcl::PointCloud<pcl::PointXYZI>& cloud)
{
  std::vector<pcl::PointXYZ> obb_vertex;
  pcl::PointXYZ centroid, minPoint, maxPoint;
  pcl::PointCloud<pcl::PointXYZ> convex_points;

  pcl::copyPointCloud(cloud, convex_points);
  approxMVBB_.setInputCloud(convex_points);
  approxMVBB_.Compute(obb_vertex, centroid, minPoint, maxPoint, convex_points);
  
  return convex_points;
}