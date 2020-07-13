#include "object_generator.h"

pcl::PointCloud<pcl::PointXYZ> ObjectGenerator::pointsToPolygon(pcl::PointCloud<pcl::PointXYZI>& cloud)
{
  std::vector<pcl::PointXYZ> obb_vertex;
  pcl::PointXYZ centroid, min_point, max_point;
  pcl::PointCloud<pcl::PointXYZ> convex_points;

  pcl::copyPointCloud(cloud, convex_points);
  UseApproxMVBB approxMVBB;
  approxMVBB.setInputCloud(convex_points);
  approxMVBB.Compute(obb_vertex, centroid, min_point, max_point, convex_points);

  return convex_points;
}
msgs::BoxPoint ObjectGenerator::minMax3dToBBox(MinMax3D& cube)
{
  /// 3D bounding box
  ///   p5------p6
  ///   /|  2   /|
  /// p1-|----p2 |
  ///  |p4----|-p7
  ///  |/  1  | /
  /// p0-----P3

  msgs::BoxPoint box_point;
  /// min
  box_point.p0.x = cube.p_min.x;
  box_point.p0.y = cube.p_min.y;
  box_point.p0.z = cube.p_min.z;

  /// max
  box_point.p6.x = cube.p_max.x;
  box_point.p6.y = cube.p_max.y;
  box_point.p6.z = cube.p_max.z;

  /// bottom
  box_point.p3 = box_point.p0;
  box_point.p3.y = box_point.p6.y;
  box_point.p7 = box_point.p6;
  box_point.p7.z = box_point.p0.z;
  box_point.p4 = box_point.p0;
  box_point.p4.x = box_point.p6.x;

  /// top
  box_point.p1 = box_point.p0;
  box_point.p1.z = cube.p_max.z;
  box_point.p2 = box_point.p3;
  box_point.p2.z = cube.p_max.z;
  box_point.p5 = box_point.p4;
  box_point.p5.z = cube.p_max.z;

  return box_point;
}