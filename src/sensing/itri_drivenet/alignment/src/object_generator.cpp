#include "object_generator.h"
#include <pcl/common/geometry.h> 
#include "drivenet/object_label_util.h"
#include "UseApproxMVBB.h"
#include "shape_estimator.hpp" // L-shape estimator

pcl::PointCloud<pcl::PointXYZ> ObjectGenerator::pointsToPolygon(const pcl::PointCloud<pcl::PointXYZI>& cloud)
{
  std::vector<pcl::PointXYZ> obb_vertex;
  pcl::PointXYZ centroid, min_point, max_point;
  pcl::PointCloud<pcl::PointXYZ> convex_points;

  pcl::copyPointCloud(cloud, convex_points);
  UseApproxMVBB approx_mvbb;
  approx_mvbb.setInputCloud(convex_points);
  approx_mvbb.Compute(obb_vertex, centroid, min_point, max_point, convex_points);

  return convex_points;
}
msgs::BoxPoint ObjectGenerator::clusterToBoxPoint(const CLUSTER_INFO& cluster_vector)
{
  msgs::BoxPoint box_point;
  if (!cluster_vector.obb_vertex.empty())
  {
    /// min
    box_point.p0.x = cluster_vector.obb_vertex[0].x;
    box_point.p0.y = cluster_vector.obb_vertex[0].y;
    box_point.p0.z = cluster_vector.obb_vertex[0].z;

    /// max
    box_point.p6.x = cluster_vector.obb_vertex[6].x;
    box_point.p6.y = cluster_vector.obb_vertex[6].y;
    box_point.p6.z = cluster_vector.obb_vertex[6].z;

    /// bottom
    box_point.p3.x = cluster_vector.obb_vertex[3].x;
    box_point.p3.y = cluster_vector.obb_vertex[3].y;
    box_point.p3.z = cluster_vector.obb_vertex[3].z;
    box_point.p7.x = cluster_vector.obb_vertex[7].x;
    box_point.p7.y = cluster_vector.obb_vertex[7].y;
    box_point.p7.z = cluster_vector.obb_vertex[7].z;
    box_point.p4.x = cluster_vector.obb_vertex[4].x;
    box_point.p4.y = cluster_vector.obb_vertex[4].y;
    box_point.p4.z = cluster_vector.obb_vertex[4].z;

    /// top
    box_point.p1.x = cluster_vector.obb_vertex[1].x;
    box_point.p1.y = cluster_vector.obb_vertex[1].y;
    box_point.p1.z = cluster_vector.obb_vertex[1].z;
    box_point.p2.x = cluster_vector.obb_vertex[2].x;
    box_point.p2.y = cluster_vector.obb_vertex[2].y;
    box_point.p2.z = cluster_vector.obb_vertex[2].z;
    box_point.p5.x = cluster_vector.obb_vertex[5].x;
    box_point.p5.y = cluster_vector.obb_vertex[5].y;
    box_point.p5.z = cluster_vector.obb_vertex[5].z;
  }
  return box_point;
}

msgs::BoxPoint ObjectGenerator::pointsToLShapeBBox(const pcl::PointCloud<pcl::PointXYZI>& cloud, const int class_id)
{
  /// Preprocessing
  CLUSTER_INFO cluster_vector;
  pcl::PointCloud<pcl::PointXYZ> convex_points;
  pcl::copyPointCloud(cloud, convex_points);
  cluster_vector.cloud = convex_points;

  pcl::getMinMax3D(cluster_vector.cloud, cluster_vector.min, cluster_vector.max);
  cluster_vector.dx = fabs(cluster_vector.max.x - cluster_vector.min.x);
  cluster_vector.dy = fabs(cluster_vector.max.y - cluster_vector.min.y);
  cluster_vector.dz = fabs(cluster_vector.max.z - cluster_vector.min.z);
  cluster_vector.center = pcl::PointXYZ((cluster_vector.max.x + cluster_vector.min.x) / 2,
                                          (cluster_vector.max.y + cluster_vector.min.y) / 2, 0);

  cluster_vector.dis_center_origin = pcl::geometry::distance(cluster_vector.center, pcl::PointXYZ(0, 0, 0));

  cluster_vector.abb_vertex.resize(8);
  cluster_vector.abb_vertex.at(0) =
      pcl::PointXYZ(cluster_vector.min.x, cluster_vector.min.y, cluster_vector.min.z);
  cluster_vector.abb_vertex.at(1) =
      pcl::PointXYZ(cluster_vector.min.x, cluster_vector.min.y, cluster_vector.max.z);
  cluster_vector.abb_vertex.at(2) =
      pcl::PointXYZ(cluster_vector.max.x, cluster_vector.min.y, cluster_vector.max.z);
  cluster_vector.abb_vertex.at(3) =
      pcl::PointXYZ(cluster_vector.max.x, cluster_vector.min.y, cluster_vector.min.z);
  cluster_vector.abb_vertex.at(4) =
      pcl::PointXYZ(cluster_vector.min.x, cluster_vector.max.y, cluster_vector.min.z);
  cluster_vector.abb_vertex.at(5) =
      pcl::PointXYZ(cluster_vector.min.x, cluster_vector.max.y, cluster_vector.max.z);
  cluster_vector.abb_vertex.at(6) =
      pcl::PointXYZ(cluster_vector.max.x, cluster_vector.max.y, cluster_vector.max.z);
  cluster_vector.abb_vertex.at(7) =
      pcl::PointXYZ(cluster_vector.max.x, cluster_vector.max.y, cluster_vector.min.z);

  /// L-shape estimator
  ShapeEstimator estimator;
  bool do_apply_filter = false; // default: false
  if (class_id == static_cast<int>(DriveNet::common_type_id::person))
  {
    estimator.getShapeAndPose(nnClassID::Person, cluster_vector, do_apply_filter);
  }
  else if (class_id == static_cast<int>(DriveNet::common_type_id::bicycle) ||
      class_id == static_cast<int>(DriveNet::common_type_id::motorbike))
  {
    estimator.getShapeAndPose(nnClassID::Motobike, cluster_vector, do_apply_filter);
  }
  else if (class_id == static_cast<int>(DriveNet::common_type_id::car) ||
      class_id == static_cast<int>(DriveNet::common_type_id::bus) ||
      class_id == static_cast<int>(DriveNet::common_type_id::truck))
  {
    estimator.getShapeAndPose(nnClassID::Car, cluster_vector, do_apply_filter);
  }

  msgs::BoxPoint box_point;
  box_point = clusterToBoxPoint(cluster_vector);

  return box_point;
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