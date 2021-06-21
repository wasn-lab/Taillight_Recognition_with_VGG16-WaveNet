#include <glog/logging.h>
#include <glog/stl_logging.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pc_transform
{
int pc_transform_by_cpu(pcl::PointCloud<pcl::PointXYZI>& cloud, const Eigen::Affine3f& a3f)
{
  for (int i = cloud.points.size() - 1; i >= 0; i--)
  {
    Eigen::Vector3f p{ cloud.points[i].x, cloud.points[i].y, cloud.points[i].z };
    p = a3f * p;
    cloud[i].x = p[0];
    cloud[i].y = p[1];
    cloud[i].z = p[2];
  }
  return 0;
}
};  // namespace pc_transform
