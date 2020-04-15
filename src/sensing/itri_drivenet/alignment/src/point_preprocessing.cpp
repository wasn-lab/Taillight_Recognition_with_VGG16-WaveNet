#include "point_preprocessing.h"

bool comparePoint(pcl::PointXYZI p1, pcl::PointXYZI p2)
{
  if (p1.x != p2.x)
  {
    return p1.x > p2.x;
  }
  else if (p1.y != p2.y)
  {
    return p1.y > p2.y;
  }
  else
  {
    return p1.z > p2.z;
  }
}
bool equalPoint(pcl::PointXYZI p1, pcl::PointXYZI p2)
{
  return (p1.x == p2.x && p1.y == p2.y && p1.z == p2.z);
}
void removeDuplePoints(std::vector<pcl::PointXYZI>& points)
{
  std::sort(points.begin(), points.end(), comparePoint);
  auto unique_end = std::unique(points.begin(), points.end(), equalPoint);
  points.erase(unique_end, points.end());
}