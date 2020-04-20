#ifndef POINT_PREPROCESSING_H_
#define POINT_PREPROCESSING_H_

///
#include <vector>
/// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

bool comparePoint(pcl::PointXYZI p1, pcl::PointXYZI p2);
bool equalPoint(pcl::PointXYZI p1, pcl::PointXYZI p2);
void removeDuplePoints(std::vector<pcl::PointXYZI>& points);

#endif
