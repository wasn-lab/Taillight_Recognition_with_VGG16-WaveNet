#ifndef EDGE_DETECT_H
#define EDGE_DETECT_H

#include "../../UserDefine.h"
#include <cmath>
using namespace pcl;

pcl::PointCloud<PointXYZI> 
getContour(const pcl::PointCloud<PointXYZIL>::Ptr input_cloud, const float theta_sample, const float range_low_bound, const float range_up_bound);

#endif  // EDGE_DETECT_H
