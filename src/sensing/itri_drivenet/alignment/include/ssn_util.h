#ifndef SSN_UTIL_H_
#define SSN_UTIL_H_

/// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/// lidar lib
#include "UserDefine.h"

/// Camera lib
#include "drivenet/object_label_util.h"

int transferCommonLabelToSSNLabel(DriveNet::common_type_id label_id);
pcl::PointCloud<pcl::PointXYZI>::Ptr getClassObjectPoint(const pcl::PointCloud<pcl::PointXYZIL>::Ptr& points_ptr,
                                                         DriveNet::common_type_id label_id);
pcl::PointCloud<pcl::PointXYZI>::Ptr getClassObjectPoint(const pcl::PointCloud<pcl::PointXYZIL>::Ptr& points_ptr,
                                                         nnClassID label_id);

#endif
