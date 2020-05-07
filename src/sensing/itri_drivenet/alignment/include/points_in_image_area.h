#ifndef POINTS_IN_IMAGE_AREA_H_
#define POINTS_IN_IMAGE_AREA_H_

/// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/conditional_removal.h>

/// ros
#include <msgs/DetectedObjectArray.h>

/// util
#include "drivenet/object_label_util.h"
#include "drivenet/image_preprocessing.h"
#include "point_preprocessing.h"
#include "alignment.h"

void getPointCloudInImageFOV(pcl::PointCloud<pcl::PointXYZI>::Ptr lidarall_ptr,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr cams_points_ptr,
                             std::vector<DriveNet::PixelPosition>& cam_pixels, int image_w, int image_h,
                             Alignment alignment);
void getPointCloudInBoxFOV(msgs::DetectedObjectArray& objects, pcl::PointCloud<pcl::PointXYZI>::Ptr cams_points_ptr,
                           pcl::PointCloud<pcl::PointXYZI>::Ptr cams_bbox_points_ptr,
                           std::vector<DriveNet::PixelPosition>& cam_pixels,
                           std::vector<pcl_cube>& cams_bboxs_cube_min_max, Alignment alignment,
                           bool is_enable_default_3d_bbox);
void getPointCloudIn3DBox(const pcl::PointCloud<pcl::PointXYZI> cloud_src, int object_class_id,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_ptr);
#endif
