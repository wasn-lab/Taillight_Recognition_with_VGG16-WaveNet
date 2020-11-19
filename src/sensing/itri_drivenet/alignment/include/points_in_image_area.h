#ifndef POINTS_IN_IMAGE_AREA_H_
#define POINTS_IN_IMAGE_AREA_H_

/// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/conditional_removal.h>

/// ros
#include <msgs/DetectedObjectArray.h>
#include <msgs/DetectedObject.h>

/// util
#include "drivenet/object_label_util.h"
#include "drivenet/image_preprocessing.h"
#include "point_preprocessing.h"
#include "alignment.h"
#include "cloud_cluster.h"

void getPointCloudInAllImageRectCoverage(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidarall_ptr,
                                         std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_points_ptr,
                                         std::vector<Alignment>& alignment);
void getPointCloudInImageRectCoverage(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidarall_ptr,
                                      pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_points_ptr, Alignment& alignment);
void getPointCloudInImageFOV(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidarall_ptr,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_points_ptr,
                             std::vector<DriveNet::PixelPosition>& cam_pixels, int image_w, int image_h,
                             Alignment& alignment);
void getPointCloudInAllImageFOV(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidarall_ptr,
                                std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_points_ptr,
                                std::vector<std::vector<DriveNet::PixelPosition>>& cam_pixels, int image_w, int image_h,
                                std::vector<Alignment>& alignment);
void getPointCloudInImageFOV(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidarall_ptr,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_points_ptr, int image_w, int image_h,
                             Alignment& alignment);
void getPointCloudInBoxFOV(const msgs::DetectedObjectArray& objects,
                           const pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_points_ptr,
                           pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_bbox_points_ptr,
                           const std::vector<DriveNet::PixelPosition>& cam_pixels_cam,
                           std::vector<std::vector<DriveNet::PixelPosition>>& cam_pixels_obj,
                           msgs::DetectedObjectArray& objects_2d_bbox,
                           std::vector<pcl::PointCloud<pcl::PointXYZI>>& cam_bboxs_points, Alignment& alignment,
                           CloudCluster& cloud_cluster, bool is_enable_default_3d_bbox, bool do_clustering,
                           bool do_display);
void getPointCloudInBoxFOV(const msgs::DetectedObjectArray& objects, msgs::DetectedObjectArray& remaining_objects,
                           const pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_points_ptr,
                           pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_bbox_points_ptr,
                           const std::vector<DriveNet::PixelPosition>& cam_pixels_cam,
                           std::vector<std::vector<DriveNet::PixelPosition>>& cam_pixels_obj,
                           msgs::DetectedObjectArray& objects_2d_bbox,
                           std::vector<pcl::PointCloud<pcl::PointXYZI>>& cam_bboxs_points, Alignment& alignment,
                           CloudCluster& cloud_cluster, bool is_enable_default_3d_bbox, bool do_clustering,
                           bool do_display);
void getPointCloudIn3DBox(const pcl::PointCloud<pcl::PointXYZI>& cloud_src, int object_class_id,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_filtered_ptr, std::vector<int>& inliers_remove);
#endif
