#ifndef __ALIGNMENT_UTILS_H__
#define __ALIGNMENT_UTILS_H__

#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace alignment
{
cv::Point map_pcd_ground_point_to_image_point(const pcl::PointXYZ& pcd_point, const int cam_sn);
cv::Point map_pcd_point_to_image_point(const pcl::PointXYZ& pcd_point, const int cam_sn);
int map_pcd_ground_to_image_points(pcl::PointCloud<pcl::PointXYZ> release_cloud, const int cam_sn);
int map_pcd_to_image_points(pcl::PointCloud<pcl::PointXYZ> release_cloud, const int cam_sn);
double* get_dist_by_image_point(pcl::PointCloud<pcl::PointXYZ> release_cloud, int image_x, int image_y,
                                const int cam_sn);
int get_image_x_min(const int image_x, const int cam_sn);
int get_image_x_max(const int image_x, const int cam_sn);
int get_image_y_min(const int image_y, const int cam_sn);
int get_image_y_max(const int image_y, const int cam_sn);
};
#endif  // __ALIGNMENT_UTILS_H__
