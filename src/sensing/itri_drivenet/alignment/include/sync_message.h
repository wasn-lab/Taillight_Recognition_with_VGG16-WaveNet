#ifndef SYNC_MESSAGE_H_
#define SYNC_MESSAGE_H_

/// ros
#include "ros/ros.h"
#include <message_filters/cache.h>
#include <cv_bridge/cv_bridge.h>

/// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

/// lidar lib
#include "UserDefine.h"

cv::Mat getSpecificTimeCameraMessage(message_filters::Cache<sensor_msgs::Image>& cache_image, ros::Time target_time,
                                     ros::Duration duration_time);
pcl::PointCloud<pcl::PointXYZI>::Ptr
getSpecificTimeLidarMessage(message_filters::Cache<pcl::PointCloud<pcl::PointXYZI>>& cache_lidar, ros::Time target_time,
                            ros::Duration duration_time);
pcl::PointCloud<pcl::PointXYZIL>::Ptr
getSpecificTimeLidarMessage(message_filters::Cache<pcl::PointCloud<pcl::PointXYZIL>>& cache_lidar,
                            ros::Time target_time, ros::Duration duration_time);
#endif
