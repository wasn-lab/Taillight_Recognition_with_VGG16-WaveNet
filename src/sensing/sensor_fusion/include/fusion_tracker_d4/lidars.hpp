#ifndef LIDARS_HPP
#define LIDARS_HPP

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <fstream>

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
void lidFrontLeftCb(const msgs::PointCloud::ConstPtr& pc);
void lidFrontRightCb(const msgs::PointCloud::ConstPtr& pc);
void lidFrontTopCb(const msgs::PointCloud::ConstPtr& pc);
void lidRearLeftCb(const msgs::PointCloud::ConstPtr& pc);
void lidRearRightCb(const msgs::PointCloud::ConstPtr& pc);




#endif

