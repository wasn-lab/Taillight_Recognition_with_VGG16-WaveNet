#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

int pc_transform_by_cpu(pcl::PointCloud<pcl::PointXYZI>& cloud, const Eigen::Affine3f& a3f);

