#include <cstdlib>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <gtest/gtest.h>
#include "pc2_compressor.h"
#include "point_os1.h"

extern pcl::PointCloud<ouster_ros::OS1::PointOS1> g_cloud;
extern pcl::PCLPointCloud2 g_pcl_pc2;
extern sensor_msgs::PointCloud2Ptr g_org_ros_pc2_ptr;

void gen_rand_cloud();
