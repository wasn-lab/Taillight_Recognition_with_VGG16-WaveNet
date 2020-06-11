#ifndef HEADERS_H_
#define HEADERS_H_

/// ros
#include <ros/package.h>
#include "ros/ros.h"
#include "std_msgs/Header.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

/// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/io/pcd_io.h>

#include <pcl/search/impl/search.hpp>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>

#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/impl/fpfh.hpp>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/approximate_progressive_morphological_filter.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>

#include <pcl/common/common.h>      //getMinMax3D
#include <pcl/common/transforms.h>  //transform​
#include <pcl/common/common_headers.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <boost/chrono.hpp>
#include <pcl/console/parse.h>

#include <pcl/console/time.h>
#include <unistd.h>  //sleep

#include <netinet/in.h>
#include <arpa/inet.h>

#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <linux/can/error.h>

#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_msgs/GridMap.h>

using namespace std;
using namespace pcl;
using namespace pcl::console;
using namespace pcl::visualization;

typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> Cloud;
typedef typename Cloud::ConstPtr CloudConstPtr;

#define PI 3.14159265359
#define DEG_PER_RAD (180.0 / PI)
#define BUF_SIZE 10000

#endif /* HEADERS_H_ */
