#ifndef HEADERS_H_
#define HEADERS_H_

///ros
#include <ros/package.h>
#include <ros/ros.h>

///pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/console/time.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>

#include <pcl/common/common.h>  //getMinMax3D
#include <pcl/common/transforms.h> //transform​

#include <boost/chrono.hpp>

#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

//ndt_omp
#include "ndt_omp/ndt_omp.h"

using namespace std;
using namespace pcl;

typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> Cloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
typedef typename Cloud::ConstPtr CloudConstPtr;

#define PI 3.14159265359
#define DEG_PER_RAD (180.0/PI)
#define BUF_SIZE 10000

#endif /* ALL_HEADER_H_ */
