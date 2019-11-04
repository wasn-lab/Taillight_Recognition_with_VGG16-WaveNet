
#ifndef ALL_HEADER_H_
#define ALL_HEADER_H_

// =============================================
//                      STD
// =============================================

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdexcept>

#include <iostream>
#include <sstream>
#include <fstream>
#include <csignal>

#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <functional>
#include <algorithm>
#include <errno.h>
#include <unistd.h> //sleep
#include <omp.h>
#include <mutex>

#include <time.h>
#include <ctime>

#include <limits.h>
#include <math.h>

// =============================================
//                      CUDA
// =============================================

#include <cuda.h>
#include <cuda_runtime.h>

// =============================================
//                      PCL
// =============================================

//#include "pcl/pcl_config.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/boost.h>

#include <pcl/common/geometry.h>
#include <pcl/common/common.h> //getMinMax3D
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h> //transform
#include <pcl/common/centroid.h>
#include <pcl/common/time.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/random_sample.h>

#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>

#include <pcl/octree/octree.h>
#include <pcl/kdtree/kdtree.h>

#include <pcl_ros/point_cloud.h>

// =============================================
//                      ROS
// =============================================

#include <ros/ros.h>
#include <ros/package.h>
#include <rosbag/bag.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/PointCloud2.h>

// =============================================
//                   Others
// =============================================

using namespace std;
using namespace pcl;

#include "UserDefine.h"

#endif /* ALL_HEADER_H_ */
