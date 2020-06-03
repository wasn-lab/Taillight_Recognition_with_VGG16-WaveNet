#ifndef ALL_HEADER_H_
#define ALL_HEADER_H_

// =============================================
//                      STD
// =============================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>

#include <iostream>
#include <sstream>
#include <fstream>

#include <string>
#include <vector>
#include <thread>
#include <functional>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <unistd.h>  //sleep
#include <omp.h>
#include <mutex>

#include <ctime>
#include <ctime>

#include <climits>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

// =============================================
//                      CUDA
// =============================================

#include <cuda.h>
#include <cuda_runtime.h>

// =============================================
//                      PCL
// =============================================

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>

#include <pcl/io/boost.h>

#include <pcl/common/geometry.h>
#include <pcl/common/common.h>  //getMinMax3D
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>  //transform
#include <pcl/common/centroid.h>
#include <pcl/common/time.h>

#include <pcl/ModelCoefficients.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <pcl_ros/point_cloud.h>

// =============================================
//                      ROS
// =============================================

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Imu.h>

// =============================================
//                   Others
// =============================================

#include "UserDefine.h"

#define ENABLE_DEBUG_MODE false
#define ENABLE_LABEL_MODE true

using namespace std;
using namespace pcl;
using namespace pcl::console;
using namespace pcl::visualization;
using namespace Eigen;

#endif /* ALL_HEADER_H_ */
