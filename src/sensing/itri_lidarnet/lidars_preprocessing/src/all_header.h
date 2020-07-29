
#ifndef ALL_HEADER_H_
#define ALL_HEADER_H_

// =============================================
//                      STD
// =============================================

#include <cstdio>

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
#include <cerrno>
#include <omp.h>
#include <mutex>

#include <ctime>

#include <climits>
#include <Eigen/Geometry>

//#include <pcap.h>

// =============================================
//                      CUDA
// =============================================

#include <cuda.h>
#include <cuda_runtime.h>

// =============================================
//                      QT
// =============================================

#include <QtCore>
#include <QApplication>
#include <QMainWindow>
#include <vtkRenderWindow.h>

// =============================================
//                      PCL
// =============================================

//#include "pcl/pcl_config.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/boost.h>

#include <pcl/common/geometry.h>
#include <pcl/common/common.h>  //getMinMax3D
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>  //transform
#include <pcl/common/centroid.h>
#include <pcl/common/time.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/crop_box.h>

#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/segmentation/approximate_progressive_morphological_filter.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/features/moment_of_inertia_estimation.h>
//#include <pcl/features/boundary.h>
//#include <pcl/features/vfh.h>
//#include <pcl/features/fpfh_omp.h>
//#include <pcl/features/fpfh.h>
//#include <pcl/features/impl/fpfh.hpp>
//#include <pcl/features/impl/3dsc.hpp>
//#include <pcl/features/feature.h>
//#include <pcl/features/3dsc.h>
//#include <pcl/features/normal_3d.h>
//#include <pcl/features/normal_3d_omp.h>
//#include <pcl/features/don.h>
//#include <pcl/features/grsd.h>

//#include <pcl/sample_consensus/method_types.h>
//#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
//#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_plane.h>
//#include <pcl/sample_consensus/sac_model_sphere.h>
//#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>

//#include <pcl/tracking/tracking.h>
//#include <pcl/tracking/particle_filter.h>
//#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>
//#include <pcl/tracking/particle_filter_omp.h>
//#include <pcl/tracking/coherence.h>
//#include <pcl/tracking/distance_coherence.h>
//#include <pcl/tracking/hsv_color_coherence.h>
//#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
//#include <pcl/tracking/nearest_pair_point_cloud_coherence.h>

#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>

#include <pcl/octree/octree.h>
#include <pcl/kdtree/kdtree.h>

#include <pcl/surface/concave_hull.h>

#include <pcl/ModelCoefficients.h>

#include <pcl/ml/svm.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

//#include <pcl/gpu/containers/device_array.h>
//#include <pcl/gpu/features/features.hpp>
//#include <pcl/gpu/surface/convex_hull.h>

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
//                      boost
// =============================================

#include <boost/asio.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>

// =============================================
//                   Others
// =============================================

using namespace std;
using namespace pcl;
using namespace pcl::console;
using namespace pcl::visualization;
using namespace boost::property_tree;
using namespace Eigen;

#include "UserDefine.h"

#define ENABLE_DEBUG_MODE false

#endif /* ALL_HEADER_H_ */
