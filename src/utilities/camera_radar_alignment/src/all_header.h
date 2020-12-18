/*
 * all_header.h
 *
 *  Created on: Dec 14, 2016
 *      Author: root
 */

#ifndef ALL_HEADER_H_
#define ALL_HEADER_H_

//#include "pcl/pcl_config.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/*
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>

#include <pcl/io/boost.h>
#include <pcl/io/pcd_io.h>

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>  //getMinMax3D
#include <pcl/common/common_headers.h>
#include <pcl/common/geometry.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>  //transform

#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/uniform_sampling.h>

#include <pcl/segmentation/approximate_progressive_morphological_filter.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/3dsc.h>
#include <pcl/features/boundary.h>
#include <pcl/features/don.h>
#include <pcl/features/feature.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/grsd.h>
#include <pcl/features/impl/3dsc.hpp>
#include <pcl/features/impl/fpfh.hpp>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/vfh.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
*/
//#include <pcl/tracking/tracking.h>
//#include <pcl/tracking/particle_filter.h>
//#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>
//#include <pcl/tracking/particle_filter_omp.h>
//#include <pcl/tracking/coherence.h>
//#include <pcl/tracking/distance_coherence.h>
//#include <pcl/tracking/hsv_color_coherence.h>
//#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
//#include <pcl/tracking/nearest_pair_point_cloud_coherence.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <pcl/search/kdtree.h>
#include <pcl/search/organized.h>

#include <pcl/kdtree/kdtree.h>
#include <pcl/octree/octree.h>

#include <pcl/surface/concave_hull.h>

#include <pcl/ModelCoefficients.h>

#include <pcl/ml/svm.h>

//#include <pcl/gpu/containers/device_array.h>
//#include <pcl/gpu/features/features.hpp>
//#include <pcl/gpu/surface/convex_hull.h>

#include <boost/asio.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include <QApplication>
#include <QMainWindow>
#include <QtCore>
#include <vtkRenderWindow.h>

#include <linux/can.h>
#include <linux/can/error.h>
#include <linux/can/raw.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <arpa/inet.h>
#include <ctime>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <omp.h>
#include <pcap.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>  //sleep
#include <vector>

#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
//#include <rosbag/bag.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Header.h>

using namespace std;
using namespace pcl;
using namespace pcl::console;
using namespace pcl::visualization;
// using namespace pcl::tracking;

using namespace boost::property_tree;
using namespace Eigen;

// user define

#define D2R (M_PI / 180.0)
#define PATH_RAW_DATA "../../raw_data/"

// user define
struct CLUSTER_INFO
{
  // Cluster
  PointCloud<PointXYZ> cloud;
  float hull_vol;
  float GRSD21[21];

  PointXYZ min;
  PointXYZ max;
  PointXYZ center;
  float dis_max_min;  // size of vehicle
  float dis_center_origin;
  float angle_from_x_axis;  // when z = 0
  float dx;
  float dy;
  float dz;

  Eigen::Matrix3f covariance;
  vector<PointXYZ> obb_vertex;  // oriented bounding box
  PointXYZ obb_center;
  float obb_dx;
  float obb_dy;
  float obb_dz;
  float obb_orient;  // when z = 0

  // Tracking
  bool found_num;
  int tracking_id;
  PointXYZ track_last_center;
  PointXYZ predict_next_center;
  PointXYZ velocity;  // use point to represent velocity x,y,z

  // Classification
  int cluster_tag;  // tag =0 (not key object) tag =1 (unknown object) tag =2
                    // (pedestrian) tag =3 (motorcycle)  tag =4 (car)  tag =5
                    // (bus)
  int confidence;

  // Output
  PointXY to_2d_PointWL[2];  // to_2d_PointWL[0] --> 2d image coordinate x,y
                             // to_2d_PointWL[1] --> x = width y = height
  PointXY to_2d_points[4];   // to_2d_points[0]~[4] --> 2d image coordinate x,y
};
typedef struct CLUSTER_INFO CLUSTER_INFO;

#endif /* ALL_HEADER_H_ */
