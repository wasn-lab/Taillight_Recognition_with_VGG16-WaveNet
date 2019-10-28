#ifndef Alignment_H_
#define Alignment_H_

// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>  //velodyne

/// Opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// cout
#include <iostream>

#define PI 3.1415926

using namespace std;

class Alignment
{
public:
  Alignment();

  void a(int a);
  void camera_alignment_module(int num, double& cosfi_min, double& cosfi_max, double& costhta_min, double& costhta_max);

  double* value_distance_array(pcl::PointCloud<pcl::PointXYZ> release_cloud, double image_x, double image_y, int num);

  double* value_distance_array_2pixel(pcl::PointCloud<pcl::PointXYZ> release_cloud, double image_x, double image_y,
                                      double image_x2, double image_y2, int num);
  // int num;
  // double cosfi_min;
  // double cosfi_max;

  // double costhta_min;
  // double costhta_max;

private:
  cv::Mat alig_CameraExtrinsicMat;
  cv::Mat alig_DistCoeff_;
  cv::Mat alig_CameraExtrinsicMat_tr;
  cv::Mat alig_CameraMat_;
};

#endif