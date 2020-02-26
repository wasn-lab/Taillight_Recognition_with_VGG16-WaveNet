#ifndef ALIGNMENT_OFFLINE_H_
#define ALIGNMENT_OFFLINE_H_

#define INIT_COORDINATE_VALUE (0)

// ROS message
#include "ros/ros.h"
#include <std_msgs/String.h>
#include <std_msgs/Header.h>
#include "sensor_msgs/PointCloud2.h"

#include "camera_params.h"  // include camera topic name
#include <msgs/BoxPoint.h>
#include <msgs/PointXYZ.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/mat.hpp"
#include "projection/projector2.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <jsoncpp/json/json.h>

class AlignmentOff
{
private:
  int carId = 1;
  Projector2 pj;

public:
  void init(int carId);
  vector<int> run(float x, float y, float z);
  vector<int> out;
  int imgW, imgH;
  float groundUpBound, groundLowBound;
  cv::Point3d** spatial_points_;

  bool is_valid_image_point(const int row, const int col) const;
  bool spatial_point_is_valid(const cv::Point3d& point) const;
  bool spatial_point_is_valid(const int row, const int col) const;
  void approx_nearest_points_if_necessary();
  bool search_valid_neighbor(const int row, const int col, cv::Point* valid_neighbor) const;
  void dump_distance_in_json() const;
  std::string jsonize_spatial_points(cv::Point3d** spatial_points_, int rows, int cols) const;
};

#endif /*ALIGNMENT_OFFLINE_H_*/
