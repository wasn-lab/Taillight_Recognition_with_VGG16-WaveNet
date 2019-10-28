/*
   CREATER: ICL U300
   DATE: July, 2019
 */

#ifndef __PARKNET_ALIGNMENT_NODE_H__
#define __PARKNET_ALIGNMENT_NODE_H__

#include <boost/shared_ptr.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include "pcl_ros/point_cloud.h"
#include "opencv2/core/mat.hpp"
#include "alignment_params.h"

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
#define INIT_COORDINATE_VALUE (0)

class ParknetAlignmentNode
{
private:
  // vars:
  PointCloud pcd_;
  ros::NodeHandle node_handle_;
  ros::Subscriber pcd_subscribers_;
  cv::Point3d** spatial_points_;
  int num_pcd_received_;

  // functions:
  void subscribe_and_advertise_topics();
  void pcd_callback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud);
  void dump_distance_in_json() const;

public:
  const int image_width_, image_height_;

  ParknetAlignmentNode();
  ParknetAlignmentNode(const int width, const int height);
  ~ParknetAlignmentNode();

  bool is_valid_image_point(const int row, const int col) const;
  bool spatial_point_is_valid(const cv::Point3d& point) const;
  bool spatial_point_is_valid(const int row, const int col) const;
  bool search_valid_neighbor(const int row, const int col, cv::Point* valid_neighbor) const;
  int set_spatial_point(const int row, const int col, const cv::Point3d& in_point);
  cv::Point3d get_spatial_point(const int row, const int col) const;
  void dump_dist_mapping() const;
  void approx_nearest_points_if_necessary();

  // ROS related:
  void run(int argc, char* argv[]);
};

#endif  // __PARKNET_ALIGNMENT_NODE_H__
