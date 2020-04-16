#ifndef LSHAPE_H_
#define LSHAPE_H_

#include <iostream>
#include <cstdlib>
#include <cstring>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>  //getMinMax3D
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <Eigen/Eigenvalues>

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_datatypes.h>

class LShape
{
public:
  LShape();
  virtual ~LShape();

  void setInputCloud(const pcl::PointCloud<pcl::PointXYZ> input);

  void Compute(std::vector<pcl::PointXYZ>& out_cube, pcl::PointXYZ& out_centroid, pcl::PointXYZ& out_minPoint,
               pcl::PointXYZ& out_maxPoint);

private:
  pcl::PointCloud<pcl::PointXYZ> cloud_3d;

  float sensor_height_;
  int random_points_;
  float slope_dist_thres_;
  int num_points_thres_;

  float roi_m_;
  float pic_scale_;

  void getPointsInPointcloudFrame(cv::Point2f rect_points[], std::vector<cv::Point2f>& pointcloud_frame_points,
                                  const cv::Point& offset_point);

  void updateCpFromPoints(const std::vector<cv::Point2f>& pointcloud_frame_points, double& PoseX, double& PoseY,
                          double& PoseZ);

  void toRightAngleBBox(std::vector<cv::Point2f>& pointcloud_frame_points);

  void updateDimentionAndEstimatedAngle(const std::vector<cv::Point2f>& pointcloud_frame_points, double& PoseOX,
                                        double& PoseOY, double& PoseOZ, double& PoseOW, double& DimX, double& DimY,
                                        double& DimZ, double& Yaw);
};

#endif
