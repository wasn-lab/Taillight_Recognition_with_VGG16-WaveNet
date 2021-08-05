#ifndef PROJECTOR3_H
#define PROJECTOR3_H

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
class Projector3
{
private:
  cv::Mat projection_matrix;
  cv::Mat camera_extrinsic_mat;
  cv::Mat rotation_mat;
  cv::Mat rotation_mat_fix;
  cv::Mat rotation_vec;
  cv::Mat translation_vec;
  cv::Mat rotation_vec_fix;
  cv::Mat translation_vec_fix;
  cv::Mat camera_mat;
  cv::Mat camera_mat_fix;
  cv::Mat dist_Coeff;
  cv::Size image_size;
  void readCameraParameters(const char* yml_filename);

public:
  void init(/*int camera_id*/);
  void setprojectionMat(double yaw, double pitch, double roll, double tx, double ty, double tz);
  void setcameraMat(double fx, double fy, double cx, double cy);
  std::vector<int> project(float x, float y, float z);
  bool outOfFov(float x, float y, float z);
};
#endif
