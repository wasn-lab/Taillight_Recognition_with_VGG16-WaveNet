#ifndef PROJECTOR3_H
#define PROJECTOR3_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
class Projector3
{
private:
  cv::Mat projectionMatrix;
  cv::Mat cameraExtrinsicMat;
  cv::Mat rotarionMat;
  cv::Mat rotarionVec;
  cv::Mat translationVec;
  cv::Mat cameraMat;
  cv::Mat distCoeff;
  cv::Size ImageSize;
  cv::Mat coverage_mat;
  float min_x;
  float max_x;
  float min_y;
  float max_y;  
  void readCameraParameters(const char* yml_filename);

public:
  void init(int camera_id);
  std::vector<int> project(float x, float y, float z);
};
#endif
