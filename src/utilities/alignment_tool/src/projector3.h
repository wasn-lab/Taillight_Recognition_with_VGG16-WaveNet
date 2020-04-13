#ifndef PROJECTOR3_H
#define PROJECTOR3_H

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
class Projector3
{
private:
  cv::Mat projectionMatrix;
  cv::Mat cameraExtrinsicMat;
  cv::Mat rotationMat;
  cv::Mat rotationVec;
  cv::Mat translationVec;
  cv::Mat rotationVec_fix;
  cv::Mat translationVec_fix;
  cv::Mat cameraMat;
  cv::Mat distCoeff;
  cv::Size ImageSize;
  void readCameraParameters(const char* yml_filename);

public:
  void init(int camera_id);
  void setprojectionMat(double yaw, double pitch, double roll, double tx, double ty, double tz);
  std::vector<int> project(float x, float y, float z);
};
#endif
