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
  void readCameraParameters(const char* yml_filename );

public:
  void init(const char* camera_topic_name);
  std::vector<int> project(double x, double y, double z);
};
#endif
