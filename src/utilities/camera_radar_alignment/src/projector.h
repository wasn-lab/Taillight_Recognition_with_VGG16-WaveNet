#ifndef PROJECTOR3_H
#define PROJECTOR3_H

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
class Projector
{
private:
  cv::Mat projectionMatrix;
  cv::Mat cameraExtrinsicMat;
  cv::Mat rotationMat;
  cv::Mat rotationVec;
  cv::Mat translationVec;
  cv::Mat cameraMat;
  cv::Mat distCoeff;
  cv::Mat radarMat;
  cv::Size ImageSize;
  void readCameraParameters(const char* yml_filename);

public:
  void init();
  std::vector<double> calculateCameraAngle(double h_camera, double x_p, double y_p, double x_cw, double y_cw, double z_cw, bool debug);
  std::vector<int> calculatePixel(double camera_alpha, double camera_beta, double h_camera, double x_cw, double y_cw, double z_cw, bool debug);
  std::vector<double> calculateRadarAngle(double camera_alpha, double camera_beta, double h_camera, double h_r, double h_o, double x_p, double y_p, double x_r, double y_r, double L_x, double L_y, bool debug);
  std::vector<int> project(double camera_alpha, double camera_beta, double h_camera, double radar_alpha, double radar_beta, double h_r, double h_o, double x_r, double y_r, double L_x, double L_y, bool debug);
  std::vector<int> project(double x_r, double y_r, double L_x, double L_y);
};
#endif
