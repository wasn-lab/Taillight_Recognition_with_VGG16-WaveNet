#ifndef PROJECTOR2_H
#define PROJECTOR2_H

#include "parameters.h"
#include "camera_params_b1.h"
#include <opencv2/opencv.hpp>
#define _USE_MATH_DEFINES

class Projector2
{
private:
  const float RADIUS = (M_PI / 180);
  //當前的相機內外參
  CalibrateParameters current_parameters_;
  //旋轉矩陣
  float r_float_array_[9];
  //平移矩陣
  float t_float_array_[3];
  //相機內參矩陣
  float k_float_array_[9];

  void multiplyMatrix(const float m[9], const float v[3], float result[3]);
  void addMatrix(const float v1[3], const float v2[3], float result[3]);
  void initMatrixT();
  void initMatrixR();
  void initMatrixK();
  void copyMatToFloatArray(cv::Mat mat, float array[9]);

  // static const int CALIBRATE_COMPLETED = F_left;

public:
  void init(camera::id camera_id);
  vector<int> project(float x, float y, float z);
};
#endif
