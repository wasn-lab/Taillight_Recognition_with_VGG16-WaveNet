#include "projector2.h"
#include "camera_params_b1.h"
#include <chrono>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
// m: 3X3矩陣
// v: 3X1矩陣
// result = m * v

using namespace std::chrono;
using namespace cv;
using namespace camera;

void Projector2::multiplyMatrix(const float m[9], const float v[3],
                                float result[3]) {
  result[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
  result[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
  result[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

// v1: 3X1矩陣
// v2: 3X1矩陣
// result = v1 + v2
void Projector2::addMatrix(const float v1[3], const float v2[3],
                           float result[3]) {
  result[0] = v1[0] + v2[0];
  result[1] = v1[1] + v2[1];
  result[2] = v1[2] + v2[2];
}

//初始化,輸入camera id , 自動設好內外參, 目前只校正好前3個camera
void Projector2::init(int camera_id) {
  if (camera_id != front_60 & camera_id != top_front_120 &
      camera_id != left_60) {
    throw std::invalid_argument("這個相機的外參還沒校正好...");
  }
  switch (camera_id) {
  case id::front_60:
    current_parameters_ =
        CalibrateParameters(0.0, -1.34, 0.0, 0.0, -11, -0.4, 579, 304, 192,
                            "/cam/F_center", "/CamObjFrontCenter");
    break;
  case id::top_front_120:
    current_parameters_ =
        CalibrateParameters(0.0, 0.0, 0.0, 0.0, 35, -0.6, 293, 304, 192,
                            "/cam/F_top", "/CamObjFrontTop");
    break;
  case id::left_60:
    current_parameters_ =
        CalibrateParameters(0.0, -1.00, 0.0, 0.0, -13, 73, 293, 304, 192,
                            "/cam/F_left", "/CamObjFrontLeft");
    break;
  case id::right_60:
    current_parameters_ =
        CalibrateParameters(0.0, -1.00, 0.0, 0.0, 0, -80, 579, 304, 192,
                            "/cam/F_right", "/CamObjFrontRight");
    break;
  case id::top_rear_120:
    current_parameters_ =
        CalibrateParameters(0.0, -1.34, 0.0, 0.0, -11, -0.4, 579, 304, 192,
                            "/cam/B_top", "/CamObjBottomTop");
    break;
  case id::top_left_front_120:
    current_parameters_ =
        CalibrateParameters(0.0, -1.34, 0.0, 0.0, -11, -0.4, 579, 304, 192,
                            "/cam/L_front", "/CamObjLeftFront");
    break;
  case id::top_left_rear_120:
    current_parameters_ =
        CalibrateParameters(0.0, -1.34, 0.0, 0.0, -11, -0.4, 579, 304, 192,
                            "/cam/L_rear", "/CamObjLeftRear");
    break;
  case id::top_right_front_120:
    current_parameters_ =
        CalibrateParameters(0.0, -1.34, 0.0, 0.0, -11, -0.4, 579, 304, 192,
                            "/cam/R_front", "/CamObjRightFront");
    break;
  case id::top_right_rear_120:
    current_parameters_ =
        CalibrateParameters(0.0, -1.34, 0.0, 0.0, -11, -0.4, 579, 304, 192,
                            "/cam/R_rear", "/CamObjRightRear");
    break;
  }

  initMatrixT();
  initMatrixR();
  initMatrixK();
}

void Projector2::initMatrixT() {
  //相機位置相對於光達位置的X軸,Y軸,Z軸平移量
  t_float_array_[0] = current_parameters_.get_t_y();
  t_float_array_[1] = current_parameters_.get_t_z();
  t_float_array_[2] = current_parameters_.get_t_x();
}

void Projector2::initMatrixR() {
  float degree[3];
  // X軸,Y軸,Z軸的旋轉角度
  degree[0] = current_parameters_.get_degree_x();
  degree[1] = current_parameters_.get_degree_y();
  degree[2] = current_parameters_.get_degree_z();

  //將角度轉換為弧度並計算sin與cos值
 
  float cos_x = cosf(degree[0] * RADIUS);
  float sin_x = sinf(degree[0] * RADIUS);
  float cos_y = cosf(degree[1] * RADIUS);
  float sin_y = sinf(degree[1] * RADIUS);
  float cos_z = cosf(degree[2] * RADIUS);
  float sin_z = sinf(degree[2] * RADIUS);

  // X軸Y軸Z軸各自的旋轉矩陣
  float rx_array[9] = {cos_x, -sin_x, 0, sin_x, cos_x, 0, 0, 0, 1};
  float ry_array[9] = {1, 0, 0, 0, cos_y, -sin_y, 0, sin_y, cos_y};
  float rz_array[9] = {cos_z, 0, sin_z, 0, 1, 0, -sin_z, 0, cos_z};

  // X軸Y軸Z軸各自的旋轉矩陣相乘得到整體旋轉矩陣
  cv::Mat rx_mat = cv::Mat(3, 3, CV_32FC1, rx_array);
  cv::Mat ry_mat = cv::Mat(3, 3, CV_32FC1, ry_array);
  cv::Mat rz_mat = cv::Mat(3, 3, CV_32FC1, rz_array);
  cv::Mat r_mat = (rx_mat * ry_mat) * rz_mat;

  // copy到R陣列
  copyMatToFloatArray(r_mat, r_float_array_);
}

void Projector2::copyMatToFloatArray(cv::Mat mat, float array[9]) {
  std::vector<float> temp_vector(9);
  temp_vector.assign((float *)mat.data, (float *)mat.data + mat.total());
  std::copy(temp_vector.begin(), temp_vector.end(), array);
}

void Projector2::initMatrixK() {
  //焦距(理想針孔照像機 X軸焦距=Y軸焦距）
  k_float_array_[0] = current_parameters_.get_focal_length();
  // skew（理想針孔照像機為0)
  k_float_array_[1] = 0;
  //主點座標u(主點：光軸與成像平面的交點)
  k_float_array_[2] = current_parameters_.get_center_point_u();
  k_float_array_[3] = 0;
  //焦距
  k_float_array_[4] = current_parameters_.get_focal_length();
  //主點座標v
  k_float_array_[5] = current_parameters_.get_center_point_v();
  k_float_array_[6] = 0;
  k_float_array_[7] = 0;
  k_float_array_[8] = 1;
}

//首先把照相機的3D座標系與光達的3D座標系對齊(step1,step2)
//再透過針孔成像原理把3D點投影到位於焦距上的平面(step3,step4)
vector<int> Projector2::project(float x, float y, float z) {
  //開始時間
  // steady_clock::time_point start = steady_clock::now();

  vector<int> result(2);
  // lidar為原點座標
  float lidar_xyz_float_[3];
  // lidar旋轉後座標
  float temp_float_[3];
  // camera為原點座標乘上內參後的結果
  float image_point_float_[3];
  //以camera為原點座標
  float camera_xyz_float_[3];

  lidar_xyz_float_[0] = -y;
  lidar_xyz_float_[1] = -z;
  lidar_xyz_float_[2] = x;

  // step1.lidar原點乘上旋轉矩陣R,結果存放在temp_float_
  multiplyMatrix(r_float_array_, lidar_xyz_float_, temp_float_);
  // step2.temp_float_再加上平移矩陣T,結果存放在camera_xyz_float_
  addMatrix(t_float_array_, temp_float_, camera_xyz_float_);
  // step3.camera_xyz_float_再乘上內參K,結果存放在image_point_float_
  multiplyMatrix(k_float_array_, camera_xyz_float_, image_point_float_);
  // step4.除以X
  if (image_point_float_[2] == 0.00) {
    result[0] = 0;
    result[1] = 0;
  } else {
    result[0] = image_point_float_[0] / image_point_float_[2];
    result[1] = image_point_float_[1] / image_point_float_[2];
  }
  //結束時間
  // steady_clock::time_point end = steady_clock::now();
  //執行時間
  // std::cout << "Elapsed time: " << duration_cast<microseconds> (end -
  // start).count() << " 微秒" << std::endl;
  return result;
}
