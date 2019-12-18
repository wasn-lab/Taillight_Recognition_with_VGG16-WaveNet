#include "projector2.h"
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <chrono>
//m: 3X3矩陣 
//v: 3X1矩陣
//result = m * v 

using namespace std::chrono;
using namespace cv;
void Projector2::matrix_vector_multiply_3x3_3d(const float m[9],const float v[3],float result[3]) 
{
  result[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
  result[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
  result[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

//v1: 3X1矩陣
//v2: 3X1矩陣
//result = v1 + v2
void Projector2::vector_add_3d(const float v1[3], const float v2[3],float result[3]) 
{
  result[0] = v1[0] + v2[0];
  result[1] = v1[1] + v2[1];
  result[2] = v1[2] + v2[2];
}

//初始化,輸入camera id , 自動設好內外參, 目前只校正好前3個camera
void Projector2::init(int camera_id)
{
  if(camera_id > CALIBRATE_COMPLETED )
  {
    throw std::invalid_argument("這個相機的外參還沒校正好...");
  }
  switch(camera_id)
  {
    case F_center:
                                            //tx  , ty  , tz  , deg x, deg y
      currentParameters = CalibrateParameters(0.0, -1.34, 0.0, 0.0, -11, -0.4, 579, 304, 192, "/cam/F_center", "/CamObjFrontCenter");
      break;
    case F_top:
      currentParameters = CalibrateParameters(0.0,   0.0, 0.0, 0.0,  35, -0.6, 293, 304, 192, "/cam/F_top", "/CamObjFrontTop");
      break;
    case F_left:
      currentParameters = CalibrateParameters(0.0, -1.00, 0.0, 0.0, -13,   73, 293, 304, 192, "/cam/F_left", "/CamObjFrontLeft");
      break;
    case F_right:
      currentParameters = CalibrateParameters(0.0, -1.00, 0.0, 0.0,   0,  -80, 579, 304, 192, "/cam/F_right", "/CamObjFrontRight");
      break;
    case B_top:
      currentParameters = CalibrateParameters(0.0, -1.34, 0.0, 0.0, -11, -0.4, 579, 304, 192, "/cam/B_top", "/CamObjBottomTop");
      break;
    case L_front:
      currentParameters = CalibrateParameters(0.0, -1.34, 0.0, 0.0, -11, -0.4, 579, 304, 192, "/cam/L_front", "/CamObjLeftFront");
      break;
    case L_rear:
      currentParameters = CalibrateParameters(0.0, -1.34, 0.0, 0.0, -11, -0.4, 579, 304, 192, "/cam/L_rear", "/CamObjLeftRear");
      break;
    case R_front:
      currentParameters = CalibrateParameters(0.0, -1.34, 0.0, 0.0, -11, -0.4, 579, 304, 192, "/cam/R_front", "/CamObjRightFront");
      break;
    case R_rear:
      currentParameters = CalibrateParameters(0.0, -1.34, 0.0, 0.0, -11, -0.4, 579, 304, 192, "/cam/R_rear", "/CamObjRightRear");
      break;  
  }
  
  init_T_matrix();  
  init_R_matrix();
  init_K_matrix();
}

void Projector2::init_T_matrix()
{
  //相機位置相對於光達位置的X軸,Y軸,Z軸平移量
  T[0] = currentParameters.get_t_y();
  T[1] = currentParameters.get_t_z();
  T[2] = currentParameters.get_t_x();
}

void Projector2::init_R_matrix()
{
  float degree[3];
  //X軸,Y軸,Z軸的旋轉角度
  degree[0] = currentParameters.get_degree_x();
  degree[1] = currentParameters.get_degree_y();
  degree[2] = currentParameters.get_degree_z();

  //將角度轉換為弧度並計算sin與cos值
  float radius_float_ = (3.14 / 180);
  float cos_x_float_ = cosf(degree[0] * radius_float_);
  float sin_x_float_ = sinf(degree[0] * radius_float_);
  float cos_y_float_ = cosf(degree[1] * radius_float_);
  float sin_y_float_ = sinf(degree[1] * radius_float_);
  float cos_z_float_ = cosf(degree[2] * radius_float_);
  float sin_z_float_ = sinf(degree[2] * radius_float_);
  
  //X軸Y軸Z軸各自的旋轉矩陣  
  float rx_float_[9] = {cos_x_float_, -sin_x_float_, 0, sin_x_float_, cos_x_float_, 0, 0, 0, 1};
  float ry_float_[9] = { 1, 0, 0, 0, cos_y_float_, -sin_y_float_, 0, sin_y_float_, cos_y_float_};
  float rz_float_[9] = {cos_z_float_,  0, sin_z_float_, 0, 1, 0, -sin_z_float_, 0, cos_z_float_};
  
  //X軸Y軸Z軸各自的旋轉矩陣相乘得到整體旋轉矩陣
  vector<float> R_v;
  cv::Mat rx_mat_ = cv::Mat(3, 3, CV_32FC1, rx_float_);
  cv::Mat ry_mat_ = cv::Mat(3, 3, CV_32FC1, ry_float_);
  cv::Mat rz_mat_ = cv::Mat(3, 3, CV_32FC1, rz_float_);
  cv::Mat r_mat_ = (rx_mat_ * ry_mat_) * rz_mat_;

  //copy到R陣列
  std::vector<float> array_vector_;
  if (r_mat_.isContinuous()) {
    array_vector_.assign((float *)r_mat_.data, (float *)r_mat_.data + r_mat_.total());
  } else {
    for (int i = 0; i < r_mat_.rows; ++i) {
      array_vector_.insert(array_vector_.end(), r_mat_.ptr<float>(i), r_mat_.ptr<float>(i) + r_mat_.cols);
    }
  }
  std::copy(array_vector_.begin(), array_vector_.end(), R);
}

void Projector2::init_K_matrix()
{
  //焦距(理想針孔照像機 X軸焦距=Y軸焦距）
  K[0] = currentParameters.get_focal_length();
  //skew（理想針孔照像機為0)
  K[1] = 0;
  //主點座標u(主點：光軸與成像平面的交點)
  K[2] = currentParameters.get_center_point_u();
  K[3] = 0;
  //焦距
  K[4] = currentParameters.get_focal_length();
  //主點座標v
  K[5] = currentParameters.get_center_point_v();
  K[6] = 0;
  K[7] = 0;
  K[8] = 1;
}

//首先把照相機的3D座標系與光達的3D座標系對齊(step1,step2)
//再透過針孔成像原理把3D點投影到位於焦距上的平面(step3,step4)
vector<int> Projector2::project(float x, float y, float z)
{
  //開始時間
  //steady_clock::time_point start = steady_clock::now();
  
  std::cout << "X: " << x << " Y: "<< y << " Z: " << z << std::endl;
  vector<int> result(2);
  //lidar為原點座標
  float lidar_xyz_float_[3];
  //lidar旋轉後座標
  float temp_float_[3];
  //camera為原點座標乘上內參後的結果 
  float image_point_float_[3];
  //以camera為原點座標
  float camera_xyz_float_[3];

  lidar_xyz_float_[0] = -y;
  lidar_xyz_float_[1] = -z;
  lidar_xyz_float_[2] =  x;

  //step1.lidar原點乘上旋轉矩陣R,結果存放在temp_float_
  matrix_vector_multiply_3x3_3d(R, lidar_xyz_float_, temp_float_);
  //step2.temp_float_再加上平移矩陣T,結果存放在camera_xyz_float_
  vector_add_3d(T, temp_float_, camera_xyz_float_);
  //step3.camera_xyz_float_再乘上內參K,結果存放在image_point_float_
  matrix_vector_multiply_3x3_3d(K, camera_xyz_float_, image_point_float_);
  //step4.除以X
  if(image_point_float_[2] == 0.00){
    result[0] = 0;
    result[1] = 0;
  }else{
    result[0]= image_point_float_[0] / image_point_float_[2];
    result[1]= image_point_float_[1] / image_point_float_[2];
  }
  //結束時間
  //steady_clock::time_point end = steady_clock::now();
  //執行時間
  //std::cout << "Elapsed time: " << duration_cast<microseconds> (end - start).count() << " 微秒" << std::endl;
  return result;
}
