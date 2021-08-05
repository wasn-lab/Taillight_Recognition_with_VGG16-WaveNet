#include "projector3.h"
#include "GlobalVariable.h"
#include <cstring>
#include <cmath>
#include <opencv2/opencv.hpp>

void Projector3::init(/*int camera_id*/)
{
  char* file_name = (char*)"b1_v3_front_bottom_60.yml";
  readCameraParameters(file_name);

  camera_mat_fix = cv::Mat::zeros(3, 3, cv::DataType<double>::type);
  rotation_vec_fix = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
  translation_vec_fix = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
}

std::vector<int> Projector3::project(float x, float y, float z)
{
  std::vector<int> result(2);
  if (!projection_matrix.empty())
  {
    std::vector<cv::Point3d> object_point;
    object_point.emplace_back(cv::Point3d((double)x, (double)y, (double)z));
    std::vector<cv::Point2d> image_point;
    cv::projectPoints(object_point, rotation_vec_fix, translation_vec_fix, camera_mat_fix, dist_Coeff, image_point);
    result[0] = image_point[0].x;
    result[1] = image_point[0].y;
  }
  else
  {
    std::cerr << " Projection Matrix is empty." << std::endl;
  }
  return result;
}

void Projector3::setprojectionMat(double yaw, double pitch, double roll, double tx, double ty, double tz)
{
  rotation_vec_fix.at<double>(0, 0) = rotation_vec.at<double>(0, 0) + (yaw * M_PI / 180);
  rotation_vec_fix.at<double>(1, 0) = rotation_vec.at<double>(1, 0) + (pitch * M_PI / 180);
  rotation_vec_fix.at<double>(2, 0) = rotation_vec.at<double>(2, 0) + (roll * M_PI / 180);

  translation_vec_fix.at<double>(0, 0) = translation_vec.at<double>(0, 0) + (tx / 100);
  translation_vec_fix.at<double>(1, 0) = translation_vec.at<double>(1, 0) + (ty / 100);
  translation_vec_fix.at<double>(2, 0) = translation_vec.at<double>(2, 0) + (tz / 100);

  cv::Mat translation_vec_temp = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
  double x = -translation_vec_fix.at<double>(2, 0);  // y
  double y = translation_vec_fix.at<double>(0, 0);   // z
  double z = translation_vec_fix.at<double>(1, 0);   //-x

  translation_vec_temp.at<double>(0, 0) = x;
  translation_vec_temp.at<double>(1, 0) = y;
  translation_vec_temp.at<double>(2, 0) = z;

  rotation_mat_fix = cv::Mat::zeros(3, 3, cv::DataType<double>::type);
  cv::Rodrigues(rotation_vec_fix, rotation_mat_fix);
  cv::hconcat(rotation_mat_fix.t(), translation_vec_temp, projection_matrix);
  std::cout << "Camera Extrinsic Matrix (autoware)= " << std::endl << " " << projection_matrix << std::endl << std::endl;
  cv::hconcat(rotation_mat_fix, translation_vec_fix, projection_matrix);
  std::cout << "Camera Extrinsic Matrix = " << std::endl << " " << projection_matrix << std::endl << std::endl;
  projection_matrix = camera_mat_fix * projection_matrix;
  std::cout << "Projection Matrix = " << std::endl << " " << projection_matrix << std::endl << std::endl;
}

void Projector3::setcameraMat(double fx, double fy, double cx, double cy)
{
  camera_mat_fix.at<double>(0, 0) = camera_mat.at<double>(0, 0) + fx;
  camera_mat_fix.at<double>(1, 1) = camera_mat.at<double>(1, 1) + fy;
  camera_mat_fix.at<double>(0, 2) = camera_mat.at<double>(0, 2) + cx;
  camera_mat_fix.at<double>(1, 2) = camera_mat.at<double>(1, 2) + cy;
  std::cout << "Camera Matrix = " << std::endl << " " << camera_mat_fix << std::endl << std::endl;
}

bool Projector3::outOfFov(float x, float y, float z)
{
  if (!projection_matrix.empty())
  {  
    cv::Mat point = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    point.at<double>(0, 0) = x;
    point.at<double>(1, 0) = y;
    point.at<double>(2, 0) = z;
    cv::Mat r_m = rotation_mat_fix(cv::Range(0, 3), cv::Range(0, 3));
    cv::Mat t_v = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    t_v.at<double>(0, 0) = translation_vec_fix.at<double>(0, 0);     
    t_v.at<double>(1, 0) = translation_vec_fix.at<double>(1, 0);
    t_v.at<double>(2, 0) = translation_vec_fix.at<double>(2, 0);
    point = r_m * point + t_v;
    if (point.at<double>(2, 0) < 0)
    {
      return true;
    }
    double tan = point.at<double>(0, 0) / point.at<double>(2, 0);
    double angle = atan(tan) * 180 / M_PI;
    if (angle > 45 || angle < -45)
    {
      return true;
    }
  }
  return false;
}

void Projector3::readCameraParameters(const char* yml_filename)
{
  cv::FileStorage fs;
  fs.open(yml_filename, cv::FileStorage::READ);
  if (!fs.isOpened())
  {
    std::cerr << " Failed to open " << yml_filename << std::endl;
    exit(EXIT_FAILURE);
  }
  fs["CameraExtrinsicMat"] >> camera_extrinsic_mat;
  rotation_mat = camera_extrinsic_mat(cv::Range(0, 3), cv::Range(0, 3)).t();
  cv::Rodrigues(rotation_mat, rotation_vec);
  // autoware
  double x = camera_extrinsic_mat.at<double>(1, 3);   // y
  double y = camera_extrinsic_mat.at<double>(2, 3);   // z
  double z = -camera_extrinsic_mat.at<double>(0, 3);  //-x

  translation_vec = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
  translation_vec.at<double>(0, 0) = x;
  translation_vec.at<double>(1, 0) = y;
  translation_vec.at<double>(2, 0) = z;
  fs["CameraMat"] >> camera_mat;
  fs["DistCoeff"] >> dist_Coeff;
  fs["ImageSize"] >> image_size;
  cv::hconcat(rotation_mat, translation_vec, projection_matrix);
  projection_matrix = camera_mat * projection_matrix;

  // Debug: Camera Parameters

  std::cout << std::endl << " ====== Camera Parameters ====== " << std::endl;
  std::cout << "Projection Matrix = " << std::endl << " " << projection_matrix << std::endl << std::endl;
  std::cout << "Camera Extrinsic Matrix = " << std::endl << " " << camera_extrinsic_mat << std::endl << std::endl;
  std::cout << "Rotarion Matrix = " << std::endl << " " << rotation_mat << std::endl << std::endl;
  std::cout << "rotationVec = " << std::endl << " " << rotation_vec << std::endl << std::endl;
  std::cout << "translationVec = " << std::endl << " " << translation_vec << std::endl << std::endl;
  std::cout << "Camera Matrix = " << std::endl << " " << camera_mat << std::endl << std::endl;
  std::cout << "Distortion Coefficients = " << std::endl << " " << dist_Coeff << std::endl << std::endl;
  std::cout << " =============================== " << std::endl;

  fs.release();
}
