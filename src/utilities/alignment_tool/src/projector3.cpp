#include "projector3.h"
#include "GlobalVariable.h"
#include <cstring>
#include <cmath>
#include <opencv2/opencv.hpp>

void Projector3::init(int camera_id)
{
  char* file_name;
  char* file_path;
  file_name = (char*)"b1_v2_left_back_60.yml";
  file_path = new char[std::strlen("") + std::strlen(file_name) + 1];
  // filePath = new char[std::strlen(file_name) + 1];
  std::strcpy(file_path, "");
  std::strcat(file_path, file_name);
  readCameraParameters(file_path);

  cameraMat_fix = cv::Mat::zeros(3, 3, cv::DataType<double>::type);
  rotationVec_fix = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
  translationVec_fix = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
}

std::vector<int> Projector3::project(float x, float y, float z)
{
  std::vector<int> result(2);
  if (!projectionMatrix.empty())
  {
    std::vector<cv::Point3d> object_point;
    object_point.emplace_back(cv::Point3d((double)x, (double)y, (double)z));
    std::vector<cv::Point2d> image_point;
    cv::projectPoints(object_point, rotationVec_fix, translationVec_fix, cameraMat_fix, distCoeff, image_point);
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
  rotationVec_fix.at<double>(0, 0) = rotationVec.at<double>(0, 0) + (yaw * M_PI / 180);
  rotationVec_fix.at<double>(1, 0) = rotationVec.at<double>(1, 0) + (pitch * M_PI / 180);
  rotationVec_fix.at<double>(2, 0) = rotationVec.at<double>(2, 0) + (roll * M_PI / 180);

  translationVec_fix.at<double>(0, 0) = translationVec.at<double>(0, 0) + (tx / 10);
  translationVec_fix.at<double>(1, 0) = translationVec.at<double>(1, 0) + (ty / 10);
  translationVec_fix.at<double>(2, 0) = translationVec.at<double>(2, 0) + (tz / 10);

  cv::Mat translationVec_temp = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
  double x = -translationVec_fix.at<double>(2, 0);  // y
  double y = translationVec_fix.at<double>(0, 0);   // z
  double z = translationVec_fix.at<double>(1, 0);   //-x

  translationVec_temp.at<double>(0, 0) = x;
  translationVec_temp.at<double>(1, 0) = y;
  translationVec_temp.at<double>(2, 0) = z;

  rotationMat_fix = cv::Mat::zeros(3, 3, cv::DataType<double>::type);
  cv::Rodrigues(rotationVec_fix, rotationMat_fix);
  cv::hconcat(rotationMat_fix.t(), translationVec_temp, projectionMatrix);
  std::cout << "Camera Extrinsic Matrix (autoware)= " << std::endl << " " << projectionMatrix << std::endl << std::endl;
  cv::hconcat(rotationMat_fix, translationVec_fix, projectionMatrix);
  std::cout << "Camera Extrinsic Matrix = " << std::endl << " " << projectionMatrix << std::endl << std::endl;
  projectionMatrix = cameraMat_fix * projectionMatrix;
  std::cout << "Projection Matrix = " << std::endl << " " << projectionMatrix << std::endl << std::endl;
}

void Projector3::setcameraMat(double fx, double fy, double cx, double cy)
{
  cameraMat_fix.at<double>(0, 0) = cameraMat.at<double>(0, 0) + fx;
  cameraMat_fix.at<double>(1, 1) = cameraMat.at<double>(1, 1) + fy;
  cameraMat_fix.at<double>(0, 2) = cameraMat.at<double>(0, 2) + cx;
  cameraMat_fix.at<double>(1, 2) = cameraMat.at<double>(1, 2) + cy;
  std::cout << "Camera Matrix = " << std::endl << " " << cameraMat_fix << std::endl << std::endl;
}

bool Projector3::outOfFov(float x, float y, float z)
{
  if (!projectionMatrix.empty())
  {  
    cv::Mat point = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    point.at<double>(0, 0) = x;
    point.at<double>(1, 0) = y;
    point.at<double>(2, 0) = z;
    cv::Mat rM = rotationMat_fix(cv::Range(0, 3), cv::Range(0, 3));
    cv::Mat tV = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    tV.at<double>(0, 0) = translationVec_fix.at<double>(0, 0);     
    tV.at<double>(1, 0) = translationVec_fix.at<double>(1, 0);
    tV.at<double>(2, 0) = translationVec_fix.at<double>(2, 0);
    point = rM * point + tV;
    if (point.at<double>(2, 0) < 0)
    {
      return true;
    }
    double tan = point.at<double>(0, 0) / point.at<double>(2, 0);
    double angle = atan(tan) * 180 / M_PI;
    if (angle > 30 || angle < -30)
      return true;
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
  fs["CameraExtrinsicMat"] >> cameraExtrinsicMat;
  rotationMat = cameraExtrinsicMat(cv::Range(0, 3), cv::Range(0, 3)).t();
  cv::Rodrigues(rotationMat, rotationVec);
  // autoware
  double x = cameraExtrinsicMat.at<double>(1, 3);   // y
  double y = cameraExtrinsicMat.at<double>(2, 3);   // z
  double z = -cameraExtrinsicMat.at<double>(0, 3);  //-x

  translationVec = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
  translationVec.at<double>(0, 0) = x;
  translationVec.at<double>(1, 0) = y;
  translationVec.at<double>(2, 0) = z;
  fs["CameraMat"] >> cameraMat;
  fs["DistCoeff"] >> distCoeff;
  fs["ImageSize"] >> ImageSize;
  cv::hconcat(rotationMat, translationVec, projectionMatrix);
  projectionMatrix = cameraMat * projectionMatrix;

  // Debug: Camera Parameters

  std::cout << std::endl << " ====== Camera Parameters ====== " << std::endl;
  std::cout << "Projection Matrix = " << std::endl << " " << projectionMatrix << std::endl << std::endl;
  std::cout << "Camera Extrinsic Matrix = " << std::endl << " " << cameraExtrinsicMat << std::endl << std::endl;
  std::cout << "Rotarion Matrix = " << std::endl << " " << rotationMat << std::endl << std::endl;
  std::cout << "rotationVec = " << std::endl << " " << rotationVec << std::endl << std::endl;
  std::cout << "translationVec = " << std::endl << " " << translationVec << std::endl << std::endl;
  std::cout << "Camera Matrix = " << std::endl << " " << cameraMat << std::endl << std::endl;
  std::cout << "Distortion Coefficients = " << std::endl << " " << distCoeff << std::endl << std::endl;
  std::cout << " =============================== " << std::endl;

  fs.release();
}
