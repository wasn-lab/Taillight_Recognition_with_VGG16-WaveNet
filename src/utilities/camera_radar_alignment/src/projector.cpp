#include "projector.h"
#include "GlobalVariable.h"
#include <cstring>
#include <cmath>
#include <opencv2/opencv.hpp>

void Projector::init()
{
  char* file_name;
  char* file_path;
  file_name = (char*)"b1_v2_front_bottom_60.yml";
  file_path = new char[std::strlen("") + std::strlen(file_name) + 1];
  // filePath = new char[std::strlen(file_name) + 1];
  std::strcpy(file_path, "");
  std::strcat(file_path, file_name);
  readCameraParameters(file_path);
}

std::vector<double> Projector::calculateCameraAngle(double h_camera, double x_p, double y_p, double x_cw, double y_cw, double z_cw)
{
  std::vector<double> result(2);
  for (int i = -900; i < 901; i++)
  {
    for (int j = -900; j < 901; j++)
    {
      double camera_alpha = ((double)i / 10.0) / 180.0 * M_PI;
      double camera_beta = ((double)j / 10.0) / 180.0 * M_PI;
      std::vector<int> pixel = calculatePixel(camera_alpha, camera_beta, h_camera, x_cw, y_cw, z_cw);

      if (pixel[0] == x_p && pixel[1] == y_p) 
      {
        result[0] = camera_alpha;
        result[1] = camera_beta;
        std::cout << "find angle, pitch: " << camera_alpha << ", yaw: " << camera_beta << std::endl;
        break;
      }
      if ( abs(pixel[0] - x_p) < 2 && abs(pixel[1] - y_p) < 2) 
      {
        result[0] = camera_alpha;
        result[1] = camera_beta;
        std::cout << "find near angle, pitch: " << camera_alpha << ", yaw: " << camera_beta << std::endl;
      }
    }
  }
  return result;
}

std::vector<int> Projector::calculatePixel(double camera_alpha, double camera_beta, double h_camera, double x_cw, double y_cw, double z_cw)
{
  double x_c, y_c, z_c, x_c_yaw, z_c_yaw, r_image_x, r_image_y;
  double f_x = cameraMat.at<double>(0, 0);
  double f_y = cameraMat.at<double>(1, 1);
  double c_x = cameraMat.at<double>(0, 2);
  double c_y = cameraMat.at<double>(1, 2);
  std::vector<int> result(2);
  //pitch rotation
  x_c = x_cw;  
  y_c = -1 * y_cw * sin(camera_alpha) - z_cw * cos(camera_alpha) + h_camera * cos(camera_alpha);
  z_c = y_cw * cos(camera_alpha) - z_cw * sin(camera_alpha) + h_camera * sin(camera_alpha);
  //yaw rotation
  x_c_yaw = z_c * sin(camera_beta) + x_c * cos(camera_beta);
  z_c_yaw = z_c * cos(camera_beta) - x_c * sin(camera_beta);
  //pixel transform
  r_image_x = (x_c_yaw / z_c_yaw) * f_x + c_x;
  r_image_y = (y_c / z_c_yaw) * f_y + c_y;
  result[0] = (int)r_image_x;
  result[1] = (int)r_image_y;
  
  return result;
}

std::vector<double> Projector::calculateRadarAngle(double camera_alpha, double camera_beta, double h_camera, double h_r, double x_p, double y_p, double x_r, double y_r, double L_x, double L_y)
{
  std::vector<double> result(2);
  for (int i = -900; i < 901; i++)
  {
    for (int j = -900; j < 901; j++)
    {
      double radar_alpha = ((double)i / 10.0) / 180.0 * M_PI;
      double radar_beta = ((double)j / 10.0) / 180.0 * M_PI;
      std::vector<int> pixel = project(camera_alpha, camera_beta, h_camera, radar_alpha, radar_beta, h_r, x_r, y_r, L_x, L_y);
      if (pixel[0] == x_p && pixel[1] == y_p) 
      {
        result[0] = radar_alpha;
        result[1] = radar_beta;
        std::cout << "find angle, pitch: " << camera_alpha << ", yaw: " << camera_beta << std::endl;
        break;
      }
      if ( abs(pixel[0] - x_p) < 2 && abs(pixel[1] - y_p) < 2) 
      {
        result[0] = radar_alpha;
        result[1] = radar_beta;
        std::cout << "find near angle, pitch: " << camera_alpha << ", yaw: " << camera_beta << std::endl;
      }
    }
  }
  return result;
}


std::vector<int> Projector::project(double camera_alpha, double camera_beta, double h_camera, double radar_alpha, double radar_beta, double h_r, double x_r, double y_r, double L_x, double L_y)
{
  double x_rw, y_rw, z_rw, z_r, x_rw_yaw, y_rw_yaw, x_c, y_c, z_c, r_image_x, r_image_y;
  double f_x = cameraMat.at<double>(0, 0);
  double f_y = cameraMat.at<double>(1, 1);
  double c_x = cameraMat.at<double>(0, 2);
  double c_y = cameraMat.at<double>(1, 2);
  //pitch rotation
  z_r = h_r/cos(radar_alpha);
  x_rw = x_r;
  y_rw = y_r * cos(radar_alpha) + z_r * sin(radar_alpha) + h_r * sin(radar_alpha);
  z_rw = -y_r * sin(radar_alpha) + h_r;
  //yaw rotation
  x_rw_yaw = x_rw * cos(radar_beta) - y_rw * sin(radar_beta) - L_x;
  y_rw_yaw = x_rw * sin(radar_beta) - y_rw * cos(radar_beta) + L_y;
  //pixel transform
  std::vector<int> pixel = calculatePixel(camera_alpha, camera_beta, h_camera, x_rw_yaw, y_rw_yaw, z_rw);
  return pixel;
}

void Projector::readCameraParameters(const char* yml_filename)
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
