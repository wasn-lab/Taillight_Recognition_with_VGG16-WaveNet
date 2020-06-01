#include "projector3.h"
#include <opencv2/opencv.hpp>
#include "car_model.h"
#include "camera_params.h"
#include <camera_utils_defs.h>
#include <string.h>

void Projector3::init(int camera_id)
{
  char* file_name;
  char* file_path;
#if CAR_MODEL_IS_B1_V2
  switch (camera_id)
  {
    case camera::id::front_bottom_60:
      file_name = (char*)"/fix_front_bottom_60_0310.yml";
      file_path = new char[std::strlen(CAMERA_UTILS_DATA_DIR) + std::strlen(file_name) + 1];
      std::strcpy(file_path, CAMERA_UTILS_DATA_DIR);
      std::strcat(file_path, file_name);
      readCameraParameters(file_path);
      delete file_name;
      delete [] file_path;
      break;

    case camera::id::right_back_60:
      file_name = (char*)"/fix_right_back_60_0511.yml";
      file_path = new char[std::strlen(CAMERA_UTILS_DATA_DIR) + std::strlen(file_name) + 1];
      std::strcpy(file_path, CAMERA_UTILS_DATA_DIR);
      std::strcat(file_path, file_name);
      readCameraParameters(file_path);
      delete file_name;
      delete [] file_path;
      break;

    default:
      std::cerr << " No match camera id, init failed." << std::endl;
      break;
  }
#endif
}

std::vector<int> Projector3::project(float x, float y, float z)
{
  std::vector<int> result(2);
  if (!projectionMatrix.empty())
  {
    std::vector<cv::Point3d> object_point;
    object_point.emplace_back(cv::Point3d((double)x, (double)y, (double)z));
    std::vector<cv::Point2d> image_point;
    cv::projectPoints(object_point, rotarionVec, translationVec, cameraMat, distCoeff, image_point);
    result[0] = image_point[0].x;
    result[1] = image_point[0].y;
  }
  else
  {
    std::cerr << " Projection Matrix is empty." << std::endl;
  }
  return result;
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
  rotarionMat = cameraExtrinsicMat(cv::Range(0, 3), cv::Range(0, 3)).t();
  // cv::Mat R = angleToRotation(M_PI_2, -M_PI_2, 0.0f);
  // rotarionMat = rotarionMat*R;
  cv::Rodrigues(rotarionMat, rotarionVec);
  double x = cameraExtrinsicMat.at<double>(1, 3);
  double y = cameraExtrinsicMat.at<double>(2, 3);
  double z = -cameraExtrinsicMat.at<double>(0, 3);
  translationVec = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
  translationVec.at<double>(0, 0) = x;
  translationVec.at<double>(1, 0) = y;
  translationVec.at<double>(2, 0) = z;
  fs["CameraMat"] >> cameraMat;
  fs["DistCoeff"] >> distCoeff;
  fs["ImageSize"] >> ImageSize;
  cv::hconcat(rotarionMat, translationVec, projectionMatrix);
  projectionMatrix = cameraMat * projectionMatrix;

  // Debug: Camera Parameters
  /*
  std::cout << std::endl << " ====== Camera Parameters ====== " << std::endl;
  std::cout << "Projection Matrix = " << std::endl << " " << projectionMatrix << std::endl << std::endl;
  std::cout << "Camera Extrinsic Matrix = " << std::endl << " " << cameraExtrinsicMat << std::endl << std::endl;
  std::cout << "Rotarion Matrix = " << std::endl << " " << rotarionMat << std::endl << std::endl;
  std::cout << "rotarionVec = " << std::endl << " " << rotarionVec << std::endl << std::endl;
  std::cout << "translationVec = " << std::endl << " " << translationVec << std::endl << std::endl;
  std::cout << "Camera Matrix = " << std::endl << " " << cameraMat << std::endl << std::endl;
  std::cout << "Distortion Coefficients = " << std::endl << " " << distCoeff << std::endl << std::endl;
  std::cout << " =============================== " << std::endl;
  */
  fs.release();
}
