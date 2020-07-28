#include "projector3.h"
#include <opencv2/opencv.hpp>
#include "car_model.h"
#include "camera_params.h"
#include <camera_utils_defs.h>
#include <cstring>

void Projector3::init(int camera_id)
{
#if CAR_MODEL_IS_B1_V2
  switch (camera_id)
  {
    case camera::id::front_bottom_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/b1_v2_front_bottom_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::front_top_far_30:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/b1_v2_front_top_far_30.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::right_back_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/b1_v2_right_back_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::left_back_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/b1_v2_left_back_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    default:
      std::cerr << " No match camera id, init failed." << std::endl;
      break;
  }
#endif
}

std::vector<int> Projector3::project(float x, float y, float z)
{
  std::vector<int> result(2, -1);
  if (!outOfFov(x,y,z))
  {
    if (!projectionMatrix.empty())
    {
      std::vector<cv::Point3d> object_point;
      object_point.emplace_back(cv::Point3d((double)x, (double)y, (double)z));
      std::vector<cv::Point2d> image_point;
      cv::projectPoints(object_point, rotationVec, translationVec, cameraMat, distCoeff, image_point);
      result[0] = image_point[0].x;
      result[1] = image_point[0].y;
    }
    else
    {
      std::cerr << " Projection Matrix is empty." << std::endl;
    }
  }
  return result;
}

bool Projector3::outOfFov(float x, float y, float z)
{
  if (!projectionMatrix.empty())
  {  
    cv::Mat point = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    point.at<double>(0, 0) = x;
    point.at<double>(1, 0) = y;
    point.at<double>(2, 0) = z;
    cv::Mat rM = rotationMat(cv::Range(0, 3), cv::Range(0, 3));
    cv::Mat tV = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    tV.at<double>(0, 0) = translationVec.at<double>(0, 0);     
    tV.at<double>(1, 0) = translationVec.at<double>(1, 0);
    tV.at<double>(2, 0) = translationVec.at<double>(2, 0);
    point = rM * point + tV;
    if (point.at<double>(2, 0) < 0)
      return true;
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
  // cv::Mat R = angleToRotation(M_PI_2, -M_PI_2, 0.0f);
  // rotationMat = rotationMat*R;
  cv::Rodrigues(rotationMat, rotationVec);
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
  cv::hconcat(rotationMat, translationVec, projectionMatrix);
  projectionMatrix = cameraMat * projectionMatrix;

  fs["CoverageMat"] >> coverage_mat;
  if (!coverage_mat.empty())
  {
    min_x = coverage_mat.at<float>(0, 0);
    max_x = coverage_mat.at<float>(0, 1);
    min_y = coverage_mat.at<float>(1, 0);
    max_y = coverage_mat.at<float>(1, 1);
    // std::cout << "min_x: " << min_x << ", max_x: " << max_x << std::endl;
    // std::cout << "min_y: " << min_y << ", max_y: " << max_y << std::endl;
  }

  // Debug: Camera Parameters
  /*
  std::cout << std::endl << " ====== Camera Parameters ====== " << std::endl;
  std::cout << "Projection Matrix = " << std::endl << " " << projectionMatrix << std::endl << std::endl;
  std::cout << "Camera Extrinsic Matrix = " << std::endl << " " << cameraExtrinsicMat << std::endl << std::endl;
  std::cout << "Rotation Matrix = " << std::endl << " " << rotationMat << std::endl << std::endl;
  std::cout << "rotationVec = " << std::endl << " " << rotationVec << std::endl << std::endl;
  std::cout << "translationVec = " << std::endl << " " << translationVec << std::endl << std::endl;
  std::cout << "Camera Matrix = " << std::endl << " " << cameraMat << std::endl << std::endl;
  std::cout << "Distortion Coefficients = " << std::endl << " " << distCoeff << std::endl << std::endl;
  std::cout << " =============================== " << std::endl;
  */
  fs.release();
}
