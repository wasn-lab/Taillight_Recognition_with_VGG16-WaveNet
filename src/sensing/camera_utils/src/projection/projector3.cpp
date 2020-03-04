#include "projector3.h"
#include <opencv2/opencv.hpp>
#include <string.h>
#include <camera_utils_defs.h>

void Projector3::init(const char* camera_topic_name)
{
  if(strcmp(camera_topic_name,"/cam/F_center") == 0)
  {
    char* fileName = (char *)"test_0225_F_center.yml";
    char* filePath = new char[std::strlen(CAMERA_UTILS_DATA_DIR) + std::strlen(fileName) + 1];
    std::strcpy(filePath,test);
    std::strcat(filePath,fileName);
    readCameraParameters(filePath);
  }
  else
  {
    std::cerr << " No match topic name, init failed."<< std::endl;
  }
}

std::vector<int> Projector3::project(double x, double y, double z)
{
  std::vector<int> result(2);
  if(!projectionMatrix.empty())
  {
    std::vector<cv::Point3d> objectPoint;
    objectPoint.push_back(cv::Point3d(x, y, z));
    std::vector<cv::Point2d> imagePoint;
    cv::projectPoints(objectPoint, rotarionVec, translationVec, cameraMat, distCoeff, imagePoint);
    result[0] = imagePoint[0].x;
    result[1] = imagePoint[0].y;
  }
  else
  {
    std::cerr << " Projection Matrix is empty."<< std::endl;
  }
  return result;
}

void Projector3::readCameraParameters(const char* yml_filename )
{
  cv::FileStorage fs;
  fs.open(yml_filename, cv::FileStorage::READ);
  if( !fs.isOpened() ){
    std::cerr << " Failed to open " << yml_filename << std::endl;
    exit(EXIT_FAILURE);
  }
  fs["CameraExtrinsicMat"] >> cameraExtrinsicMat;
  rotarionMat = cameraExtrinsicMat(cv::Range(0,3), cv::Range(0,3)).t();
  //cv::Mat R = angleToRotation(M_PI_2, -M_PI_2, 0.0f);
  //rotarionMat = rotarionMat*R;
  cv::Rodrigues(rotarionMat, rotarionVec);
  double x = cameraExtrinsicMat.at<double>(1,3);
  double y = cameraExtrinsicMat.at<double>(2,3);
  double z = -cameraExtrinsicMat.at<double>(0,3);
  translationVec = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
  translationVec.at<double>(0,0) = x;
  translationVec.at<double>(1,0) = y;
  translationVec.at<double>(2,0) = z;
  fs["CameraMat"] >> cameraMat;
  fs["DistCoeff"] >> distCoeff; 
  fs["ImageSize"] >> ImageSize;
  cv::hconcat(rotarionMat,translationVec,projectionMatrix);
  projectionMatrix = cameraMat*projectionMatrix;
  

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
