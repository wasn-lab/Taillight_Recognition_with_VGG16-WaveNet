#include "projector3.h"
#include <opencv2/opencv.hpp>
#include "car_model.h"
#include "camera_params.h"
#include <camera_utils_defs.h>
#include <cstring>

void Projector3::init(int camera_id)
{
#if CAR_MODEL_IS_B1_V2
  camera_id_ = camera_id;
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

    case camera::id::front_top_close_120:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/b1_v2_front_top_close_120.yml");
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

    case camera::id::right_front_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/b1_v2_right_front_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::left_front_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/b1_v2_left_front_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    default:
      std::cerr << " No match camera id, init failed." << std::endl;
      break;
  }

#elif CAR_MODEL_IS_B1_V3
  camera_id_ = camera_id;
  switch (camera_id)
  {
    case camera::id::front_bottom_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/b1_v3_front_bottom_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::front_top_far_30:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/b1_v3_front_top_far_30.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::front_top_close_120:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/b1_v3_front_top_close_120.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::right_back_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/b1_v3_right_back_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::left_back_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/b1_v3_left_back_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::right_front_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/b1_v3_right_front_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::left_front_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/b1_v3_left_front_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    default:
      std::cerr << " No match camera id, init failed." << std::endl;
      break;
  }

#elif CAR_MODEL_IS_C1
  camera_id_ = camera_id;
  switch (camera_id)
  {
    case camera::id::front_bottom_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c1_front_bottom_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::front_top_far_30:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c1_front_top_far_30.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::front_top_close_120:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c1_front_top_close_120.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::right_back_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c1_right_back_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::left_back_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c1_left_back_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::right_front_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c1_right_front_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::left_front_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c1_left_front_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    default:
      std::cerr << " No match camera id, init failed." << std::endl;
      break;
  }
#elif CAR_MODEL_IS_C2
  camera_id_ = camera_id;
  switch (camera_id)
  {
    case camera::id::front_bottom_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c2_front_bottom_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::front_top_far_30:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c2_front_top_far_30.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::front_top_close_120:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c2_front_top_close_120.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::right_back_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c2_right_back_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::left_back_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c2_left_back_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::right_front_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c2_right_front_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::left_front_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c2_left_front_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    default:
      std::cerr << " No match camera id, init failed." << std::endl;
      break;
  }
#elif CAR_MODEL_IS_C3
  camera_id_ = camera_id;
  switch (camera_id)
  {
    case camera::id::front_bottom_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c3_front_bottom_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::front_top_far_30:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c3_front_top_far_30.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::front_top_close_120:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c3_front_top_close_120.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::right_back_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c3_right_back_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::left_back_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c3_left_back_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::right_front_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c3_right_front_60.yml");
      readCameraParameters(file_path.c_str());
    }
    break;

    case camera::id::left_front_60:
    {
      std::string file_path = std::string(CAMERA_UTILS_DATA_DIR) + std::string("/projection/c3_left_front_60.yml");
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
    {
      return true;
    }
    double tan = point.at<double>(0, 0) / point.at<double>(2, 0);
    double angle = atan(tan) * 180 / M_PI;

    double tan_h = point.at<double>(1, 0) / point.at<double>(2, 0);
    double angle_h = atan(tan_h) * 180 / M_PI;
#if CAR_MODEL_IS_B1_V2
    switch (camera_id_)
    {
      case camera::id::front_bottom_60: case camera::id::right_back_60: case camera::id::left_back_60: case camera::id::right_front_60: case camera::id::left_front_60:
      {
        if (angle > 40 || angle < -40)
        {
          return true;
        }
        if (angle_h > 40 || angle_h < -40)
        {
          return true;
        }
      }
      break;

      case camera::id::front_top_far_30:
      {
        if (angle > 25 || angle < -25)
        {
          return true;
        }
        if (angle_h > 25 || angle_h < -25)
        {
          return true;
        }
      }
      break;

      case camera::id::front_top_close_120:
      {
        if (angle > 70 || angle < -70)
        {
          return true;
        }
        if (angle_h > 70 || angle_h < -70)
        {
          return true;
        }
      }
      break;

      default:
        std::cerr << " No match camera id, init failed." << std::endl;
      break;
    }
#elif CAR_MODEL_IS_B1_V3
    switch (camera_id_)
    {
      case camera::id::front_bottom_60: case camera::id::right_back_60: case camera::id::left_back_60: case camera::id::right_front_60: case camera::id::left_front_60:
      {
        if (angle > 40 || angle < -40)
        {
          return true;
        }
        if (angle_h > 40 || angle_h < -40)
        {
          return true;
        }
      }
      break;

      case camera::id::front_top_far_30:
      {
        if (angle > 25 || angle < -25)
        {
          return true;
        }
        if (angle_h > 25 || angle_h < -25)
        {
          return true;
        }
      }
      break;

      case camera::id::front_top_close_120:
      {
        if (angle > 70 || angle < -70)
        {
          return true;
        }
        if (angle_h > 70 || angle_h < -70)
        {
          return true;
        }
      }
      break;

      default:
        std::cerr << " No match camera id, init failed." << std::endl;
      break;
    }
#elif CAR_MODEL_IS_C1
    switch (camera_id_)
    {
      case camera::id::front_bottom_60: case camera::id::right_back_60: case camera::id::left_back_60: case camera::id::right_front_60: case camera::id::left_front_60:
      {
        if (angle > 40 || angle < -40)
        {
          return true;
        }
        if (angle_h > 40 || angle_h < -40)
        {
          return true;
        }
      }
      break;

      case camera::id::front_top_far_30:
      {
        if (angle > 25 || angle < -25)
        {
          return true;
        }
        if (angle_h > 25 || angle_h < -25)
        {
          return true;
        }
      }
      break;

      case camera::id::front_top_close_120:
      {
        if (angle > 70 || angle < -70)
        {
          return true;
        }
        if (angle_h > 70 || angle_h < -70)
        {
          return true;
        }
      }
      break;

      default:
        std::cerr << " No match camera id, init failed." << std::endl;
      break;
    }
#elif CAR_MODEL_IS_C2
    switch (camera_id_)
    {
      case camera::id::front_bottom_60: case camera::id::right_back_60: case camera::id::left_back_60: case camera::id::right_front_60: case camera::id::left_front_60:
      {
        if (angle > 40 || angle < -40)
        {
          return true;
        }
        if (angle_h > 40 || angle_h < -40)
        {
          return true;
        }
      }
      break;

      case camera::id::front_top_far_30:
      {
        if (angle > 25 || angle < -25)
        {
          return true;
        }
        if (angle_h > 25 || angle_h < -25)
        {
          return true;
        }
      }
      break;

      case camera::id::front_top_close_120:
      {
        if (angle > 70 || angle < -70)
        {
          return true;
        }
        if (angle_h > 70 || angle_h < -70)
        {
          return true;
        }
      }
      break;

      default:
        std::cerr << " No match camera id, init failed." << std::endl;
      break;
    }
#elif CAR_MODEL_IS_C3
    switch (camera_id_)
    {
      case camera::id::front_bottom_60: case camera::id::right_back_60: case camera::id::left_back_60: case camera::id::right_front_60: case camera::id::left_front_60:
      {
        if (angle > 40 || angle < -40)
        {
          return true;
        }
        if (angle_h > 40 || angle_h < -40)
        {
          return true;
        }
      }
      break;

      case camera::id::front_top_far_30:
      {
        if (angle > 25 || angle < -25)
        {
          return true;
        }
        if (angle_h > 25 || angle_h < -25)
        {
          return true;
        }
      }
      break;

      case camera::id::front_top_close_120:
      {
        if (angle > 70 || angle < -70)
        {
          return true;
        }
        if (angle_h > 70 || angle_h < -70)
        {
          return true;
        }
      }
      break;

      default:
        std::cerr << " No match camera id, init failed." << std::endl;
      break;
    }
#endif
  }
  return false;
}
bool Projector3::outOfCoverage(float x, float y, float z)
{
  if(!coverage_mat.empty())
  {
    if (x < min_x || x > max_x || y < min_y || y > max_y || z > 0)
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
