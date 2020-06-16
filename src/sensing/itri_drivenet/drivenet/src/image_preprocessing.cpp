#include "drivenet/image_preprocessing.h"

using namespace cv;
using namespace std;

namespace DriveNet
{
cv::Scalar CvColor::white_(255, 255, 255, 0);
cv::Scalar CvColor::blue_(255, 0, 0, 0);
cv::Scalar CvColor::green_(0, 255, 0, 0);
cv::Scalar CvColor::red_(0, 0, 255, 0);
cv::Scalar CvColor::purple_(139, 0, 139, 0);
cv::Scalar CvColor::yellow_(0, 255, 255, 0);
cv::Scalar CvColor::gray_(125, 125, 125, 0);

cv::Scalar intToColor(int index)
{
  cv::Scalar output_color = CvColor::white_;
  if (index == static_cast<int>(color_enum::white))
  {
    output_color = CvColor::white_;
  }
  else if (index == static_cast<int>(color_enum::blue))
  {
    output_color = CvColor::blue_;
  }
  else if (index == static_cast<int>(color_enum::green))
  {
    output_color = CvColor::green_;
  }
  else if (index == static_cast<int>(color_enum::red))
  {
    output_color = CvColor::red_;
  }
  else if (index == static_cast<int>(color_enum::yellow))
  {
    output_color = CvColor::yellow_;
  }
  else if (index == static_cast<int>(color_enum::gray))
  {
    output_color = CvColor::gray_;
  }
  else
  {
    output_color = CvColor::white_;
  }
  return output_color;
}

void loadCalibrationMatrix(const string& yml_filename, Mat& cameraMatrix, Mat& distCoeffs)
{
  cout << "yml_filename: " << yml_filename << endl;
  int image_width, image_height;
  string camera_name;
  FileStorage fs;
  fs.open(yml_filename, FileStorage::READ);
  if (!fs.isOpened())
  {
    cerr << " Fail to open " << yml_filename << endl;
    exit(EXIT_FAILURE);
  }
  // Get camera parameters

  fs["image_width"] >> image_width;
  fs["image_height"] >> image_height;
  fs["camera_name"] >> camera_name;
  cout << "Get camera_matrix" << endl;
  fs["camera_matrix"] >> cameraMatrix;
  cout << "Get distortion_coefficients" << endl;
  fs["distortion_coefficients"] >> distCoeffs;

  // Print out the camera parameters
  // cout << "\n -- Camera parameters -- " << endl;
  // cout << "\n CameraMatrix = " << endl << " " << cameraMatrix << endl << endl;
  // cout << " Distortion coefficients = " << endl << " " << distCoeffs << endl << endl;

  fs.release();
}
void calibrationImage(const Mat& src, Mat& dst, const Mat& cameraMatrix, const Mat& distCoeffs)
{
  Mat m_raw = src.clone();
  undistort(m_raw, dst, cameraMatrix, distCoeffs);
}

}  // namespace DriveNet