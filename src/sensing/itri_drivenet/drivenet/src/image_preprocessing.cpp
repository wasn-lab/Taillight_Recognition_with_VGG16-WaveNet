#include "drivenet/image_preprocessing.h"

using namespace cv;
using namespace std;

namespace DriveNet
{
cv::Scalar Color::blue_(255, 0, 0, 0);
cv::Scalar Color::green_(0, 255, 0, 0);
cv::Scalar Color::red_(0, 0, 255, 0);
cv::Scalar Color::yellow_(51, 255, 255, 0);
cv::Scalar Color::gray_(125, 125, 125, 0);

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
} // namespace DriveNet
