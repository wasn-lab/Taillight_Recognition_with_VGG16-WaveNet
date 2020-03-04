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
  int imageWidth, imageHeight;
  string cameraName;
  FileStorage fs;
  fs.open(yml_filename, FileStorage::READ);
  if (!fs.isOpened())
  {
    cerr << " Fail to open " << yml_filename << endl;
    exit(EXIT_FAILURE);
  }
  // Get camera parameters

  fs["image_width"] >> imageWidth;
  fs["image_height"] >> imageHeight;
  fs["camera_name"] >> cameraName;
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
  Mat M_raw = src.clone();
  undistort(M_raw, dst, cameraMatrix, distCoeffs);
}
} // namespace DriveNet
