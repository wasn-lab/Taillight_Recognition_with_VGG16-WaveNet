#ifndef IMAGE_PREPROCESSING_H_
#define IMAGE_PREPROCESSING_H_

#include <iostream>
#include <opencv2/core/version.hpp>
#include <opencv2/core/core.hpp>
#if CV_VERSION_MAJOR == 4
#include <opencv2/calib3d/calib3d.hpp>
#else
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <opencv2/highgui/highgui.hpp>

namespace DriveNet
{
struct PixelPosition
{
  int u;
  int v;
};

class CvColor
{
public:
  static cv::Scalar white_;
  static cv::Scalar blue_;
  static cv::Scalar green_;
  static cv::Scalar red_;
  static cv::Scalar yellow_;
  static cv::Scalar gray_;
};

enum color_enum
{
  white = 0,  // 0
  blue,       // 1
  red,        // 2
  green,      // 3
  yellow,     // 4
  gray,       // 5
};

cv::Scalar intToColor(int index);
void loadCalibrationMatrix(const std::string& yml_filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs);
void calibrationImage(const cv::Mat& src, cv::Mat& dst, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs);
}  // namespace DriveNet

#endif /*IMAGE_PREPROCESSING_H_*/
