#ifndef IMAGE_PREPROCESSING_H_
#define IMAGE_PREPROCESSING_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace DriveNet
{
class Color
{
public:
  static cv::Scalar g_color_blue;
  static cv::Scalar g_color_red;
  static cv::Scalar g_color_green;
  static cv::Scalar g_color_yellow;
  static cv::Scalar g_color_gray;
};

void loadCalibrationMatrix(std::string yml_filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs);
void calibrationImage(const cv::Mat& src, cv::Mat& dst, cv::Mat cameraMatrix, cv::Mat distCoeffs);
} // namespace DriveNet
#endif /*IMAGE_PREPROCESSING_H_*/