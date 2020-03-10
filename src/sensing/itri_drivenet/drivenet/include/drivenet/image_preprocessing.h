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
  static cv::Scalar blue_;
  static cv::Scalar red_;
  static cv::Scalar green_;
  static cv::Scalar yellow_;
  static cv::Scalar gray_;
};

void loadCalibrationMatrix(const std::string& yml_filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs);
void calibrationImage(const cv::Mat& src, cv::Mat& dst, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs);
} // namespace DriveNet
#endif /*IMAGE_PREPROCESSING_H_*/