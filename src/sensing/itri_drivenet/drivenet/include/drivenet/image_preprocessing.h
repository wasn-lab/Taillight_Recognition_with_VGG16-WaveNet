#ifndef IMAGE_PREPROCESSING_H_
#define IMAGE_PREPROCESSING_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

namespace DriveNet
{
class Color{
    public:
    static cv::Scalar g_color_blue;
    static cv::Scalar g_color_red;
    static cv::Scalar g_color_green;
    static cv::Scalar g_color_gray;
};

void loadCalibrationMatrix(String yml_filename, Mat& cameraMatrix, Mat& distCoeffs);
void calibrationImage(const Mat& src, Mat& dst, Mat cameraMatrix, Mat distCoeffs);
}
#endif /*IMAGE_PREPROCESSING_H_*/