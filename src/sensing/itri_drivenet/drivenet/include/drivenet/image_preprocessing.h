#ifndef IMAGE_PREPROCESSING_H_
#define IMAGE_PREPROCESSING_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void loadCalibrationMatrix(String yml_filename, Mat& cameraMatrix, Mat& distCoeffs);
void calibrationImage(const Mat src, Mat& dst, Mat cameraMatrix, Mat distCoeffs);

#endif /*IMAGE_PREPROCESSING_H_*/