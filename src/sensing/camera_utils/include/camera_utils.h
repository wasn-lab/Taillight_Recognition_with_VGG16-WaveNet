#ifndef __CAMERA_UTILS_H__
#define __CAMERA_UTILS_H__

#include "opencv2/core/mat.hpp"
#include "npp_resizer.h"

namespace camera
{
int get_undistortion_maps(cv::Mat& mapx, cv::Mat& mapy);
int fit_yolov3_image_size(const cv::Mat& in_img, cv::Mat& yolov3_img);
int fit_yolov3_image_size(const cv::Mat& in_img, cv::Mat& yolov3_img, NPPResizer& resizer);
int scale_yolov3_image_to_raw_size(const cv::Mat& in_img, cv::Mat& scaled_img);
bool has_yolov3_image_size(const cv::Mat& in_img);
bool has_raw_image_size(const cv::Mat& in_img);
int camera_to_yolov3_xy(const int x, const int y, int* yolov3_x, int* yolov3_y);
int yolov3_to_camera_xy(const int x, const int y, int* camera_x, int* camera_y);
bool cvmats_are_equal(const cv::Mat& img1, const cv::Mat& img2);
uint32_t calc_cvmat_checksum(const cv::Mat& img);
uint32_t calc_bytes_checksum(const unsigned char* bytes, size_t len);
bool is_black_image(const cv::Mat& img);
int release_cv_mat_if_necessary(cv::Mat& img);
}  // namespace camera
#endif  // __CAMERA_UTILS_H__
