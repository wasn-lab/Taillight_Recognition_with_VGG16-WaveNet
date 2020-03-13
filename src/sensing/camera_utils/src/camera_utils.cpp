#include <cmath>
#include <cassert>
#include <memory>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/version.hpp>
#include "npp_resizer.h"
#include "camera_params.h"
#include "camera_utils.h"
#include "camera_utils_defs.h"

namespace camera
{
const cv::Scalar g_color_black(0, 0, 0, 0);

/**
 * Get the undistortion parameters
 *
 * @param[out] mapx
 * @parma[out] mapy
 */
int get_undistortion_maps(cv::Mat& mapx, cv::Mat& mapy)
{
  assert(mapx.empty());
  assert(mapy.empty());
  cv::Mat intrinsic_0, distortion_0, intrinsic_1, distortion_1;
  cv::Mat opt_camera_matrix_0, opt_camera_matrix_1, opt_camera_matrix_2;
  cv::Mat camera_matrix, distortion_coefficients;

  cv::Mat mapx_16sc2, mapy_16uc1;

  // load insorinsic file
  cv::FileStorage fs;
  fs.open(CAMERA_UTILS_DATA_DIR "/sf3324.yml", cv::FileStorage::READ);
  fs["camera_matrix"] >> camera_matrix;
  fs["distortion_coefficients"] >> distortion_coefficients;
  fs.release();

  const cv::Size img_size = cv::Size(camera::raw_image_width, camera::raw_image_height);
  opt_camera_matrix_2 = cv::getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, img_size, 0);

  cv::initUndistortRectifyMap(camera_matrix, distortion_coefficients, cv::Mat(), camera_matrix, img_size, 0, mapx_16sc2,
                              mapy_16uc1);
  cv::convertMaps(mapx_16sc2, mapy_16uc1, mapx, mapy, CV_32FC1);
  return 0;
}

int fit_yolov3_image_size(const cv::Mat& in_img, cv::Mat& yolov3_img)
{
  cv::Mat temp_img;

  if (!yolov3_img.empty())
  {
    yolov3_img.release();
  }

  if (has_yolov3_image_size(in_img))
  {
    LOG(INFO) << "Already has yolov3 image, simply copy it.";
    yolov3_img = in_img;
    return 0;
  }

  cv::resize(in_img, temp_img, cv::Size(), camera::image_ratio_on_yolov3, camera::image_ratio_on_yolov3,
             cv::INTER_NEAREST);
  cv::copyMakeBorder(temp_img, yolov3_img, camera::top_border, camera::bottom_border, camera::left_border,
                     camera::right_border, cv::BORDER_CONSTANT, g_color_black);
  LOG_IF(WARNING, yolov3_img.rows != camera::yolov3_image_height) << yolov3_img.rows;
  LOG_IF(WARNING, yolov3_img.cols != camera::yolov3_image_width) << yolov3_img.cols;
  assert(yolov3_img.rows == camera::yolov3_image_height);
  assert(yolov3_img.cols == camera::yolov3_image_width);
  if (!temp_img.empty())
  {
    temp_img.release();
  }
  return 0;
}

int fit_yolov3_image_size(const cv::Mat& in_img, cv::Mat& yolov3_img, NPPResizer& resizer)
{
  if (!yolov3_img.empty())
  {
    yolov3_img.release();
  }

  if (camera::has_yolov3_image_size(in_img))
  {
    yolov3_img = in_img;
    return 0;
  }

  cv::Mat resized;
  resizer.resize(in_img, resized);
  cv::copyMakeBorder(resized, yolov3_img, camera::npp_top_border, camera::npp_bottom_border, camera::left_border,
                     camera::right_border, cv::BORDER_CONSTANT, g_color_black);
  assert(yolov3_img.rows == yolov3_image_height);
  assert(yolov3_img.cols == yolov3_image_width);
  return 0;
}

int scale_yolov3_image_to_raw_size(const cv::Mat& in_img, cv::Mat& scaled_img)
{
  if (!scaled_img.empty())
  {
    scaled_img.release();
  }
  if (!has_yolov3_image_size(in_img))
  {
    LOG(WARNING) << "Accept yolov3 size only";
    return 1;
  }
  cv::Mat temp_img;
  cv::Mat roi = in_img(cv::Rect(0, camera::top_border, camera::yolov3_image_width,
                                camera::yolov3_image_height - camera::bottom_border - camera::top_border));
  cv::resize(roi, temp_img, cv::Size(), camera::inv_image_ratio_on_yolov3, camera::inv_image_ratio_on_yolov3,
             cv::INTER_NEAREST);
  if (temp_img.rows == camera::raw_image_height + 1)
  {
    scaled_img = temp_img(cv::Rect(0, 0, camera::raw_image_width, camera::raw_image_height));
  }
  else
  {
    scaled_img = temp_img;
  }
  roi.release();
  temp_img.release();
  return 0;
}

bool has_yolov3_image_size(const cv::Mat& in_img)
{
  if ((in_img.rows == camera::yolov3_image_height) && (in_img.cols == camera::yolov3_image_width))
  {
    return true;
  }
  VLOG(2) << "Not yolov3 image size: Expected: 608x608, Actual:" << in_img.cols << "x" << in_img.rows;
  return false;
}

bool has_raw_image_size(const cv::Mat& in_img)
{
  if ((in_img.rows == camera::raw_image_height) || (in_img.cols == camera::raw_image_width))
  {
    return true;
  }
  VLOG(2) << "Not raw image size: Expected: 1920x1208, Actual:" << in_img.cols << "x" << in_img.rows;
  return false;
}

int camera_to_yolov3_xy(const int x, const int y, int* yolov3_x, int* yolov3_y)
{
  const auto image_ratio = camera::image_ratio_on_yolov3;
  assert(x >= 0);
  assert(x < camera::raw_image_width);
  assert(y >= 0);
  assert(y < camera::raw_image_height);
  *yolov3_x = x * image_ratio + camera::left_border;
  *yolov3_y = y * image_ratio + camera::top_border;
  return 0;
}

int yolov3_to_camera_xy(const int x, const int y, int* camera_x, int* camera_y)
{
  const auto image_ratio = camera::image_ratio_on_yolov3;
  assert(x >= camera::left_border);
  assert(x < camera::yolov3_image_width);
  assert(y >= camera::top_border);
  assert(y < camera::yolov3_image_height - camera::bottom_border);
  *camera_x = (x - camera::left_border) / image_ratio;
  *camera_y = (y - camera::top_border) / image_ratio;
  return 0;
}

bool cvmats_are_equal(const cv::Mat& img1, const cv::Mat& img2)
{
  if (img1.empty() && img2.empty())
  {
    return true;
  }
  if (img1.cols != img2.cols || img1.rows != img2.rows || img1.dims != img2.dims)
  {
    return false;
  }
  cv::Mat gray1, gray2, diff;
#if CV_VERSION_MAJOR == 4
  cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
#else
  cv::cvtColor(img1, gray1, CV_BGR2GRAY);
  cv::cvtColor(img2, gray2, CV_BGR2GRAY);
#endif
  cv::bitwise_xor(gray1, gray2, diff);
  int nz = cv::countNonZero(diff);
  return nz == 0;
}

uint32_t calc_cvmat_checksum(const cv::Mat& img)
{
  assert(img.total() > 0);
  return calc_bytes_checksum(img.data, img.total() * img.elemSize());
}

uint32_t calc_bytes_checksum(const unsigned char* bytes, size_t len)
{
  const uint32_t p = 16777619;
  auto hash = static_cast<uint32_t>(2166136261);
  for (size_t i = 0; i < len; i++)
  {
    hash = (hash ^ bytes[i]) * p;
  }

  hash += hash << 13u;
  hash ^= hash >> 7u;
  hash += hash << 3u;
  hash ^= hash >> 17u;
  hash += hash << 5u;
  return hash;
}

bool is_black_image(const cv::Mat& img)
{
  if ((img.rows == 0) || (img.cols == 0)) {
    return false;
  }
  const auto nbytes = img.total() * img.elemSize();
  std::unique_ptr<uint8_t[]> zeros{new uint8_t[nbytes]};
  std::memset(zeros.get(), 0, nbytes);
  return std::memcmp(img.data, zeros.get(), nbytes) == 0;
}

int release_cv_mat_if_necessary(cv::Mat& img)
{
  if (!img.empty())
  {
    img.release();
  }
  return 0;
}

} // namespace camera
