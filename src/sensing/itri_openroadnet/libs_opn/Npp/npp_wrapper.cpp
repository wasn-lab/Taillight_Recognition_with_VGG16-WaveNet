/*
   CREATER: ICL U300
   DATE: Aug, 2019
*/

#include <assert.h>
#include "npp_wrapper.h"
#include "nppi_geometry_transforms.h"
#include "npp.h"

namespace npp_wrapper
{
npp::npp()
{
}

npp::~npp()
{
  nppiFree(dst_npp8u_ptr);
  nppiFree(dst_npp8u_ptr_);
  nppiFree(src_npp8u_ptr);
  nppiFree(src_npp8u_ptr_);
}

void npp::init_1(const int ori_cols, const int ori_rows, const double wscale, const double hscale)
{
  assert(hscale > 0);
  assert(wscale > 0);

  int dummy;

  src_npp8u_ptr = nppiMalloc_8u_C1(ori_cols, ori_rows, &dummy);

  assert(src_npp8u_ptr);

  const int dst_rows = hscale;
  const int dst_cols = wscale;

  dst_npp8u_ptr = nppiMalloc_8u_C1(dst_cols, dst_rows, &dummy);

  assert(dst_npp8u_ptr);
}

void npp::init_3(const int ori_cols, const int ori_rows, const double wscale, const double hscale)
{
  assert(hscale > 0);
  assert(wscale > 0);

  int dummy;

  src_npp8u_ptr_ = nppiMalloc_8u_C3(ori_cols, ori_rows, &dummy);

  assert(src_npp8u_ptr_);

  const int dst_rows = hscale;
  const int dst_cols = wscale;

  dst_npp8u_ptr_ = nppiMalloc_8u_C3(dst_cols, dst_rows, &dummy);

  assert(dst_npp8u_ptr_);
  RGBOrder[0] = 0;
  RGBOrder[1] = 1;
  RGBOrder[2] = 2;
}

int npp::resize1(const cv::Mat& src, cv::Mat& dst, const double hscale, const double wscale,
                 const NppiInterpolationMode interpolation_mode)
{
  const NppiSize src_size = {.width = src.cols, .height = src.rows };
  const NppiRect src_roi = {.x = 0, .y = 0, .width = src.cols, .height = src.rows };
  const int src_line_steps = src.cols * 1;

  const int dst_rows = hscale;
  const int dst_cols = wscale;
  const int dst_line_steps = dst_cols * 1;
  const int num_dst_bytes = dst_cols * dst_rows * 1;
  const NppiSize dst_size = {.width = dst_cols, .height = dst_rows };
  const NppiRect dst_roi = {.x = 0, .y = 0, .width = dst_cols, .height = dst_rows };

  cudaMemcpy(src_npp8u_ptr, src.data, src.cols * src.rows * 1, cudaMemcpyHostToDevice);

  NppStatus result;

  result = nppiResize_8u_C1R(src_npp8u_ptr, src_line_steps, src_size, src_roi, dst_npp8u_ptr, dst_line_steps, dst_size,
                             dst_roi, interpolation_mode);

  assert(result == NPP_SUCCESS);

  if (!dst.empty())
  {
    dst.release();
  }

  dst = cv::Mat(dst_rows, dst_cols, CV_8UC1);

  cudaMemcpy(dst.data, dst_npp8u_ptr, num_dst_bytes, cudaMemcpyDeviceToHost);
  return result;
}

int npp::resize3(const cv::Mat& src, cv::Mat& dst, const double hscale, const double wscale,
                 const NppiInterpolationMode interpolation_mode)
{
  assert(hscale > 0);
  assert(wscale > 0);

  const NppiSize src_size = {.width = src.cols, .height = src.rows };
  const NppiRect src_roi = {.x = 0, .y = 0, .width = src.cols, .height = src.rows };
  const int src_line_steps = src.cols * 3;
  const int num_src_bytes = src.cols * src.rows * 3;

  assert(num_src_bytes > 0);

  const int dst_rows = hscale;
  const int dst_cols = wscale;
  const int dst_line_steps = dst_cols * 3;
  const int num_dst_bytes = dst_cols * dst_rows * 3;
  const NppiSize dst_size = {.width = dst_cols, .height = dst_rows };
  const NppiRect dst_roi = {.x = 0, .y = 0, .width = dst_cols, .height = dst_rows };

  assert(num_dst_bytes > 0);

  cudaMemcpy(src_npp8u_ptr_, src.data, num_src_bytes, cudaMemcpyHostToDevice);

  NppStatus result;
  result = nppiResize_8u_C3R(src_npp8u_ptr_, src_line_steps, src_size, src_roi, dst_npp8u_ptr_, dst_line_steps,
                             dst_size, dst_roi, interpolation_mode);

  assert(result == NPP_SUCCESS);

  if (!dst.empty())
  {
    dst.release();
  }

  dst = cv::Mat(dst_rows, dst_cols, CV_8UC3);

  cudaMemcpy(dst.data, dst_npp8u_ptr_, num_dst_bytes, cudaMemcpyDeviceToHost);
  return result;
}

int npp::resize3(const Npp8u* rawCUDA_, cv::Mat& dst, const double hscale, const double wscale,
                 const NppiInterpolationMode interpolation_mode)
{
  assert(hscale > 0);
  assert(wscale > 0);

  const NppiSize src_size = {.width = 1920, .height = 1208 };
  const NppiRect src_roi = {.x = 0, .y = 0, .width = 1920, .height = 1208 };
  const int src_line_steps = 1920 * 3;
  const int dst_rows = hscale;
  const int dst_cols = wscale;
  const int dst_line_steps = dst_cols * 3;
  const int num_dst_bytes = dst_cols * dst_rows * 3;
  const NppiSize dst_size = {.width = dst_cols, .height = dst_rows };
  const NppiRect dst_roi = {.x = 0, .y = 0, .width = dst_cols, .height = dst_rows };

  assert(num_dst_bytes > 0);

  NppStatus result;

  // result = nppiSwapChannels_8u_C3R(rawCUDA_, src_line_steps, src_npp8u_ptr_, src_line_steps, src_size, RGBOrder);  //
  // RGB to BGR of uint in CUDA

  // result = nppiSwapChannels_8u_C4C3R(rawCUDA_, src_line_steps, src_npp8u_ptr_, src_line_steps, src_size, RGBOrder);
  // // RGBA to RGB of uint in CUDA

  assert(result == NPP_SUCCESS);

  result = nppiResize_8u_C3R(rawCUDA_, src_line_steps, src_size, src_roi, dst_npp8u_ptr_, dst_line_steps, dst_size,
                             dst_roi, interpolation_mode);

  // result = nppiResize_8u_C3R(src_npp8u_ptr_, src_line_steps, src_size, src_roi, dst_npp8u_ptr_, dst_line_steps,
  //                                     dst_size, dst_roi, interpolation_mode);

  assert(result == NPP_SUCCESS);

  if (!dst.empty())
  {
    dst.release();
  }

  dst = cv::Mat(dst_rows, dst_cols, CV_8UC3);

  cudaMemcpy(dst.data, dst_npp8u_ptr_, num_dst_bytes, cudaMemcpyDeviceToHost);
  return result;
}

};  // namespace
