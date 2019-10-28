/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#include <assert.h>
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include "nppi_geometry_transforms.h"
#include "npp.h"
#include "npp_resizer_impl.h"
#include "camera_params.h"
#include "camera_utils.h"

// Only accepts images of 3 channels(RGB) and each color is unsigned char.
NPPResizerImpl::NPPResizerImpl(const int src_rows, const int src_cols, const double row_scale_factor,
                               const double col_scale_factor)
  : NPPResizerImpl(src_rows, src_cols, int(src_rows* row_scale_factor), int(src_cols* col_scale_factor))
{
}

NPPResizerImpl::NPPResizerImpl(const int src_rows, const int src_cols, const int dst_rows, const int dst_cols)
  : src_rows_(src_rows)
  , src_cols_(src_cols)
  , dst_rows_(dst_rows)
  , dst_cols_(dst_cols)
  , src_line_steps_(src_cols * 3)
  , dst_line_steps_(dst_cols * 3)
  , num_src_bytes_(src_rows * src_cols * 3)
  , num_dst_bytes_(dst_rows * dst_cols * 3)
{
  CHECK(src_rows > 0);
  CHECK(src_cols > 0);
  CHECK(dst_rows > 0);
  CHECK(dst_cols > 0);

  src_size_ = {.width = src_cols, .height = src_rows };
  dst_size_ = {.width = dst_cols, .height = dst_rows };
  src_roi_ = {.x = 0, .y = 0, .width = src_cols, .height = src_rows };
  dst_roi_ = {.x = 0, .y = 0, .width = dst_cols, .height = dst_rows };

  int dummy = 0;

  src_npp8u_ptr_cuda_ = nppiMalloc_8u_C3(src_cols, src_rows, &dummy);
  CHECK(src_npp8u_ptr_cuda_);
  dst_npp8u_ptr_cuda_ = nppiMalloc_8u_C3(dst_cols, dst_rows, &dummy);
  CHECK(dst_npp8u_ptr_cuda_);

  // When resize from 1920x1208, first resize to 608x304, and then pad black pixels in the border to make the image
  // 608x608
  npp8u_ptr_cuda_608x384_ =
      nppiMalloc_8u_C3(camera::yolov3_image_width, camera::yolov3_letterbox_visible_height, &dummy);
  CHECK(npp8u_ptr_cuda_608x384_);

  interpolation_mode_ = NPPI_INTER_LINEAR;
}

NPPResizerImpl::~NPPResizerImpl()
{
  nppiFree(src_npp8u_ptr_cuda_);
  nppiFree(dst_npp8u_ptr_cuda_);
  nppiFree(npp8u_ptr_cuda_608x384_);
}

int NPPResizerImpl::resize(const cv::Mat& src, cv::Mat& dst)
{
  CHECK(src.rows == src_rows_);
  CHECK(src.cols == src_cols_);
  CHECK(src.channels() == 3);
  cudaMemcpyAsync(src_npp8u_ptr_cuda_, src.data, num_src_bytes_, cudaMemcpyHostToDevice, cudaStreamPerThread);
  return resize(src_npp8u_ptr_cuda_, dst);
}

int NPPResizerImpl::resize(const Npp8u* npp8u_ptr_in, cv::Mat& dst)
{
  NppStatus result = nppiResize_8u_C3R(npp8u_ptr_in, src_line_steps_, src_size_, src_roi_, dst_npp8u_ptr_cuda_,
                                       dst_line_steps_, dst_size_, dst_roi_, interpolation_mode_);
  assert(result == NPP_SUCCESS);
  camera::release_cv_mat_if_necessary(dst);
  dst.create(dst_rows_, dst_cols_, CV_8UC3);
  cudaMemcpyAsync(dst.data, dst_npp8u_ptr_cuda_, num_dst_bytes_, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  return result;
}

int NPPResizerImpl::resize_to_letterbox_yolov3(const cv::Mat& src, cv::Mat& dst)
{
  CHECK(src.rows == src_rows_);
  CHECK(src.cols == src_cols_);
  CHECK(src.channels() == 3);
  cudaMemcpyAsync(src_npp8u_ptr_cuda_, src.data, num_src_bytes_, cudaMemcpyHostToDevice, cudaStreamPerThread);
  return resize_to_letterbox_yolov3(src_npp8u_ptr_cuda_, dst);
}

int NPPResizerImpl::resize_to_letterbox_yolov3(const Npp8u* npp8u_ptr_in, cv::Mat& dst)
{
  LOG_IF(WARNING, src_rows_ > src_cols_) << "Expect width >= height, but got" << src_cols_ << "x" << src_rows_;

  NppiSize temp_size = {.width = camera::yolov3_image_width, .height = camera::yolov3_letterbox_visible_height };
  NppiRect temp_roi = {
    .x = 0, .y = 0, .width = camera::yolov3_image_width, .height = camera::yolov3_letterbox_visible_height
  };

  NppStatus result = nppiResize_8u_C3R(npp8u_ptr_in, src_line_steps_, src_size_, src_roi_, npp8u_ptr_cuda_608x384_,
                                       dst_line_steps_, temp_size, temp_roi, interpolation_mode_);
  assert(result == NPP_SUCCESS);

  const Npp8u border_rgb_color[] = { 0, 0, 0, 0 };
  result = nppiCopyConstBorder_8u_C3R(npp8u_ptr_cuda_608x384_, dst_line_steps_, temp_size, dst_npp8u_ptr_cuda_,
                                      dst_line_steps_, dst_size_, camera::npp_top_border,
                                      /*nLeftBorderWidth*/ 0, border_rgb_color);
  camera::release_cv_mat_if_necessary(dst);
  dst.create(dst_rows_, dst_cols_, CV_8UC3);

  cudaMemcpyAsync(dst.data, dst_npp8u_ptr_cuda_, num_dst_bytes_, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  return result;
}

int NPPResizerImpl::resize_to_letterbox_yolov3(const Npp8u* npp8u_ptr_in, Npp8u* npp8u_ptr_out)
{
  NppiSize temp_size = {.width = camera::yolov3_image_width, .height = camera::yolov3_letterbox_visible_height };
  NppiRect temp_roi = {
    .x = 0, .y = 0, .width = camera::yolov3_image_width, .height = camera::yolov3_letterbox_visible_height
  };

  NppStatus result = nppiResize_8u_C3R(npp8u_ptr_in, src_line_steps_, src_size_, src_roi_, npp8u_ptr_cuda_608x384_,
                                       dst_line_steps_, temp_size, temp_roi, interpolation_mode_);
  assert(result == NPP_SUCCESS);

  const Npp8u border_rgb_color[] = { 0, 0, 0, 0 };
  result = nppiCopyConstBorder_8u_C3R(npp8u_ptr_cuda_608x384_, dst_line_steps_, temp_size, dst_npp8u_ptr_cuda_,
                                      dst_line_steps_, dst_size_, camera::npp_top_border,
                                      /*nLeftBorderWidth*/ 0, border_rgb_color);
  assert(result == NPP_SUCCESS);
  cudaMemcpyAsync(npp8u_ptr_out, dst_npp8u_ptr_cuda_, num_dst_bytes_, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
  return 0;
}

void NPPResizerImpl::set_interpolation_mode(NppiInterpolationMode mode)
{
  interpolation_mode_ = mode;
}
