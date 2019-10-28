/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#include <assert.h>
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include "nppi_geometry_transforms.h"
#include "npp.h"
#include "npp_rotater_impl_dn.h"

namespace DriveNet_npp
{
// Only accepts images of 3 channels(RGB) and each color is unsigned char.
NPPRotaterImpl::NPPRotaterImpl(const int src_rows, const int src_cols, const int rotation_degree)
:NPPRotaterImpl(src_rows, src_cols, src_cols, src_rows)
{
}

NPPRotaterImpl::NPPRotaterImpl(const int src_rows, const int src_cols, const int dst_rows, const int dst_cols)
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

  interpolation_mode_ = NPPI_INTER_LINEAR;
}

NPPRotaterImpl::~NPPRotaterImpl()
{
  nppiFree(src_npp8u_ptr_cuda_);
  nppiFree(dst_npp8u_ptr_cuda_);
}

int NPPRotaterImpl::rotate(const cv::Mat& src, cv::Mat& dst, const int rotation_degree)
{
  std::lock_guard<std::mutex> lk(mu_);
  
  if (rotation_degree == 180)
  {
    dst_rows_ = src.rows;
    dst_cols_ = src.cols;
    dst_line_steps_ = dst_cols_ * 3;
    dst_roi_.width = dst_cols_;
    dst_roi_.height = dst_rows_;
  }

  CHECK(src.rows == src_rows_);
  CHECK(src.cols == src_cols_);
  CHECK(src.channels() == 3);

  double rotated_bounding_box[2][2];
  nppiGetRotateBound(src_roi_, rotated_bounding_box, rotation_degree, 0, 0);
  const double shift_x = -rotated_bounding_box[0][0];
  const double shift_y = -rotated_bounding_box[0][1];


  cudaMemcpy(src_npp8u_ptr_cuda_, src.data, num_src_bytes_, cudaMemcpyHostToDevice);
  NppStatus result = nppiRotate_8u_C3R(src_npp8u_ptr_cuda_, src_size_, src_line_steps_, src_roi_, dst_npp8u_ptr_cuda_, dst_line_steps_,
                                       dst_roi_, rotation_degree, shift_x, shift_y, interpolation_mode_);
  if (result != NPP_SUCCESS)
  {
    LOG(WARNING) << "nppiResize_8u_C3R returns: " << result;
    assert(0);
  }
  if (!dst.empty())
  {
    dst.release();
  }
  dst = cv::Mat(dst_rows_, dst_cols_, CV_8UC3);

  cudaMemcpy(dst.data, dst_npp8u_ptr_cuda_, num_dst_bytes_, cudaMemcpyDeviceToHost);
  return result;
}

void NPPRotaterImpl::set_interpolation_mode(NppiInterpolationMode mode)
{
  interpolation_mode_ = mode;
}
}
