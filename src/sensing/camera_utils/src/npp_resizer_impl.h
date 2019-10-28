/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#ifndef __NPP_RESIZER_IMPL_H__
#define __NPP_RESIZER_IMPL_H__

#include <opencv2/core/mat.hpp>
#include <nppdefs.h>

class NPPResizerImpl
{
private:
  const int src_rows_, src_cols_;
  const int dst_rows_, dst_cols_;
  const int src_line_steps_, dst_line_steps_;
  const int num_src_bytes_, num_dst_bytes_;
  NppiSize src_size_, dst_size_;
  NppiRect src_roi_, dst_roi_;
  Npp8u* src_npp8u_ptr_cuda_;
  Npp8u* dst_npp8u_ptr_cuda_;
  Npp8u* npp8u_ptr_cuda_608x384_;
  NppiInterpolationMode interpolation_mode_;

public:
  NPPResizerImpl(const int src_rows, const int src_cols, const double row_scale_factor, const double col_scale_factor);
  NPPResizerImpl(const int src_rows, const int src_cols, const int dst_rows, const int dst_cols);
  ~NPPResizerImpl();
  int resize(const cv::Mat& src, cv::Mat& dst);
  int resize(const Npp8u* npp8u_ptr_in, cv::Mat& dst);
  int resize_to_letterbox_yolov3(const cv::Mat& src, cv::Mat& dst);
  int resize_to_letterbox_yolov3(const Npp8u* npp8u_ptr_in, cv::Mat& dst);
  int resize_to_letterbox_yolov3(const Npp8u* npp8u_ptr_in, Npp8u* npp8u_ptr_out);
  void set_interpolation_mode(NppiInterpolationMode mode);
};

#endif  // __NPP_RESIZER_IMPL_H__
