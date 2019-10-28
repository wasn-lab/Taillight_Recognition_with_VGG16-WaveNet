/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#include "npp_resizer.h"
#include "npp_resizer_impl.h"

NPPResizer::NPPResizer(const int src_rows, const int src_cols, const double row_scale_factor,
                       const double col_scale_factor)
{
  resizer_impl_.reset(new NPPResizerImpl(src_rows, src_cols, row_scale_factor, col_scale_factor));
}

NPPResizer::NPPResizer(const int src_rows, const int src_cols, const int dst_rows, const int dst_cols)
{
  resizer_impl_.reset(new NPPResizerImpl(src_rows, src_cols, dst_rows, dst_cols));
}

NPPResizer::~NPPResizer()
{
}

int NPPResizer::resize(const cv::Mat& src, cv::Mat& dst)
{
  return resizer_impl_->resize(src, dst);
}

int NPPResizer::resize(const Npp8u* npp8u_ptr_in, cv::Mat& dst)
{
  return resizer_impl_->resize(npp8u_ptr_in, dst);
}

int NPPResizer::resize_to_letterbox_yolov3(const cv::Mat& src, cv::Mat& dst)
{
  return resizer_impl_->resize_to_letterbox_yolov3(src, dst);
}

int NPPResizer::resize_to_letterbox_yolov3(const Npp8u* npp8u_ptr_in, cv::Mat& dst)
{
  return resizer_impl_->resize_to_letterbox_yolov3(npp8u_ptr_in, dst);
}

int NPPResizer::resize_to_letterbox_yolov3(const Npp8u* npp8u_ptr_in, Npp8u* npp8u_ptr_out)
{
  return resizer_impl_->resize_to_letterbox_yolov3(npp8u_ptr_in, npp8u_ptr_out);
}

void NPPResizer::set_interpolation_mode(NppiInterpolationMode mode)
{
  return resizer_impl_->set_interpolation_mode(mode);
}
