/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#ifndef __CANONICAL_NPP_RESIZER_H__
#define __CANONICAL_NPP_RESIZER_H__

#include <memory>
#include <opencv2/core/mat.hpp>
#include <nppdefs.h>

class NPPResizerImpl;
class NPPResizer
{
private:
  std::unique_ptr<NPPResizerImpl> resizer_impl_;

public:
  NPPResizer(const int src_rows, const int src_cols, const double row_scale_factor, const double col_scale_factor);
  NPPResizer(const int src_rows, const int src_cols, const int dst_rows, const int dst_cols);
  NPPResizer(NPPResizer&) = delete;
  NPPResizer(NPPResizer&&) = delete;
  NPPResizer& operator=(NPPResizer&) = delete;
  NPPResizer& operator=(NPPResizer&&) = delete;
  ~NPPResizer();
  int resize(const cv::Mat& src, cv::Mat& dst);
  int resize(const Npp8u* npp8u_ptr_in, cv::Mat& dst);
  int resize_to_letterbox_yolov3(const cv::Mat& src, cv::Mat& dst);
  int resize_to_letterbox_yolov3(const Npp8u* npp8u_ptr_in, cv::Mat& dst);
  int resize_to_letterbox_yolov3(const Npp8u* npp8u_ptr_in, Npp8u* npp8u_ptr_out);
  void set_interpolation_mode(NppiInterpolationMode mode);
};

#endif  // __CANONICAL_NPP_RESIZER_H__
