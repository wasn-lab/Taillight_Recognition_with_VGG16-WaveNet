/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#ifndef __NPP_ROTATER_IMPL_DN_H__
#define __NPP_ROTATER_IMPL_DN_H__

#include <opencv2/core/mat.hpp>
#include <nppdefs.h>
#include <mutex>

namespace DriveNet_npp
{
class NPPRotaterImpl
{
private:
  const int src_rows_, src_cols_;
  int dst_rows_, dst_cols_;
  int src_line_steps_, dst_line_steps_;
  const int num_src_bytes_, num_dst_bytes_;
  NppiSize src_size_, dst_size_;
  NppiRect src_roi_, dst_roi_;
  Npp8u* src_npp8u_ptr_cuda_;
  Npp8u* dst_npp8u_ptr_cuda_;
  NppiInterpolationMode interpolation_mode_;
  std::mutex mu_;

public:
  NPPRotaterImpl(const int src_rows, const int src_cols, const int rotation_degree);
  NPPRotaterImpl(const int src_rows, const int src_cols, const int dst_rows, const int dst_cols);
  ~NPPRotaterImpl();
  int rotate(const cv::Mat& src, cv::Mat& dst, int rotation_degree);
  void set_interpolation_mode(NppiInterpolationMode mode);
};
}
#endif  // __NPP_ROTATER_IMPL_DN_H__
