/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#ifndef __NPP_RESIZER_IMPL_DN_H__
#define __NPP_RESIZER_IMPL_DN_H__

#include <opencv2/core/mat.hpp>
#include <nppdefs.h>
#include <mutex>

namespace DriveNet_npp
{
class NPPResizerImpl
{
private:
  const int src_line_steps_, dst_line_steps_;
  NppiSize src_size_, dst_size_;
  NppiRect src_roi_, dst_roi_;
  Npp8u* src_npp8u_ptr_cuda_;
  Npp8u* dst_npp8u_ptr_cuda_;
  NppiInterpolationMode interpolation_mode_;
  std::mutex mu_;

public:
  NPPResizerImpl(const int src_rows, const int src_cols, const double row_scale_factor, const double col_scale_factor);
  NPPResizerImpl(const int src_rows, const int src_cols, const int dst_rows, const int dst_cols);
  ~NPPResizerImpl();
  int resize(Npp8u* src, Npp8u* dst);
  void set_interpolation_mode(NppiInterpolationMode mode);
};
}
#endif  // __NPP_RESIZER_IMPL_DN_H__
