/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#ifndef __NPP_RESIZER_DN_H__
#define __NPP_RESIZER_DN_H__

#include <memory>
#include <opencv2/core/mat.hpp>
#include <nppdefs.h>

namespace DriveNet_npp
{
class NPPResizerImpl;
class NPPResizer
{
private:
  std::unique_ptr<NPPResizerImpl> resizer_impl_;

public:
  NPPResizer(const int src_rows, const int src_cols, const double row_scale_factor, const double col_scale_factor);
  NPPResizer(const int src_rows, const int src_cols, const int dst_rows, const int dst_cols);
  ~NPPResizer();
  int resize(Npp8u* src, Npp8u* dst);
  void set_interpolation_mode(NppiInterpolationMode mode);
};
}
#endif  // __NPP_RESIZER_DN_H__
