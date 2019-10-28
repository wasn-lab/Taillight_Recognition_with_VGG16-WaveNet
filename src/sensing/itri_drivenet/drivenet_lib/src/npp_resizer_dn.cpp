/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#include "npp_resizer_dn.h"
#include "npp_resizer_impl_dn.h"
namespace DriveNet_npp{

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
int NPPResizer::resize(Npp8u* src, Npp8u* dst)
{
  return resizer_impl_->resize(src, dst);
}
void NPPResizer::set_interpolation_mode(NppiInterpolationMode mode)
{
  return resizer_impl_->set_interpolation_mode(mode);
}
}
