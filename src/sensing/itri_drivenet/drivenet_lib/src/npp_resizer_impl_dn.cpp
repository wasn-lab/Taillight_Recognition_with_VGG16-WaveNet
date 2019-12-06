/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#include <assert.h>
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include "nppi_geometry_transforms.h"
#include "npp.h"
#include "npp_resizer_impl_dn.h"

namespace DriveNet_npp
{
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

  interpolation_mode_ = NPPI_INTER_LINEAR;
}

NPPResizerImpl::~NPPResizerImpl()
{
  nppiFree(src_npp8u_ptr_cuda_);
  nppiFree(dst_npp8u_ptr_cuda_);
}

int NPPResizerImpl::resize(Npp8u* src, Npp8u* dst)
{
  NppStatus result = nppiResize_8u_C3R(src, src_line_steps_, src_size_, src_roi_, dst, dst_line_steps_, dst_size_,
                                       dst_roi_, interpolation_mode_);
  if (result != NPP_SUCCESS)
  {
    LOG(WARNING) << "nppiResize_8u_C3R returns: " << result;
    assert(0);
  }
  return result;
}
void NPPResizerImpl::set_interpolation_mode(NppiInterpolationMode mode)
{
  interpolation_mode_ = mode;
}
}
