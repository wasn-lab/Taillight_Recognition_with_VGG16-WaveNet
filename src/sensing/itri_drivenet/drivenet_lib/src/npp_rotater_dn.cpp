/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#include "npp_rotater_dn.h"
#include "npp_rotater_impl_dn.h"
namespace DriveNet_npp
{
NPPRotater::NPPRotater(const int src_rows, const int src_cols, const int rotation_degree)
{
  int dst_rows = 0, dst_cols = 0;
  if (rotation_degree == 90 || rotation_degree == 270)
  {
    dst_cols = src_rows;
    dst_rows = src_cols;
  }
  else if (rotation_degree == 180)
  {
    dst_rows = src_rows;
    dst_cols = src_cols;
  }
  else
  {
    assert(rotation_degree != 180 && rotation_degree != 90 && rotation_degree != 270 );
  }
  
  rotater_impl_.reset(new NPPRotaterImpl(src_rows, src_cols, dst_rows, dst_cols));
}

NPPRotater::NPPRotater(const int src_rows, const int src_cols, const int dst_rows, const int dst_cols)
{
  rotater_impl_.reset(new NPPRotaterImpl(src_rows, src_cols, dst_rows, dst_cols));
}

NPPRotater::~NPPRotater()
{
}

int NPPRotater::rotate(const cv::Mat& src, cv::Mat& dst, const int rotation_degree)
{
  return rotater_impl_->rotate(src, dst, rotation_degree);
}

void NPPRotater::set_interpolation_mode(NppiInterpolationMode mode)
{
  return rotater_impl_->set_interpolation_mode(mode);
}
}
