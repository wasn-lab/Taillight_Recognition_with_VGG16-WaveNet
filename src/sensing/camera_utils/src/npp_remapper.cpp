/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#include "npp_remapper.h"
#include "npp_remapper_impl.h"

NPPRemapper::NPPRemapper(const int rows, const int cols)
{
  remapper_impl_.reset(new NPPRemapperImpl(rows, cols));
}

NPPRemapper::~NPPRemapper()
{
}

int NPPRemapper::remap(const cv::Mat& src, cv::Mat& dst)
{
  return remapper_impl_->remap(src, dst);
}

int NPPRemapper::remap(const Npp8u* npp8u_ptr_in, cv::Mat& dst)
{
  return remapper_impl_->remap(npp8u_ptr_in, dst);
}

int NPPRemapper::remap(const Npp8u* npp8u_ptr_in, Npp8u* npp8u_ptr_out)
{
  return remapper_impl_->remap(npp8u_ptr_in, npp8u_ptr_out);
}

void NPPRemapper::set_interpolation_mode(NppiInterpolationMode mode)
{
  return remapper_impl_->set_interpolation_mode(mode);
}

int NPPRemapper::set_mapxy(const cv::Mat& mapx, const cv::Mat& mapy)
{
  return remapper_impl_->set_mapxy(mapx, mapy);
}
