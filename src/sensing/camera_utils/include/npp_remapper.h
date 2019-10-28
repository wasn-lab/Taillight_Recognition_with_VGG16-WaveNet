/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#ifndef __NPP_REMAPPER_H__
#define __NPP_REMAPPER_H__

#include <memory>
#include <opencv2/core/mat.hpp>
#include <nppdefs.h>

class NPPRemapperImpl;
class NPPRemapper
{
private:
  std::unique_ptr<NPPRemapperImpl> remapper_impl_;

public:
  NPPRemapper(const int rows, const int cols);
  ~NPPRemapper();
  int remap(const cv::Mat& src, cv::Mat& dst);
  int remap(const Npp8u* npp8u_ptr_in, cv::Mat& dst);
  int remap(const Npp8u* npp8u_ptr_in, Npp8u* npp8u_ptr_out);
  void set_interpolation_mode(NppiInterpolationMode mode);
  int set_mapxy(const cv::Mat& mapx, const cv::Mat& mapy);
};

#endif  // __NPP_REMAPPER_H__
