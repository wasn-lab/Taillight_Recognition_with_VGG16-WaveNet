/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#ifndef __NPP_WRAPPER_IMPL_H__
#define __NPP_WRAPPER_IMPL_H__

#include <opencv2/core/mat.hpp>
#include <nppdefs.h>
#include <mutex>

class NPPRemapperImpl
{
private:
  const int rows_, cols_, num_pixels_, num_image_bytes_;
  const int image_line_steps_;
  const int mapxy_line_steps_;
  cv::Mat mapx_, mapy_;
  Npp32f* mapx_ptr_cuda_;
  Npp32f* mapy_ptr_cuda_;
  Npp8u* src_npp8u_ptr_cuda_;
  Npp8u* dst_npp8u_ptr_cuda_;

  NppiSize image_size_;
  NppiRect image_roi_;
  NppiInterpolationMode interpolation_mode_;
  bool is_mapxy_initialized_;
  std::mutex mu_;

public:
  NPPRemapperImpl(const int rows, const int cols);
  ~NPPRemapperImpl();
  int remap(const cv::Mat& src, cv::Mat& dst);
  int remap(const Npp8u* npp8u_ptr_in, cv::Mat& dst);
  int remap(const Npp8u* npp8u_ptr_in, Npp8u* npp8u_ptr_out);
  void set_interpolation_mode(NppiInterpolationMode mode);
  int set_mapxy(const cv::Mat& mapx, const cv::Mat& mapy);
  int copy_to(Npp8u* npp8u_ptr);
};

#endif  // __NPP_WRAPPER_IMPL_H__
