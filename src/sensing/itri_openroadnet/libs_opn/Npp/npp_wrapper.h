/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#ifndef __NPP_WRAPPER_H__
#define __NPP_WRAPPER_H__

#include <opencv2/core/mat.hpp>
#include <nppdefs.h>

// Npp8u* dst_npp8u_ptr;
// Npp8u* dst_npp8u_ptr_;
// Npp8u* src_npp8u_ptr;
// Npp8u* src_npp8u_ptr_;

namespace npp_wrapper
{
class npp
{
private:
  Npp8u* dst_npp8u_ptr;
  Npp8u* dst_npp8u_ptr_;
  Npp8u* src_npp8u_ptr;
  Npp8u* src_npp8u_ptr_;
  int RGBOrder[3];

public:
  npp();
  ~npp();

  void init_1(const int ori_cols, const int ori_rows, const double wscale, const double hscale);

  void init_3(const int ori_cols, const int ori_rows, const double wscale, const double hscale);

  int resize1(const cv::Mat& src, cv::Mat& dst, const double hscale, const double wscale,
              const NppiInterpolationMode interpolation_mode = NPPI_INTER_LINEAR);

  int resize3(const Npp8u* rawCUDA_, cv::Mat& dst, const double hscale, const double wscale,
              const NppiInterpolationMode interpolation_mode = NPPI_INTER_LINEAR);

  int resize3(const cv::Mat& src, cv::Mat& dst, const double hscale, const double wscale,
              const NppiInterpolationMode interpolation_mode = NPPI_INTER_LINEAR);
};

}  // namespace

#endif  // __NPP_WRAPPER_H__
