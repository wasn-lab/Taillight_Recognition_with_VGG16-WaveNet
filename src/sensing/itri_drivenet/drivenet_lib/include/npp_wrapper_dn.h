/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#ifndef __NPP_WRAPPER_DN_H__
#define __NPP_WRAPPER_DN_H__

#include <opencv2/core/mat.hpp>
#include <nppdefs.h>

namespace DriveNet_npp
{
int resize(const cv::Mat& src, cv::Mat& dst, const double hscale, const double wscale, const int channel,
           const NppiInterpolationMode interpolation_mode = NPPI_INTER_LINEAR);
int rotate(const cv::Mat& src, cv::Mat& dst, const int rotation_degree,
           const NppiInterpolationMode interpolation_mode = NPPI_INTER_LINEAR);
int cvmat_to_npp8u_ptr(const cv::Mat& src, Npp8u* out_npp8u_ptr);
int npp8u_ptr_to_cvmat(const Npp8u* in_npp8u_ptr, const size_t in_num_bytes, cv::Mat& out_img, const int rows,
                       const int cols);
}  // namespace

#endif  // __NPP_WRAPPER_DN_H__
