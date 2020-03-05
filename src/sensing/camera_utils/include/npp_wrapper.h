/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#ifndef __NPP_WRAPPER_H__
#define __NPP_WRAPPER_H__

#include <opencv2/core/mat.hpp>
#include <nppdefs.h>

namespace npp_wrapper
{
int remap(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mapx, const cv::Mat& mapy,
          const NppiInterpolationMode interpolation_mode = NPPI_INTER_LINEAR);
int resize(const cv::Mat& src, cv::Mat& dst, const double hscale, const double wscale,
           const NppiInterpolationMode interpolation_mode = NPPI_INTER_LINEAR);
int rotate(const cv::Mat& src, cv::Mat& dst, const int rotation_degree,
           const NppiInterpolationMode interpolation_mode = NPPI_INTER_LINEAR);
}  // namespace npp_wrapper

#endif  // __NPP_WRAPPER_H__
