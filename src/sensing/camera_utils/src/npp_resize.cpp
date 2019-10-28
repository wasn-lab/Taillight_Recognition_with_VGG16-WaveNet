/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */
#include "npp_resizer.h"

namespace npp_wrapper
{
int resize(const cv::Mat& src, cv::Mat& dst, const double hscale, const double wscale,
           const NppiInterpolationMode interpolation_mode)
{
  NPPResizer resizer(src.rows, src.cols, hscale, wscale);
  resizer.set_interpolation_mode(interpolation_mode);
  return resizer.resize(src, dst);
}
};  // namespace
