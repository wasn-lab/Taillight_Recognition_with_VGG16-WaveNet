/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */
#include <cassert>
#include <memory>
#include "npp_wrapper.h"
#include "nppi_geometry_transforms.h"
#include "npp.h"
#include "npp_remapper.h"

namespace npp_wrapper
{
int remap(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mapx, const cv::Mat& mapy,
          const NppiInterpolationMode interpolation_mode)
{
  NPPRemapper remapper(src.rows, src.cols);
  remapper.set_mapxy(mapx, mapy);
  return remapper.remap(src, dst);
}

};  // namespace
