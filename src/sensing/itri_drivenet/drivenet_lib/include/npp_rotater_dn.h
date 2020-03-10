/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#ifndef __NPP_ROTATER_DN_H__
#define __NPP_ROTATER_DN_H__

#include <memory>
#include <opencv2/core/mat.hpp>
#include <nppdefs.h>
#include <cassert>

namespace DriveNet_npp
{
class NPPRotaterImpl;
class NPPRotater
{
private:
  std::unique_ptr<NPPRotaterImpl> rotater_impl_;

public:
  NPPRotater(const int src_rows, const int src_cols, const int rotation_degree);
  NPPRotater(const int src_rows, const int src_cols, const int dst_rows, const int dst_cols);
  ~NPPRotater();
  int rotate(const cv::Mat& src, cv::Mat& dst, const int rotation_degree);
  void set_interpolation_mode(NppiInterpolationMode mode);
};
}
#endif  // __NPP_ROTATER_DN_H__
