/*
   CREATER: ICL U300
   DATE: May, 2019
 */

#ifndef __PARKNET_IMAGE_UTILS_H__
#define __PARKNET_IMAGE_UTILS_H__

#include "msgs/PointXY.h"
#include "opencv2/core/mat.hpp"
extern "C" {
#undef __cplusplus
#include "darknet.h"
#define __cplusplus
}
#include "npp_resizer.h"

namespace parknet
{
#if USE(DARKNET)
image convert_to_darknet_image(const cv::Mat& msg);
#endif
void rgbgr_image(image& im);
int draw_parking_slot(cv::Mat& in_img, msgs::PointXY points[4]);
}  // namespace

#endif  // __PARKNET_IMAGE_UTILS_H__
