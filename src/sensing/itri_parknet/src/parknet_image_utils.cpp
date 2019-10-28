/*
   CREATER: ICL U300
   DATE: May, 2019
 */
#include <assert.h>
#include <opencv2/opencv.hpp>
#include "nppi_geometry_transforms.h"
#include "npp.h"
#include "parknet_logging.h"
#include "parknet.h"
#include "parknet_camera.h"
#include "parknet_cv_colors.h"
#include "parknet_image_utils.h"
#include "camera_utils.h"
#include "camera_params.h"

#include "npp_wrapper.h"

namespace parknet
{
#if USE(DARKNET)
image convert_to_darknet_image(const cv::Mat& msg_in_cvmat)
{
  IplImage ipl_image = msg_in_cvmat;

  unsigned char* data = (unsigned char*)ipl_image.imageData;
  int h = ipl_image.height;
  int w = ipl_image.width;
  int c = ipl_image.nChannels;
  int step = ipl_image.widthStep;
  int i, j, k;

  image darknet_image = make_image(w, h, c);

  for (i = 0; i < h; ++i)
  {
    for (k = 0; k < c; ++k)
    {
      for (j = 0; j < w; ++j)
      {
        darknet_image.data[k * w * h + i * w + j] = data[i * step + j * c + k] / 255.;
      }
    }
  }
  parknet::rgbgr_image(darknet_image);
  return darknet_image;
}
#endif

void rgbgr_image(image& im)
{
  int i;
  for (i = 0; i < im.w * im.h; ++i)
  {
    float swap = im.data[i];
    im.data[i] = im.data[i + im.w * im.h * 2];
    im.data[i + im.w * im.h * 2] = swap;
  }
}

int draw_parking_slot(cv::Mat& in_img, msgs::PointXY points[4])
{
  /*
  p1     p2
   +-----+
   |     |
   |     |
   +-----+
  p0     p3
  */
  for (int i = 0; i < 4; i++)
  {
    const int di = (i + 1) % 4;
    cv::Point src, dest;
    src.x = points[i].x;
    src.y = points[i].y;
    dest.x = points[di].x;
    dest.y = points[di].y;
    cv::line(in_img, src, dest, g_color_red, /*thickness*/ 4, /*linetype */ 4);
  }
  return 0;
}
};  // namespace parknet
