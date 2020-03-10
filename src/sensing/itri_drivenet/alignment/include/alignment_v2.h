#ifndef ALIGNMENT_B1_V2_H_
#define ALIGNMENT_B1_V2_H_

/// pcl
#include <pcl/point_types.h>

/// projection
#include <projection/projector3.h>

/// util
#include <camera_params.h>
#include "drivenet/image_preprocessing.h"
#include <string>


struct PixelPosition
{
  int u;
  int v;
};

class Alignment
{
private:
  Projector3 projector3_;
  int image_w_ = camera::image_width;
  int image_h_ = camera::image_height;

public:
  Alignment();
  ~Alignment();
  void projectMatrixInit(camera::id cam_id);
  PixelPosition projectPointToPixel(pcl::PointXYZI point);
  cv::Scalar getDistColor(float distance);
};

#endif
