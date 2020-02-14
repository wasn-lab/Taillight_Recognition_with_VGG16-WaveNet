#ifndef ALIGNMENT_H_
#define ALIGNMENT_H_

/// pcl
#include <pcl/point_types.h>

/// projection
#include <projection/projector2.h>

/// util
#include <camera_params.h>

struct PixelPosition
{
  int u;
  int v;
};

class Alignment
{
private:
  Projector2 projector2_;
  int image_w_ = camera::image_width;
  int image_h_ = camera::image_height;

public:
  Alignment();
  ~Alignment();
  void projectMatrixInit(int cam_id);
  PixelPosition projectPointToPixel(pcl::PointXYZI point);
};

#endif
