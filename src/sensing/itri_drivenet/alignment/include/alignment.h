#ifndef ALIGNMENT_H_
#define ALIGNMENT_H_

/// util
#include "car_model.h"
#include <camera_params.h>
#include "drivenet/image_preprocessing.h"

/// pcl
#include <pcl/point_types.h>

/// projection
#if CAR_MODEL_IS_B1_V2 || CAR_MODEL_IS_B1_V3 || CAR_MODEL_IS_C1 || CAR_MODEL_IS_C2
#include <projection/projector3.h>
#else
#error "car model is not well defined"
#endif

class Alignment
{
private:
  Projector3 projector3_;

  int image_w_ = camera::image_width;
  int image_h_ = camera::image_height;

public:
  Alignment() = default;
  ~Alignment() = default;
  void projectMatrixInit(camera::id cam_id);
  DriveNet::PixelPosition projectPointToPixel(pcl::PointXYZI point);
  bool checkPointInCoverage(pcl::PointXYZI point);
};

#endif
