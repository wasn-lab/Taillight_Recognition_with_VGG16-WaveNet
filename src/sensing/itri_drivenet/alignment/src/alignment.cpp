#include "alignment.h"

using namespace std;
using namespace pcl;
using namespace DriveNet;

void Alignment::projectMatrixInit(camera::id cam_id)
{
#if CAR_MODEL_IS_B1
  projector2_.init(cam_id);
#elif CAR_MODEL_IS_B1_V2
  projector3_.init(cam_id);
#endif
}

PixelPosition Alignment::projectPointToPixel(PointXYZI point)
{
  float x = point.x;
  float y = point.y;
  float z = point.z;
  vector<int> pixel_position_vect;
  PixelPosition pixel_position{ -1, -1 };

#if CAR_MODEL_IS_B1
  pixel_position_vect = projector2_.project(x, y, z);
#elif CAR_MODEL_IS_B1_V2
  pixel_position_vect = projector3_.project(x, y, z);
#endif
  pixel_position.u = pixel_position_vect[0];
  pixel_position.v = pixel_position_vect[1];

  if (pixel_position.u < 0 || pixel_position.u > image_w_-1)
  {
    pixel_position.u = -1;
  }
  if (pixel_position.v < 0 || pixel_position.v > image_h_-1)
  {
    pixel_position.v = -1;
  }
  return pixel_position;
}