#include "alignment.h"

using namespace std;
using namespace pcl;
using namespace DriveNet;

void Alignment::projectMatrixInit(camera::id cam_id)
{
  projector3_.init(cam_id);
}

PixelPosition Alignment::projectPointToPixel(PointXYZI point)
{
  float x = point.x;
  float y = point.y;
  float z = point.z;
  vector<int> pixel_position_vect;
  PixelPosition pixel_position{ -1, -1 };

  pixel_position_vect = projector3_.project(x, y, z);

  pixel_position.u = pixel_position_vect[0];
  pixel_position.v = pixel_position_vect[1];

  if (pixel_position.u < 0 || pixel_position.u > image_w_ - 1)
  {
    pixel_position.u = -1;
  }
  if (pixel_position.v < 0 || pixel_position.v > image_h_ - 1)
  {
    pixel_position.v = -1;
  }
  return pixel_position;
}
bool Alignment::checkPointInCoverage(PointXYZI point)
{
  bool is_out_of_coverage = false;
  is_out_of_coverage = projector3_.outOfCoverage(point.x, point.y, point.z);
  return !is_out_of_coverage;
}
