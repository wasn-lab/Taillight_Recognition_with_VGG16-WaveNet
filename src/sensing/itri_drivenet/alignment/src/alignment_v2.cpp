#include "alignment_v2.h"

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

  if (pixel_position.u < 0 || pixel_position.u > image_w_)
  {
    pixel_position.u = -1;
  }
  if (pixel_position.v < 0 || pixel_position.v > image_h_)
  {
    pixel_position.v = -1;
  }
  return pixel_position;
}
cv::Scalar Alignment::getDistColor(float distance_in_meters)
{
  cv::Scalar color;
  if (distance_in_meters >= 0 && distance_in_meters <= 10)
  {
    color = Color::red_;
  }
  else if (distance_in_meters > 10 && distance_in_meters <= 20)
  {
    color = Color::yellow_;
  }
  else if (distance_in_meters > 20 && distance_in_meters <= 30)
  {
    color = Color::green_;
  }
  else
  {
    color = Color::blue_;
  }
  return color;
}