#include "alignment.h"

using namespace std;
using namespace pcl;

Alignment::Alignment()
{
}

Alignment::~Alignment()
{
}

void Alignment::projectMatrixInit(int camera_id)
{
  projector2_.init(camera_id);
}
PixelPosition Alignment::projectPointToPixel(PointXYZI point)
{
  float x_ = point.x;
  float y_ = point.y;
  float z_ = point.z;
  vector<int> pixel_position_vect_;
  PixelPosition pixel_position_;

  pixel_position_vect_ = projector2_.project(x_, y_, z_);
  pixel_position_.u = pixel_position_vect_[0];
  pixel_position_.v = pixel_position_vect_[1];

  if (pixel_position_.u < 0 || pixel_position_.u > image_w_)
  {
    pixel_position_.u = -1;
  }
  if (pixel_position_.v < 0 || pixel_position_.v > image_h_)
  {
    pixel_position_.v = -1;
  }
  return pixel_position_;
}