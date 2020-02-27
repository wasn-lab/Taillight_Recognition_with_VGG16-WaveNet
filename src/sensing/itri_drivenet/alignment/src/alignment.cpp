#include "alignment.h"

using namespace std;
using namespace pcl;
using namespace DriveNet;

Alignment::Alignment()
{
}

Alignment::~Alignment()
{
}

void Alignment::projectMatrixInit(camera::id camera_id)
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
cv::Scalar Alignment::getDistColor(float distance)
{
  cv::Scalar color;
  if (distance >= 0 && distance <= 10)
  {
    color = Color::g_color_red;
  }
  else if (distance > 10 && distance <= 20)
  {
    color = Color::g_color_yellow;
  }
  else if (distance > 20 && distance <= 30)
  {
    color = Color::g_color_green;
  }
  else
  {
    color = Color::g_color_blue;
  }
  return color;
}