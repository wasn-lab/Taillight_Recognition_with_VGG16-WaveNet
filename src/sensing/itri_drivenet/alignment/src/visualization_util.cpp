#include "visualization_util.h"

/// stardard
#include <cmath>

using namespace DriveNet;

void Visualization::drawPointCloudOnImage(cv::Mat& m_src, int point_u, int point_v, float point_x)
{
  cv::Point center_point = cv::Point(point_u, point_v);
  float distance_x = point_x;
  cv::Scalar point_color = getDistColor(distance_x);
  cv::circle(m_src, center_point, 1, point_color, -1, cv::LINE_8, 0);
}

void Visualization::drawBoxOnImage(cv::Mat& m_src, std::vector<msgs::DetectedObject>& objects)
{
  std::vector<cv::Point> cv_points(2);
  std::vector<PixelPosition> pixel_positions(2);
  for (const auto& obj : objects)
  {
    pixel_positions[0].u = obj.camInfo.u;
    pixel_positions[0].v = obj.camInfo.v;
    pixel_positions[1].u = obj.camInfo.u + obj.camInfo.width;
    pixel_positions[1].v = obj.camInfo.v + obj.camInfo.height;

    transferPixelScaling(pixel_positions);
    cv_points[0].x = pixel_positions[0].u;
    cv_points[0].y = pixel_positions[0].v;
    cv_points[1].x = pixel_positions[1].u;
    cv_points[1].y = pixel_positions[1].v;
    cv::rectangle(m_src, cv_points[0], cv_points[1], CvColor::white_, 1, cv::LINE_8);
  }
}

cv::Scalar Visualization::getDistColor(float distance_in_meters)
{
  cv::Scalar color;

  distance_in_meters = std::fabs(distance_in_meters);
  if (distance_in_meters > 0 && distance_in_meters <= 10)
  {
    color = CvColor::red_;
  }
  else if (distance_in_meters > 10 && distance_in_meters <= 20)
  {
    color = CvColor::yellow_;
  }
  else if (distance_in_meters > 20 && distance_in_meters <= 30)
  {
    color = CvColor::green_;
  }
  else if (distance_in_meters > 30 && distance_in_meters <= 40)
  {
    color = CvColor::blue_;
  }
  else if (distance_in_meters > 40 && distance_in_meters <= 50)
  {
    color = CvColor::purple_;
  }
  else
  {
    color = CvColor::white_;
  }
  return color;
}

MinMax3D Visualization::getDistLinePoint(float x_dist, float y_dist, float z_dist)
{
  MinMax3D point;
  point.p_min.x = x_dist;
  point.p_min.y = (-1) * y_dist;
  point.p_min.z = z_dist;
  point.p_max.x = x_dist;
  point.p_max.y = y_dist;
  point.p_max.z = z_dist;

  return point;
}