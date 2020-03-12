#include "visualization_util.h"

using namespace DriveNet;

void Visualization::drawPointCloudOnImage(cv::Mat& m_src, int point_u, int point_v, float point_x)
{
  cv::Point center_point = cv::Point(point_u, point_v);
  float distance_x = point_x;
  cv::Scalar point_color = getDistColor(distance_x);
  cv::circle(m_src, center_point, 1, point_color, -1, cv::LINE_8, 0);
}

void Visualization::drawBoxOnImage(cv::Mat& m_src, std::vector<msgs::DetectedObject> objects)
{
  std::vector<cv::Point> cvPoints(2);
  std::vector<PixelPosition> pixelPositions(2);
  for (const auto& obj: objects)
  {
    pixelPositions[0].u = obj.camInfo.u;
    pixelPositions[0].v = obj.camInfo.v;
    pixelPositions[1].u = obj.camInfo.u + obj.camInfo.width;
    pixelPositions[1].v = obj.camInfo.v + obj.camInfo.height;

    cvPoints[0].x = int(pixelPositions[0].u * scaling_ratio_w_);
    cvPoints[0].y = int(pixelPositions[0].v * scaling_ratio_h_);
    cvPoints[1].x = int(pixelPositions[1].u * scaling_ratio_w_);
    cvPoints[1].y = int(pixelPositions[1].v * scaling_ratio_h_);
    cv::rectangle(m_src, cvPoints[0], cvPoints[1], cv::Scalar(255, 255, 255), 1, cv::LINE_8);
  }
}

cv::Scalar Visualization::getDistColor(float distance_in_meters)
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
