#include "visualization_util.h"

/// stardard
#include <cmath>

using namespace DriveNet;

void Visualization::drawPointCloudOnImage(cv::Mat& m_src, int point_u, int point_v, float point_x)
{
  cv::Point center_point = cv::Point(point_u, point_v);
  float distance_x = point_x;
  cv::Scalar point_color = getDistColor(distance_x);
  cv::circle(m_src, center_point, 1, point_color, -1, cv::FILLED, 0);
}

void Visualization::drawPointCloudOnImage(cv::Mat& m_src, int point_u, int point_v, int index)
{
  cv::Point center_point = cv::Point(point_u, point_v);
  cv::Scalar point_color = intToColor(index);
  cv::circle(m_src, center_point, 1, point_color, -1, cv::FILLED, 0);
}

void Visualization::drawBoxOnImage(cv::Mat& m_src, std::vector<msgs::DetectedObject>& objects)
{
  std::vector<cv::Point> cv_points(2);
  std::vector<PixelPosition> pixel_positions(2);
  int obj_count = 0;
  cv::Scalar color = CvColor::blue_;
  for (const auto& obj : objects)
  {
    for(uint i = 0; i < obj.camInfo.size(); i++)
    {
      pixel_positions[0].u = obj.camInfo[i].u;
      pixel_positions[0].v = obj.camInfo[i].v;
      pixel_positions[1].u = obj.camInfo[i].u + obj.camInfo[i].width;
      pixel_positions[1].v = obj.camInfo[i].v + obj.camInfo[i].height;

      transferPixelScaling(pixel_positions);
      cv_points[0].x = pixel_positions[0].u;
      cv_points[0].y = pixel_positions[0].v;
      cv_points[1].x = pixel_positions[1].u;
      cv_points[1].y = pixel_positions[1].v;

      // color = intToColor(int(obj_count % 10));
      cv::rectangle(m_src, cv_points[0], cv_points[1], color, 1, cv::LINE_8);
      obj_count++;
    }
  }
}
void Visualization::drawBoxOnImage(cv::Mat& m_src, std::vector<MinMax2D>& min_max_2d_bbox)
{
  std::vector<cv::Point> cv_points(2);
  int obj_count = 0;
  cv::Scalar color = CvColor::green_;
  for (const auto& bbox : min_max_2d_bbox)
  {
    cv_points[0].x = bbox.p_min.u;
    cv_points[0].y = bbox.p_min.v;
    cv_points[1].x = bbox.p_max.u;
    cv_points[1].y = bbox.p_max.v;

    // color = intToColor(int(obj_count % 10));
    cv::rectangle(m_src, cv_points[0], cv_points[1], color, 1, cv::LINE_8);
    obj_count++;
  }
}
void Visualization::drawCubeOnImage(cv::Mat& m_src, std::vector<std::vector<PixelPosition>>& cube_2d_bbox)
{
  /* ABB order
  | pt5 _______pt6(max)
  |     |\      \
  |     | \      \
  |     |  \______\
  | pt4 \  |pt1   |pt2
  |      \ |      |
  |       \|______|
  | pt0(min)    pt3
  |------------------------>X
  */
  std::vector<cv::Point> cv_points(12);
  // int obj_count = 0;
  cv::Scalar color = CvColor::green_;
  for (const auto& bbox : cube_2d_bbox)
  {
    cv_points[0].x = bbox[0].u;
    cv_points[0].y = bbox[0].v;
    cv_points[1].x = bbox[7].u;
    cv_points[1].y = bbox[7].v;

    cv_points[2].x = bbox[1].u;
    cv_points[2].y = bbox[1].v;
    cv_points[3].x = bbox[6].u;
    cv_points[3].y = bbox[6].v;

    cv_points[4].x = bbox[1].u;
    cv_points[4].y = bbox[1].v;
    cv_points[5].x = bbox[0].u;
    cv_points[5].y = bbox[0].v;

    cv_points[6].x = bbox[5].u;
    cv_points[6].y = bbox[5].v;
    cv_points[7].x = bbox[4].u;
    cv_points[7].y = bbox[4].v;

    cv_points[8].x = bbox[2].u;
    cv_points[8].y = bbox[2].v;
    cv_points[9].x = bbox[3].u;
    cv_points[9].y = bbox[3].v;

    cv_points[10].x = bbox[6].u;
    cv_points[10].y = bbox[6].v;
    cv_points[11].x = bbox[7].u;
    cv_points[11].y = bbox[7].v;

    // color = intToColor(int(obj_count % 10));
    if ((cv_points[0].x != -1 && cv_points[0].y != -1) && (cv_points[1].x != -1 && cv_points[1].y != -1))
    {
      cv::rectangle(m_src, cv_points[0], cv_points[1], color, 1, cv::LINE_8); // top
    }
    if ((cv_points[2].x != -1 && cv_points[2].y != -1) && (cv_points[3].x != -1 && cv_points[3].y != -1))
    {
      cv::rectangle(m_src, cv_points[2], cv_points[3], color, 1, cv::LINE_8); // bottom
    }
    if ((cv_points[4].x != -1 && cv_points[4].y != -1) && (cv_points[5].x != -1 && cv_points[5].y != -1))
    {
      cv::line(m_src, cv_points[4], cv_points[5], color, 1, cv::LINE_8);
    }
    if ((cv_points[6].x != -1 && cv_points[6].y != -1) && (cv_points[7].x != -1 && cv_points[7].y != -1))
    {
      cv::line(m_src, cv_points[6], cv_points[7], color, 1, cv::LINE_8);
    }
    if ((cv_points[8].x != -1 && cv_points[8].y != -1) && (cv_points[9].x != -1 && cv_points[9].y != -1))
    {
      cv::line(m_src, cv_points[8], cv_points[9], color, 1, cv::LINE_8);
    }
    if ((cv_points[10].x != -1 && cv_points[10].y != -1) && (cv_points[11].x != -1 && cv_points[11].y != -1))
    {
      cv::line(m_src, cv_points[10], cv_points[11], color, 1, cv::LINE_8);
    }
    // cv::circle(m_src, cv_points[0], 1, color, -1, cv::FILLED, 0);
    // cv::circle(m_src, cv_points[1], 1, color, -1, cv::FILLED, 0);
    // cv::circle(m_src, cv_points[2], 1, color, -1, cv::FILLED, 0);
    // cv::circle(m_src, cv_points[3], 1, color, -1, cv::FILLED, 0);
    // cv::circle(m_src, cv_points[4], 1, color, -1, cv::FILLED, 0);
    // cv::circle(m_src, cv_points[5], 1, color, -1, cv::FILLED, 0);
    // cv::circle(m_src, cv_points[6], 1, color, -1, cv::FILLED, 0);
    // cv::circle(m_src, cv_points[7], 1, color, -1, cv::FILLED, 0);

    // std::cout << "=======================" << std::endl;
    // std::cout << "cv_points[0]: " << cv_points[0].x << ", " << cv_points[0].y << std::endl;
    // std::cout << "cv_points[1]: " << cv_points[1].x << ", " << cv_points[1].y << std::endl;
    // std::cout << "cv_points[2]: " << cv_points[2].x << ", " << cv_points[2].y << std::endl;
    // std::cout << "cv_points[3]: " << cv_points[3].x << ", " << cv_points[3].y << std::endl;
    // std::cout << "cv_points[4]: " << cv_points[4].x << ", " << cv_points[4].y << std::endl;
    // std::cout << "cv_points[5]: " << cv_points[5].x << ", " <<  cv_points[5].y << std::endl;
    // std::cout << "cv_points[6]: " << cv_points[6].x << ", " <<  cv_points[6].y << std::endl;
    // std::cout << "cv_points[7]: " << cv_points[7].x << ", " <<  cv_points[7].y << std::endl;
    // std::cout << "cv_points[8]: " << cv_points[8].x << ", " <<  cv_points[8].y << std::endl;
    // std::cout << "cv_points[9]: " << cv_points[9].x << ", " <<  cv_points[9].y << std::endl;
    // std::cout << "cv_points[10]: " << cv_points[10].x << ", " <<  cv_points[10].y << std::endl;
    // std::cout << "cv_points[11]: " << cv_points[11].x << ", " <<  cv_points[11].y << std::endl;
    // obj_count++;
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
    color = CvColor::black_;
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