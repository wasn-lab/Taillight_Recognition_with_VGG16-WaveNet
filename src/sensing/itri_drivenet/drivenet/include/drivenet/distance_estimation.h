#ifndef DISTANCEESTIMATION_H_
#define DISTANCEESTIMATION_H_

// ROS message
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

struct DisEstiParams
{
  std::vector<int> regionHeight_x;
  std::vector<float> regionDist_x;
  std::vector<int> regionHeight_y;
  std::vector<float> regionDist_y;
  std::vector<float> regionHeightSlope_x;
  std::vector<float> regionHeightSlope_y;
};

struct CheckArea
{
  cv::Point LeftLinePoint1;
  cv::Point LeftLinePoint2;
  cv::Point RightLinePoint1;
  cv::Point RightLinePoint2;
};

#endif /*DISTANCEESTIMATION_H_*/
