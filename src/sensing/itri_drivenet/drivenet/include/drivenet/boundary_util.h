#ifndef BOUNDARYUTIL_H_
#define BOUNDARYUTIL_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

bool CheckBoxInArea(cv::Point RightLinePoint1, cv::Point RightLinePoint2, cv::Point LeftLinePoint1, cv::Point LeftLinePoint2, int object_x1, int object_y1, int object_x2, int object_y2);

#endif /*BOUNDARYUTIL_H_*/