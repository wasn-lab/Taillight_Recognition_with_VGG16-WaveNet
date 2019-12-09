#ifndef BOUNDARYUTIL_H_
#define BOUNDARYUTIL_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "drivenet/distance_estimation_b1.h"

using namespace cv;

bool checkBoxInArea(CheckArea areaCheck, int object_x1, int object_y1, int object_x2, int object_y2);
template <typename T1, typename T2>
extern void checkValueInRange(T1& value, T2 min, T2 max);
void boundaryMarker(int img_w, Point& boundaryMarker1, Point& boundaryMarker2, Point& boundaryMarker3,
                    Point& boundaryMarker4, int marker_h);

#endif /*BOUNDARYUTIL_H_*/