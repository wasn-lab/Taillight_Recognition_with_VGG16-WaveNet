#ifndef BOUNDARYUTIL_H_
#define BOUNDARYUTIL_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

bool CheckBoxInArea(Point RightLinePoint1, Point RightLinePoint2, Point LeftLinePoint1, Point LeftLinePoint2,
                    int object_x1, int object_y1, int object_x2, int object_y2);
void BoundaryMarker(int img_w, Point& BoundaryMarker1, Point& BoundaryMarker2, Point& BoundaryMarker3,
                    Point& BoundaryMarker4, int marker_h);

#endif /*BOUNDARYUTIL_H_*/