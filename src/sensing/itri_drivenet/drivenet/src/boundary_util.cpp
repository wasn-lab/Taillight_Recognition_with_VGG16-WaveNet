#include "drivenet/boundary_util.h"

using namespace cv;

bool checkBoxInArea(CheckArea areaCheck, int object_x1, int object_y1, int object_x2, int object_y2)
{
  // printf("x1: %d, y1: %d, x2: %d, y2:%d\n", object_x1, object_y1, object_x2, object_y2);
  /// right
  int C1 = (areaCheck.RightLinePoint1.x - areaCheck.RightLinePoint2.x) * (object_y2 - areaCheck.RightLinePoint2.y) -
           (object_x2 - areaCheck.RightLinePoint2.x) * (areaCheck.RightLinePoint1.y - areaCheck.RightLinePoint2.y);
  int C2 = (areaCheck.RightLinePoint1.x - areaCheck.RightLinePoint2.x) * (object_y2 - areaCheck.RightLinePoint2.y) -
           (object_x1 - areaCheck.RightLinePoint2.x) * (areaCheck.RightLinePoint1.y - areaCheck.RightLinePoint2.y);
  /// left
  int C3 = (areaCheck.LeftLinePoint1.x - areaCheck.LeftLinePoint2.x) * (object_y2 - areaCheck.LeftLinePoint2.y) -
           (object_x2 - areaCheck.LeftLinePoint2.x) * (areaCheck.LeftLinePoint1.y - areaCheck.LeftLinePoint2.y);
  int C4 = (areaCheck.LeftLinePoint1.x - areaCheck.LeftLinePoint2.x) * (object_y2 - areaCheck.LeftLinePoint2.y) -
           (object_x1 - areaCheck.LeftLinePoint2.x) * (areaCheck.LeftLinePoint1.y - areaCheck.LeftLinePoint2.y);
  /// up
  int C5 = (areaCheck.RightLinePoint1.x - areaCheck.LeftLinePoint1.x) * (object_y2 - areaCheck.LeftLinePoint1.y) -
           (object_x2 - areaCheck.LeftLinePoint1.x) * (areaCheck.RightLinePoint1.y - areaCheck.LeftLinePoint1.y);
  int C6 = (areaCheck.RightLinePoint1.x - areaCheck.LeftLinePoint1.x) * (object_y2 - areaCheck.LeftLinePoint1.y) -
           (object_x1 - areaCheck.LeftLinePoint1.x) * (areaCheck.RightLinePoint1.y - areaCheck.LeftLinePoint1.y);
  /// bottom
  int C7 = (areaCheck.RightLinePoint2.x - areaCheck.LeftLinePoint2.x) * (object_y2 - areaCheck.LeftLinePoint2.y) -
           (object_x2 - areaCheck.LeftLinePoint2.x) * (areaCheck.RightLinePoint2.y - areaCheck.LeftLinePoint2.y);
  int C8 = (areaCheck.RightLinePoint2.x - areaCheck.LeftLinePoint2.x) * (object_y2 - areaCheck.LeftLinePoint2.y) -
           (object_x1 - areaCheck.LeftLinePoint2.x) * (areaCheck.RightLinePoint2.y - areaCheck.LeftLinePoint2.y);

  // printf("C1:%d, C3:%d, C5:%d, C7:%d\n", C1, C3, C5, C7);
  // printf("C2:%d, C4:%d, C6:%d, C8:%d\n", C2, C4, C6, C8);

  if (C1 < 0 && C3 > 0 && C5 > 0 && C7 < 0 && C2 < 0 && C4 > 0 && C6 > 0 && C8 < 0)
  {
    return true;
  }
  else
  {
    return false;
}
}

template <typename T1, typename T2>
void checkValueInRange(T1& value, T2 min, T2 max)
{
  if (value < min)
  {
    value = min;
  }
  else if (value > max)
  {
    value = max;
}
}
template void checkValueInRange<float, int>(float& value, int min, int max);

void boundaryMarker(int img_w, Point& boundaryMarker1, Point& boundaryMarker2, Point& boundaryMarker3,
                    Point& boundaryMarker4, int marker_h)
{
  boundaryMarker1 = Point(img_w / 2 + 20, marker_h);
  boundaryMarker2 = Point(img_w / 2 - 20, marker_h);
  boundaryMarker3 = Point(img_w / 2, marker_h + 20);
  boundaryMarker4 = Point(img_w / 2, marker_h - 20);
}