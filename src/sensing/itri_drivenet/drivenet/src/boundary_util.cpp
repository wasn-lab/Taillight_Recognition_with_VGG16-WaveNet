#include "drivenet/boundary_util.h"

bool CheckBoxInArea(Point RightLinePoint1, Point RightLinePoint2, Point LeftLinePoint1, Point LeftLinePoint2,
                    int object_x1, int object_y1, int object_x2, int object_y2)
{
  // printf("x1: %d, y1: %d, x2: %d, y2:%d\n", object_x1, object_y1, object_x2, object_y2);
  /// right
  int C1 = (RightLinePoint1.x - RightLinePoint2.x) * (object_y2 - RightLinePoint2.y) -
           (object_x2 - RightLinePoint2.x) * (RightLinePoint1.y - RightLinePoint2.y);
  int C2 = (RightLinePoint1.x - RightLinePoint2.x) * (object_y2 - RightLinePoint2.y) -
           (object_x1 - RightLinePoint2.x) * (RightLinePoint1.y - RightLinePoint2.y);
  /// left
  int C3 = (LeftLinePoint1.x - LeftLinePoint2.x) * (object_y2 - LeftLinePoint2.y) -
           (object_x2 - LeftLinePoint2.x) * (LeftLinePoint1.y - LeftLinePoint2.y);
  int C4 = (LeftLinePoint1.x - LeftLinePoint2.x) * (object_y2 - LeftLinePoint2.y) -
           (object_x1 - LeftLinePoint2.x) * (LeftLinePoint1.y - LeftLinePoint2.y);
  /// up
  int C5 = (RightLinePoint1.x - LeftLinePoint1.x) * (object_y2 - LeftLinePoint1.y) -
           (object_x2 - LeftLinePoint1.x) * (RightLinePoint1.y - LeftLinePoint1.y);
  int C6 = (RightLinePoint1.x - LeftLinePoint1.x) * (object_y2 - LeftLinePoint1.y) -
           (object_x1 - LeftLinePoint1.x) * (RightLinePoint1.y - LeftLinePoint1.y);
  /// bottom
  int C7 = (RightLinePoint2.x - LeftLinePoint2.x) * (object_y2 - LeftLinePoint2.y) -
           (object_x2 - LeftLinePoint2.x) * (RightLinePoint2.y - LeftLinePoint2.y);
  int C8 = (RightLinePoint2.x - LeftLinePoint2.x) * (object_y2 - LeftLinePoint2.y) -
           (object_x1 - LeftLinePoint2.x) * (RightLinePoint2.y - LeftLinePoint2.y);

  // printf("C1:%d, C3:%d, C5:%d, C7:%d\n", C1, C3, C5, C7);
  // printf("C2:%d, C4:%d, C6:%d, C8:%d\n", C2, C4, C6, C8);

  if (C1 < 0 && C3 > 0 && C5 > 0 && C7 < 0 && C2 < 0 && C4 > 0 && C6 > 0 && C8 < 0)
    return true;
  else
    return false;
}

void BoundaryMarker(int img_w, Point& BoundaryMarker1, Point& BoundaryMarker2, Point& BoundaryMarker3,
                    Point& BoundaryMarker4, int marker_h)
{
  BoundaryMarker1 = Point(img_w / 2 + 20, marker_h);
  BoundaryMarker2 = Point(img_w / 2 - 20, marker_h);
  BoundaryMarker3 = Point(img_w / 2, marker_h + 20);
  BoundaryMarker4 = Point(img_w / 2, marker_h - 20);
}