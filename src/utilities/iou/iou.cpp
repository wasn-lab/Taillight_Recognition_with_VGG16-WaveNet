/*
 * iou.cpp
 *
 * Calculate intesection over union (IOU) of two 2D bounding boxes
 *
 * Author: Ching-Hao Liu
 *
 * Nov 26, 2019
 */

#include <iostream>

#define DEBUG 1

struct Box2D
{
  int x;
  int y;
  int w;
  int h;

  // This constructor does no initialization.
  Box2D()
  {
  }

  // This constructor initializes the four variables individually.
  Box2D(int x, int y, int w, int h) : x(x), y(y), w(w), h(h)
  {
  }

  // This constructor initializes all four variables to the same value
  Box2D(int i) : x(i), y(i), w(i), h(i)
  {
  }
};

bool inner_pair(int& a1, int& a2, int p1, int p2, int q1, int q2)
{
  // ensure p1 <= p2
  if (p1 > p2)
  {
    int tmp = p1;
    p1 = p2;
    p2 = tmp;
  }

  // ensure q1 <= q2
  if (q1 > q2)
  {
    int tmp = q1;
    q1 = q2;
    q2 = tmp;
  }

  if (q1 > p2 || q2 < p1)
    return false;

  a1 = (q1 >= p1) ? q1 : p1;
  a2 = (q2 <= p2) ? q2 : p2;

  return true;
}

bool box2d_intersection(Box2D& box_ints, const Box2D b1, const Box2D b2)
{
  int ints_x1 = -1;
  int ints_x2 = -1;
  int b1_x2 = b1.x + b1.w - 1;
  int b2_x2 = b2.x + b2.w - 1;
  bool is_x_inner = inner_pair(ints_x1, ints_x2, b1.x, b1_x2, b2.x, b2_x2);

  int ints_y1 = -1;
  int ints_y2 = -1;
  int b1_y2 = b1.y + b1.h - 1;
  int b2_y2 = b2.y + b2.h - 1;
  bool is_y_inner = inner_pair(ints_y1, ints_y2, b1.y, b1_y2, b2.y, b2_y2);

  if (is_x_inner && is_y_inner)
  {
    box_ints.x = ints_x1;
    box_ints.y = ints_y1;
    box_ints.w = ints_x2 - ints_x1 + 1;
    box_ints.h = ints_y2 - ints_y1 + 1;
    return true;
  }

  return false;
}

void iou(double& iou, const Box2D b1, const Box2D b2)
{
  Box2D box_ints(-1);

  bool is_ints = box2d_intersection(box_ints, b1, b2);

  int b1_area = b1.w * b1.h;
  int b2_area = b2.w * b2.h;

  int box_ints_area = (is_ints) ? box_ints.w * box_ints.h : 0;
  int box_union_area = b1_area + b2_area - box_ints_area;

  iou = box_ints_area / (double)box_union_area;

#if DEBUG
  std::cout << "b1_area=" << b1_area << " b2_area=" << b2_area << std::endl;
  std::cout << "is_ints=" << is_ints << " x=" << box_ints.x << " y=" << box_ints.y << " w=" << box_ints.w
            << " h=" << box_ints.h << " box_ints_area=" << box_ints_area << std::endl;
  std::cout << "box_ints_area=" << box_ints_area << " box_union_area=" << box_union_area << " iou=" << iou << std::endl;
#endif
}

int main()
{
  Box2D b1(1, 1, 5, 5);
  Box2D b2(3, 3, 5, 5);
  double iou_val = 0.;

  iou(iou_val, b1, b2);
}
