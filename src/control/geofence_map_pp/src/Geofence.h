
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdlib.h>

struct Point
{
  double X;
  double Y;
  double Speed;
  double Direction;
};

class Geofence
{
public:
  Geofence(double Bound);
  double getDistance();
  double getDistance_w();
  double getFarest();
  bool getTrigger();
  double getObjSpeed();
  double getNearest_X();
  double getNearest_Y();
  bool setIntersectPoint(bool state);
  Point findDirection();

  void setPointCloud(const std::vector<Point>& PointCloud, bool isLocal, double SLAM_x, double SLAM_y,
                     double Heading);                  // Update pointcloud, must set before fist execution of Calcuator
  void setPath(const std::vector<Point>& PathPoints);  // Update Path points in absolute coordibate, must set before
                                                       // fist execution of Calcuator
  int Calculator(int PP_timetick_index_ = 0, double time_threshold = 0, double vehicle_speed = 0);            
                                                       // Calculate geofence by currently set Poly and PointCloud

private:
  double dist0 = 300.;
  std::vector<Point> PathPoints;
  std::vector<double> PathLength;
  std::vector<Point> PointCloud;
  double Distance;       // Geofence distance
  double Distance_wide;  // for path planning
  double Farest;         // The farest point
  bool Trigger;          // Whether there is a object in the path
  double ObjSpeed;       // Geofence speed
  double Nearest_X;      // Nearest point's (X,Y)
  double Nearest_Y;
  double Boundary;
  bool PPAlreadyIntersected = false;
  bool PossiblePointofCollision(int PP_index, int minElementIndex, double vehicle_speed, double time_threshold);
};
