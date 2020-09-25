
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

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
  struct Point findDirection();

  int setPointCloud(const vector<Point>& PointCloud, bool isLocal, double SLAM_x, double SLAM_y,
                    double Heading);             // Update pointcloud, must set before fist execution of Calcuator
  int setPath(const vector<Point>& PathPoints);  // Update Path points in absolute coordibate, must set before fist
                                                 // execution of Calcuator
  int Calculator();  // Calculate geofence by currently set Poly and PointCloud

private:
  vector<Point> PathPoints;
  vector<double> PathLength;
  vector<Point> PointCloud;
  double Distance;       // Geofence distance
  double Distance_wide;  // for path planning
  double Farest;         // The farest point
  bool Trigger;          // Whether there is a object in the path
  double ObjSpeed;       // Geofence speed
  double Nearest_X;      // Nearest point's (X,Y)
  double Nearest_Y;
  double Boundary;
};
