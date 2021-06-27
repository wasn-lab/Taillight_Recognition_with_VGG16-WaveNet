
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#define EVENT_TEST 1
#if EVENT_TEST == 1
#include <std_msgs/Header.h>
#include <geometry_msgs/Point.h>
#include <unique_id/unique_id.h>
#include <msgs/GeoFenceTest.h>
#endif

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
  Point findDirection();

  void setPointCloud(const std::vector<Point>& PointCloud, bool isLocal, double SLAM_x, double SLAM_y,
                     double Heading);                  // Update pointcloud, must set before fist execution of Calcuator
  void setPath(const std::vector<Point>& PathPoints);  // Update Path points in absolute coordibate, must set before
                                                       // fist execution of Calcuator
  int Calculator();                                    // Calculate geofence by currently set Poly and PointCloud

#if EVENT_TEST == 1
  void plotGeofenceTest(const std_msgs::Header& header, const uuid_msgs::UniqueID& id,
                        const geometry_msgs::Point& pp_point, const double pp_time);
#endif

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
};
