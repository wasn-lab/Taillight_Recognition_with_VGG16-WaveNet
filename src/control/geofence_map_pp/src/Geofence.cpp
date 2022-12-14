#include "Geofence.h"

//#define BOUNDARY 1.2
//#define DEBUG
//#define TEST

Geofence::Geofence(double Bound)
{
  std::cout << "Geofence is being created, Boundary = " << Bound << std::endl;
  Boundary = Bound;
}

double Geofence::getDistance()
{
  return Distance;
}
double Geofence::getDistance_w()
{
  return Distance_wide;
}
double Geofence::getFarest()
{
  return Farest;
}
bool Geofence::getTrigger()
{
  return Trigger;
}
double Geofence::getObjSpeed()
{
  return ObjSpeed;
}
double Geofence::getNearest_X()
{
  return Nearest_X;
}
double Geofence::getNearest_Y()
{
  return Nearest_Y;
}
bool Geofence::setIntersectPoint(bool state)
{ 
  PPAlreadyIntersected = state;
  return PPAlreadyIntersected;
}
void Geofence::setObjectWidth(double obj_width)
{
  ObjWidth = obj_width;
}

Point Geofence::findDirection()
{
  Point p;
  double X_bar = 1000;
  double Y_bar = 1000;

  for (size_t i = 1; i < this->PathLength.size(); i++)
  {
    if (this->PathLength[i] > this->Distance)
    {
      X_bar = this->PathPoints[i].X - this->PathPoints[i - 1].X;
      Y_bar = this->PathPoints[i].Y - this->PathPoints[i - 1].Y;
      p.X = this->PathPoints[i].X;
      p.Y = this->PathPoints[i].Y;
      break;
    }
  }

  p.Direction = atan2(Y_bar, X_bar);

  return p;
}

void Geofence::setPath(const std::vector<Point>& PathPoints)
{
  std::vector<Point>().swap(this->PathPoints);
  this->PathPoints.assign(PathPoints.begin(), PathPoints.end());

  std::vector<double>().swap(this->PathLength);

  for (size_t i = 0; i < PathPoints.size(); i++)
  {
    double Segment;
    double Length_temp;

    if (i == 0)
    {
      Length_temp = 0.0;  // For first element
    }
    else
    {
      Segment = sqrt(pow((PathPoints[i].X - PathPoints[i - 1].X), 2) + pow((PathPoints[i].Y - PathPoints[i - 1].Y), 2));
      Length_temp = PathLength[i - 1] + Segment;
    }

    this->PathLength.push_back(Length_temp);
  }
}

void Geofence::setPointCloud(const std::vector<Point>& PointCloud, bool isLocal, double SLAM_x, double SLAM_y,
                             double Heading)
{
  std::vector<Point>().swap(this->PointCloud);
  this->PointCloud.reserve(PointCloud.size());

  Point p;
  if (isLocal)
  {
    for (size_t i = 0; i < PointCloud.size(); i++)
    {
      p.X = cos(Heading) * PointCloud[i].X - sin(Heading) * PointCloud[i].Y + SLAM_x;
      p.Y = sin(Heading) * PointCloud[i].X + cos(Heading) * PointCloud[i].Y + SLAM_y;
      p.Speed = PointCloud[i].Speed;
      this->PointCloud.push_back(p);
    }
  }
  else
  {
    this->PointCloud = PointCloud;
  }
#ifdef TEST
  cout << "Size of PoinClout:" << PointCloud.size() << endl;
#endif
}

int Geofence::Calculator(int PP_timetick_index_, double time_threshold, double vehicle_speed_)
{
  // Check if all information is initialized
  if (PathPoints.size() < 1)
  {
    std::cerr << "Path not initialized" << std::endl;
    return 1;
  }

  if (PointCloud.size() < 1)
  {
    std::cerr << "PointCloud not initialized" << std::endl;
    return 1;
  }

  std::vector<double> P_Distance(PointCloud.size(), dist0);  // Distance of every pointcloud (default 100)
  std::vector<double> P_Distance_w(PointCloud.size(),
                                   dist0);  // Distance of every pointcloud in wider range (default 100)

  for (size_t i = 0; i < PointCloud.size(); i++)
  {
    std::vector<double> V_Distance(PathPoints.size(), dist0);  // Vertical diatnce to the path

    for (size_t j = 0; j < PathPoints.size(); j++)
    {
      V_Distance[j] = sqrt(pow(PointCloud[i].X - PathPoints[j].X, 2) + pow(PointCloud[i].Y - PathPoints[j].Y, 2)) - ObjWidth/2;
    }

    std::cout<<ObjWidth<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl; //for debug

    int minElementIndex = std::min_element(V_Distance.begin(), V_Distance.end()) - V_Distance.begin();
    double minElement = *std::min_element(V_Distance.begin(), V_Distance.end());

    if (minElement < Boundary && !PPAlreadyIntersected)
    { 
      PPAlreadyIntersected = true;
      if (PossiblePointofCollision(PP_timetick_index_, minElementIndex, vehicle_speed_, time_threshold))
      {
        P_Distance[i] = PathLength[minElementIndex];
      }
    }

    if (minElement < Boundary && !PPAlreadyIntersected)
    {
      PPAlreadyIntersected = true;
      if (PossiblePointofCollision(PP_timetick_index_, minElementIndex, vehicle_speed_, time_threshold))
      {
        P_Distance_w[i] = PathLength[minElementIndex];
      }
    }
  }

  int minElementIndex = std::min_element(P_Distance.begin(), P_Distance.end()) - P_Distance.begin();
  double minElement = *std::min_element(P_Distance.begin(), P_Distance.end());

  if (minElement < 99)
  {
    Trigger = true;
    ObjSpeed = PointCloud[minElementIndex].Speed;
    Nearest_X = PointCloud[minElementIndex].X;
    Nearest_Y = PointCloud[minElementIndex].Y;
  }
  else
  {
    Trigger = false;
    ObjSpeed = 0;
    Nearest_X = 3000;
    Nearest_Y = 3000;
  }

  Distance = minElement;  // key parameter to decide plot PP or not

  minElement = *std::min_element(P_Distance_w.begin(), P_Distance_w.end());
  Distance_wide = minElement;

  // Calculate farest point
  for (size_t i = 0; i < P_Distance.size(); i++)
  {
    if (P_Distance[i] >= dist0)
    {
      P_Distance[i] = -100;
    }
  }

  Farest = *std::max_element(P_Distance.begin(), P_Distance.end());

  return 0;
}

bool Geofence::PossiblePointofCollision(int PP_timetick_index_, int minElementIndex, double vehicle_speed_, double time_threshold)
{ 
  if (vehicle_speed_ == 0 && time_threshold == 0)
  { 
    std::cout << "Non-mapPP Geofence" << std::endl; 
    return true;
  }

  vehicle_dist_to_geofence = PathLength[minElementIndex] - 10;
  vehicle_speed   = vehicle_speed_;
  vehicle_time    = vehicle_dist_to_geofence / vehicle_speed_; 
  object_time     = 0.5 * PP_timetick_index_;
  time_difference = object_time - vehicle_time;
  filter_state    = 0;

  std::cout << "Dist_to_geofence = " << vehicle_dist_to_geofence << " Vehicle_speed = " << vehicle_speed << std::endl;
  std::cout << "Object_time = "      << object_time              << " Vehicle_time = "  << vehicle_time 
            << " Time_difference--------> " << time_difference << std::endl;

  //use narrow threshold (1/2) if the ego-vehicle arrives earlier than the object
  if (-time_threshold < time_difference && time_difference < time_threshold/2) 
  { 
    if (vehicle_time <= abs(time_difference))
    { 
      filter_state = 2;
      std::cout << "G e o f e n c e ********************* f i l t e r e d " << std::endl; //for debug
      return false;
    }
    else 
    { 
      filter_state = 0;
      std::cout << "G e o f e n c e ===================== r e m a i n e d" << std::endl; //for debug
      return true;
    }
  }
  else 
  { 
    filter_state = 1;
    std::cout << "G e o f e n c e ********************* f i l t e r e d " << std::endl; //for debug
    return false;
  }
}

void Geofence::getSpeedTimeInfo(std::vector<double>& speed_time_info)
{
  speed_time_info[0] = vehicle_dist_to_geofence;
  speed_time_info[1] = vehicle_speed;
  speed_time_info[2] = vehicle_time;
  speed_time_info[3] = object_time;
  speed_time_info[4] = time_difference;
  speed_time_info[5] = filter_state;
}
