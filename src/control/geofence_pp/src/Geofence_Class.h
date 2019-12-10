
#include <iostream>
#include <vector>
#include <string>
#include <math.h>

using namespace std;
 
struct Point {
    double X;
    double Y;
    double Speed;
};


class Geofence{
    public:
    double getDistance();
    double getFarest();
    bool getTrigger();
    double getObjSpeed();
    double getNearest_X();
    double getNearest_Y();

    int setPointCloud(const vector<Point> &PointCloud,bool isLocal, double SLAM_x, double SLAM_y, double Heading); //Update pointcloud, must set before fist execution of Calcuator
    int setPath(const vector<Point> &PathPoints); //Update Path points in absolute coordibate, must set before fist execution of Calcuator
    int Calculator(); //Calculate geofence by currently set Poly and PointCloud
    
    

    private:
    vector<Point> PathPoints;
    vector<double> PathLength;
    vector<Point> PointCloud;
    double Distance; //Geofence distance
    double Farest; //The farest point
    bool Trigger; //Whether there is a object in the path
    double ObjSpeed; //Geofence speed
    double Nearest_X; //Nearest point's (X,Y)
    double Nearest_Y; 
};
