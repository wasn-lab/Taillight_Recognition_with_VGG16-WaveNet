
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
    bool getTrigger();
    double getObjSpeed();
    double getNearest_X();
    double getNearest_Y();

    int setPoly(const vector<double> &Xpoly, const vector<double> &Ypoly, int PNumber); //Update Polynimials, must set before fist execution of Calcuator
    int setPointCloud(const vector<Point> &PointCloud,bool isLocal, double SLAM_x, double SLAM_y, double Heading); //Update pointcloud, must set before fist execution of Calcuator
    int Calculator(); //Calculate geofence by currently set Poly and PointCloud
    
    

    private:
    vector<double> Xpoly_one, Xpoly_two;
    vector<double> Ypoly_one, Ypoly_two;
    vector<Point> PointCloud;
    double Distance; //Geofence distance
    bool Trigger; //Whether there is a object in the path
    double ObjSpeed; //Geofence speed
    double Nearest_X; //Nearest point's (X,Y)
    double Nearest_Y; 
};
