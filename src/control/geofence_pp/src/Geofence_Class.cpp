
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "Geofence_Class.h"

#define BOUNDARY 1.6
//#define DEBUG
//#define TEST
using namespace std;


double Geofence::getDistance(){
    return Distance;
}
double Geofence::getDistance_w(){
    return Distance_wide;
}
double Geofence::getFarest(){
    return Farest;
}
bool Geofence::getTrigger(){
    return Trigger;
}
double Geofence::getObjSpeed(){
    return ObjSpeed;
}
double Geofence::getNearest_X(){
    return Nearest_X;
}
double Geofence::getNearest_Y(){
    return Nearest_Y;
}

struct Point  Geofence::findDirection(){

    Point dir;
    Point temp;
    dir.X = 1000;
    dir.X = 1000;
    for(int i=1;i<this->PathLength.size();i++){
        if(this->PathLength[i] > this->Distance){
            temp.X = this->PathPoints[i].X - this->PathPoints[i-1].X;
            temp.Y = this->PathPoints[i].Y - this->PathPoints[i-1].Y;
            dir.X = this->PathPoints[i].X;
            dir.Y = this->PathPoints[i].Y;
            //cout << "i: " << i << endl;
            break;
        }
    }
    dir.Speed = acos((temp.X)/sqrt(pow(temp.X,2.0) + pow(temp.Y,2.0)));
    return dir;
}

int Geofence::setPath(const vector<Point> &PathPoints){
	
    this->PathPoints.clear();
    this->PathLength.clear();
    this->PathPoints = PathPoints;
   
    for(int i=0;i<PathPoints.size();i++){
        double Segment;
        double Length_temp;
        if(i==0){
            Length_temp = 0.0; //For first element
        }
        else{
            Segment = sqrt(pow((PathPoints[i].X-PathPoints[i-1].X),2)+pow((PathPoints[i].Y-PathPoints[i-1].Y),2));
            Length_temp = PathLength[i-1] + Segment;	
        }
        this->PathLength.push_back(Length_temp);
    }
    return 0;
}

int Geofence::setPointCloud(const vector<Point> &PointCloud,bool isLocal, double SLAM_x, double SLAM_y, double Heading){
	this->PointCloud.clear();
	Point Point_temp;
    if(isLocal==true){
        for (int i=0;i<PointCloud.size();i++){
			Point_temp.X = cos(Heading)*PointCloud[i].X - sin(Heading)*PointCloud[i].Y + SLAM_x;
			Point_temp.Y = sin(Heading)*PointCloud[i].X + cos(Heading)*PointCloud[i].Y + SLAM_y;
			Point_temp.Speed = PointCloud[i].Speed;
			this->PointCloud.push_back(Point_temp);
        }
    }
    else{
        this->PointCloud = PointCloud;
    }
	#ifdef TEST
        cout << "Size of PoinClout:" << PointCloud.size() << endl;
    #endif 
    return 0;
}


int Geofence::Calculator(){
    // Check if all information is initialized
    if(PathPoints.size()<1){
        cerr << "Path not initialized" << endl;
        return 1;
    }
    if(PointCloud.size()<1){
        cerr << "PointCloud not initialized" << endl;
        return 1;
    }

    vector<double> P_Distance(PointCloud.size(),300); //Distance of every pointcloud (default 100)
    vector<double> P_Distance_w(PointCloud.size(),300); //Distance of every pointcloud in wider range (default 100)
    for(int i=0;i<PointCloud.size();i++){
        vector<double> V_Distance(PathPoints.size(),300); // Vertical diatnce to the path
        for(int j=0;j<PathPoints.size();j++){
			V_Distance[j] = sqrt(pow(PointCloud[i].X-PathPoints[j].X,2) + pow(PointCloud[i].Y-PathPoints[j].Y,2)) ;    
        }
        int minElementIndex = std::min_element(V_Distance.begin(),V_Distance.end()) - V_Distance.begin();
        double minElement = *std::min_element(V_Distance.begin(), V_Distance.end());
        if(minElement<BOUNDARY){
            P_Distance[i] = PathLength[minElementIndex];      
        }
        if(minElement<(BOUNDARY+0.5)){
            P_Distance_w[i] = PathLength[minElementIndex];      
        }     
    }
    int minElementIndex = std::min_element(P_Distance.begin(),P_Distance.end()) - P_Distance.begin();
    double minElement = *std::min_element(P_Distance.begin(), P_Distance.end());
	if(minElement<99){
		Trigger = true;
		ObjSpeed = PointCloud[minElementIndex].Speed;
        Nearest_X = PointCloud[minElementIndex].X;
        Nearest_Y = PointCloud[minElementIndex].Y;
	}
	else{
		Trigger = false;
		ObjSpeed = 0;
        Nearest_X = 3000;
        Nearest_Y = 3000;
	}
    Distance = minElement;

    minElement = *std::min_element(P_Distance_w.begin(), P_Distance_w.end());
    Distance_wide = minElement;

    //Calculate farest point
    for(int i=0;i<P_Distance.size();i++){
        if(P_Distance[i]>299){
            P_Distance[i] = -100;
        }
    }
    int maxElementIndex = std::max_element(P_Distance.begin(),P_Distance.end()) - P_Distance.begin();
    double maxElement = *std::max_element(P_Distance.begin(), P_Distance.end());
    Farest = maxElement;
    return 0;
}




