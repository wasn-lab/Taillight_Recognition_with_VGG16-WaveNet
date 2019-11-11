
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "Geofence_Class.h"

#define BOUNDARY 1.5
//#define DEBUG
//#define TEST
using namespace std;


double Geofence::getDistance(){
    return Distance;
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

int Geofence::setPoly(const vector<double> &Xpoly, const vector<double> &Ypoly, int PNumber){
    if (Xpoly.size()!=(PNumber*2) || Ypoly.size()!=(PNumber*2)){
        cout << "Number of parameters mismatch." << endl;
        return 1;
    }
    Xpoly_one.clear();
    Xpoly_two.clear();
    Ypoly_one.clear();
    Ypoly_two.clear();
    for(int i=0;i<PNumber;i++){ 
        Xpoly_one.push_back(Xpoly[i]);
        Xpoly_two.push_back(Xpoly[i+PNumber]);
        Ypoly_one.push_back(Ypoly[i]);
        Ypoly_two.push_back(Ypoly[i+PNumber]);
    }
    #ifdef DEBUG
        cout << "Xpoly_one["; 
        for(int i=0;i<PNumber;i++){
            cout << Xpoly_one[i] ;
            if(i!=(PNumber-1)){
                cout << ",";
            }
        }
        cout << "]" << endl;
        cout << "Xpoly_two["; 
        for(int i=0;i<PNumber;i++){
            cout << Xpoly_two[i];
            if(i!=(PNumber-1)){
                cout << ",";
            }
        }
        cout << "]" << endl;
        cout << "Ypoly_one["; 
        for(int i=0;i<PNumber;i++){
            cout << Ypoly_one[i];
            if(i!=(PNumber-1)){
                cout << ",";
            }
        }
        cout << "]" << endl;
        cout << "Ypoly_two["; 
        for(int i=0;i<PNumber;i++){
            cout << Ypoly_two[i];
            if(i!=(PNumber-1)){
                cout << ",";
            }
        }
        cout << "]" << endl;
    #endif
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
    double Resolution = 0.001;
    // Check if all information is initialized
    if(Xpoly_one.size()<1 || Xpoly_two.size()<1 || Ypoly_one.size()<1 || Ypoly_two.size()<1 ){
        cerr << "Path polynomials not initialized" << endl;
        return 1;
    }
    if(PointCloud.size()<1){
        cerr << "PointCloud not initialized" << endl;
        return 1;
    }
    vector<Point> Position;
    vector<double> Length;
    for(double i=0.0;i<1.0;i+=Resolution){ 
        struct Point Pos;
        Pos.X = Xpoly_one[0] + Xpoly_one[1]*i +  Xpoly_one[2]*pow(i,2) +  Xpoly_one[3]*pow(i,3) +  Xpoly_one[4]*pow(i,4) +  Xpoly_one[5]*pow(i,5);
        Pos.Y = Ypoly_one[0] + Ypoly_one[1]*i +  Ypoly_one[2]*pow(i,2) +  Ypoly_one[3]*pow(i,3) +  Ypoly_one[4]*pow(i,4) +  Ypoly_one[5]*pow(i,5);             
        Position.push_back(Pos);
    }
    for(double i=0.0;i<1.0;i+=Resolution){
        struct Point Pos;
        Pos.X = Xpoly_two[0] + Xpoly_two[1]*i +  Xpoly_two[2]*pow(i,2) +  Xpoly_two[3]*pow(i,3) +  Xpoly_two[4]*pow(i,4) +  Xpoly_two[5]*pow(i,5);
        Pos.Y = Ypoly_two[0] + Ypoly_two[1]*i +  Ypoly_two[2]*pow(i,2) +  Ypoly_two[3]*pow(i,3) +  Ypoly_two[4]*pow(i,4) +  Ypoly_two[5]*pow(i,5);
        Position.push_back(Pos);
    }
    for(int i=0;i<Position.size();i++){
        double Segment;
        double Length_temp;
        if(i==0){
            Length_temp = 0.0; //For first element
        }
        else{
            Segment = sqrt(pow((Position[i].X-Position[i-1].X),2)+pow((Position[i].Y-Position[i-1].Y),2));
            Length_temp = Length[i-1] + Segment;	
        }
        Length.push_back(Length_temp);
    }
    #ifdef TEST
        cout << "Number of path point: " << Position.size() << endl;
        cout << "Number of path Length: " << Length.size() << endl;
        cout << "Path length: " << *Length.rbegin() << endl;
    #endif
    vector<double> P_Distance(PointCloud.size(),100); //Distance of every pointcloud (default 100)
    for(int i=0;i<PointCloud.size();i++){
        vector<double> V_Distance(Position.size(),100); // Vertical diatnce to the path
        for(int j=0;j<Position.size();j++){
			V_Distance[j] = sqrt(pow(PointCloud[i].X-Position[j].X,2) + pow(PointCloud[i].Y-Position[j].Y,2)) ;    
        }
        int minElementIndex = std::min_element(V_Distance.begin(),V_Distance.end()) - V_Distance.begin();
        double minElement = *std::min_element(V_Distance.begin(), V_Distance.end());
        if(minElement<BOUNDARY){
            P_Distance[i] = Length[minElementIndex];      
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
        Nearest_X = 100;
        Nearest_Y = 100;
	}
    Distance = minElement;
    return 0;
}




