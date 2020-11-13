
#include "std_msgs/Header.h"
#include "msgs/BoxPoint.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/PathPrediction.h"
#include "msgs/PointXY.h"
#include "msgs/PointXYZ.h"
#include "msgs/PointXYZV.h"
#include "msgs/TrackInfo.h"
#include "msgs/DetectedLight.h"
#include "msgs/DetectedLightArray.h"
#include "msgs/Spat.h"
#include <iostream>
#include <cstdlib>
#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Bool.h"
#include "msgs/Flag_Info.h"
#include "msgs/BackendInfo.h"
#include "std_msgs/Int8MultiArray.h"
//#include "std_msgs/String.h"

using namespace std;
std_msgs::Int8MultiArray checker;
int timeout = 15;
int const num_of_checker = 9; //一共有追蹤幾個topic
int counter[num_of_checker];





void chatterCallback_01(const msgs::Flag_Info::ConstPtr& msg)
{
	//cout << "counter reset" << endl;
	counter[0] = 0;
}

void chatterCallback_02(const msgs::Flag_Info::ConstPtr& msg)
{
	counter[1] = 0;
}

void chatterCallback_03(const msgs::Flag_Info::ConstPtr& msg)
{
	counter[2] = 0;
}

void chatterCallback_04(const msgs::Flag_Info::ConstPtr& msg)
{
	counter[3] = 0;
}

void chatterCallback_05(const msgs::Flag_Info::ConstPtr& msg)
{
	counter[4] = 0;
}

void chatterCallback_06(const msgs::Flag_Info::ConstPtr& msg)
{
	counter[5] = 0;
}

void chatterCallback_07(const std_msgs::Float64::ConstPtr& msg)
{
	counter[6] = 0;
}

void chatterCallback_08(const std_msgs::Float64::ConstPtr& msg)
{
	counter[7] = 0;
}

void chatterCallback_09(const msgs::BackendInfo::ConstPtr& msg)
{
	counter[8] = 0;
}


int main(int argc, char **argv)
{
	checker.data.resize(num_of_checker);
	ros::init(argc, argv, "control_checker");
	ros::NodeHandle n;
	ros::NodeHandle n1;
	ros::Rate rate(10);
	ros::Subscriber checker01 = n.subscribe("Flag_Info01", 1, chatterCallback_01);
	ros::Subscriber checker02 = n.subscribe("Flag_Info02", 1, chatterCallback_02);
	ros::Subscriber checker03 = n.subscribe("Flag_Info03", 1, chatterCallback_03);
	ros::Subscriber checker04 = n.subscribe("Flag_Info04", 1, chatterCallback_04);
	ros::Subscriber checker05 = n.subscribe("Flag_Info05", 1, chatterCallback_05);
	ros::Subscriber checker06 = n.subscribe("/NextStop/Info", 1, chatterCallback_06);
	ros::Subscriber checker07 = n.subscribe("/Ego_speed/kph", 1, chatterCallback_07);
	ros::Subscriber checker08 = n.subscribe("/Ego_speed/ms", 1, chatterCallback_08);
	ros::Subscriber checker09 = n.subscribe("Backend/Info", 1, chatterCallback_09);
	ros::Publisher checker_output = n1.advertise<std_msgs::Int8MultiArray>("control_checker", 1);;
	while(ros::ok()){
		ros::spinOnce();
		//cout << "counter:" << endl << counter01 << endl << counter02 << endl << counter03 << endl << counter04 << endl << counter05 << endl << counter06 << endl << counter07 << endl << counter08 << endl << counter09 <<endl;
		for(int i=0;i<num_of_checker;i++){
			counter[i] = counter[i]+1;
		}

		if (counter[0]>timeout){
			cout << "Flag_Info01 time out." << endl;
			checker.data[0] = 1;
		}
		else{
			checker.data[0] = 0;
		}
		if (counter[1]>timeout){
			cout << "Flag_Info02 time out." << endl;
			checker.data[1] = 1;
		}
		else{
			checker.data[1] = 0;
		}
		if (counter[2]>timeout){
			cout << "Flag_Info03 time out." << endl;
			checker.data[2] = 1;
		}
		else{
			checker.data[2] = 0;
		}
		if (counter[3]>timeout){
			cout << "Flag_Info04 time out." << endl;
			checker.data[3] = 1;
		}
		else{
			checker.data[3] = 0;
		}
		// if (counter[4]>timeout){
		// 	cout << "Flag_Info05 time out." << endl;
		// 	checker.data[4] = 1;
		// }
		// else{
			checker.data[4] = 0;
		// }
		if (counter[5]>timeout){
			cout << "/NextStop/Info time out." << endl;
			checker.data[5] = 1;
		}
		else{
			checker.data[5] = 0;
		}
		if (counter[6]>timeout){
			cout << "/Ego_speed/kph time out." << endl;
			checker.data[6] = 1;
		}
		else{
			checker.data[6] = 0;
		}
		if (counter[7]>timeout){
			cout << "/Ego_speed/ms time out." << endl;
			checker.data[7] = 1;
		}
		else{
			checker.data[7] = 0;
		}
		if (counter[8]>timeout){
			cout << "Backend/Info time out." << endl;
			checker.data[8] = 1;
		}
		else{
			checker.data[8] = 0;
		}
		checker_output.publish(checker);
		rate.sleep();
	}
	for(int i=0;i<checker.data.size();i++){
		cout << int(checker.data[i]) << endl;
	}

  
  return 0;
}


