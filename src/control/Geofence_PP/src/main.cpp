#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "Geofence_Class.h"

// For ROS
#include "std_msgs/Header.h"
#include "msgs/BoxPoint.h"
#include "msgs/DynamicPath.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/PathPrediction.h"
#include "msgs/PointXY.h"
#include "msgs/PointXYZ.h"
#include "msgs/PointXYZV.h"
#include "msgs/TrackInfo.h"
#include "msgs/LocalizationToVeh.h"
#include "msgs/VehInfo.h"
#include "ros/ros.h"

//For PCL
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>


//For CAN
#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <typeinfo>
#define CAN_DLC 8
#define CAN_INTERFACE_NAME "can1"


// Specify running mode
#define TEST

static double Heading, SLAM_x, SLAM_y;
static Geofence BBox_Geofence;
static double Ego_speed_ms;
static vector<double> Recommand_Speed;

void LocalizationToVehCallback(const msgs::LocalizationToVeh::ConstPtr& LTVmsg){
	Heading = LTVmsg->heading;
	SLAM_x = LTVmsg->x;
	SLAM_y = LTVmsg->y;
}

void VehinfoCallback(const msgs::VehInfo::ConstPtr& VImsg){
  	Ego_speed_ms = VImsg->ego_speed;
}

void chatterCallbackPoly(const msgs::DynamicPath::ConstPtr& msg){
	vector<double> XX{msg->XP1_0, msg->XP1_1, msg->XP1_2, msg->XP1_3, msg->XP1_4, msg->XP1_5, msg->XP2_0, msg->XP2_1, msg->XP2_2, msg->XP2_3, msg->XP2_4, msg->XP2_5};
	vector<double> YY{msg->YP1_0, msg->YP1_1, msg->YP1_2, msg->YP1_3, msg->YP1_4, msg->YP1_5, msg->YP2_0, msg->YP2_1, msg->YP2_2, msg->YP2_3, msg->YP2_4, msg->YP2_5};
	BBox_Geofence.setPoly(XX,YY,6);
}

void chatterCallbackPP(const msgs::DetectedObjectArray::ConstPtr& msg){	
	Recommand_Speed.clear();
	for(int i=0;i<msg->objects.size();i++){
		for(int j=0;j<msg->objects[i].track.forecasts.size();j++){
			Point Point_temp;
			vector<Point> PointCloud_temp;
			double time = (j+1)*0.5;
			double Range_min = time*Ego_speed_ms;
			double Range_max = time*Ego_speed_ms+7; // Length of bus = 7m
			Point_temp.X = msg->objects[i].track.forecasts[j].position.x;
			Point_temp.Y = msg->objects[i].track.forecasts[j].position.y;
			Point_temp.Speed = msg->objects[i].relSpeed;
			PointCloud_temp.push_back(Point_temp);
			#ifdef TEST
				BBox_Geofence.setPointCloud(PointCloud_temp,false,SLAM_x,SLAM_y,Heading);
			#else
				BBox_Geofence.setPointCloud(PointCloud_temp,true,SLAM_x,SLAM_y,Heading);
			#endif
			if(BBox_Geofence.Calculator()==1){
				cerr << "Please initialize all PCloud parameters first" << endl;
				return;
			}
			#ifdef TEST
				if(BBox_Geofence.getDistance()<80){
					cout << "PP Points in boundary: " << BBox_Geofence.getDistance() << "m" << endl;
				}
			#endif
			if(BBox_Geofence.getDistance()<Range_max & BBox_Geofence.getDistance()>Range_min){
				Recommand_Speed.push_back((2*BBox_Geofence.getDistance()/time)-Ego_speed_ms);
				#ifdef TEST
					cout << "Collision appears" << endl;
				#endif
			} 
		}	
	}
}


int main(int argc, char **argv){ 

	ros::init(argc, argv, "Geofence_PP");
	ros::NodeHandle n;
	ros::Subscriber PCloudGeofenceSub = n.subscribe("dynamic_path_para", 1, chatterCallbackPoly);
	ros::Subscriber LTVSub = n.subscribe("localization_to_veh", 1, LocalizationToVehCallback);
	ros::Subscriber VI_sub = n.subscribe("veh_info", 1, VehinfoCallback);
	#ifdef TEST
		ros::Subscriber BBoxGeofenceSub = n.subscribe("abs_virBB_array", 1, chatterCallbackPP);
	#else
		ros::Subscriber BBoxGeofenceSub = n.subscribe("PathPredictionOutput/lidar", 1, chatterCallbackPP);
	#endif
	ros::Rate loop_rate(10);

	int s;
	int nbytes;
	struct sockaddr_can addr;
	struct can_frame frame;
	struct ifreq ifr;
	const char *ifname = CAN_INTERFACE_NAME;
	if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0){
		perror("Error while opening socket");
	}
	strcpy(ifr.ifr_name, ifname);
	ioctl(s, SIOCGIFINDEX, &ifr);
	addr.can_family  = AF_CAN;
	addr.can_ifindex = ifr.ifr_ifindex;
	if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0){
		perror("Error in socket bind");
	}
	frame.can_dlc = CAN_DLC ;	

	while (ros::ok())
	{
		ros::spinOnce();
		if(Recommand_Speed.size()<1){
			cout << "Nothing in boundary" << endl;
		}
		else{
        	double minElement = *std::min_element(Recommand_Speed.begin(), Recommand_Speed.end());	
			cout << "Recommand speed: " << minElement << endl;
		}
		/*
		frame.can_id  = 0x590;
		frame.data[0] = (short int)(PCloud_Geofence.getDistance()*100);
		frame.data[1] = (short int)(PCloud_Geofence.getDistance()*100)>>8;
		frame.data[2] = (short int)(PCloud_Geofence.getObjSpeed()*100);
		frame.data[3] = (short int)(PCloud_Geofence.getObjSpeed()*100)>>8;
		frame.data[4] = (short int)(PCloud_Geofence.getNearest_X()*100);
		frame.data[5] = (short int)(PCloud_Geofence.getNearest_X()*100)>>8;
		frame.data[6] = (short int)(PCloud_Geofence.getNearest_Y()*100);
		frame.data[7] = (short int)(PCloud_Geofence.getNearest_Y()*100)>>8;
		nbytes = write(s, &frame, sizeof(struct can_frame));
		*/
		loop_rate.sleep();	
	}
	close(s);
	return 0;
}
