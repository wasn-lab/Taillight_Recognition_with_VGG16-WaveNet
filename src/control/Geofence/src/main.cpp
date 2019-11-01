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
//#define TEST


static Geofence PCloud_Geofence;
static Geofence BBox_Geofence;
static double Heading, SLAM_x, SLAM_y;


void LocalizationToVehCallback(const msgs::LocalizationToVeh::ConstPtr& LTVmsg){
	Heading = LTVmsg->heading;
	SLAM_x = LTVmsg->x;
	SLAM_y = LTVmsg->y;
}

void chatterCallbackPCloud(const msgs::DetectedObjectArray::ConstPtr& msg){
	Point Point_temp;
	vector<Point> PointCloud_temp;
	for(int i=0;i<msg->objects.size();i++){
		Point_temp.X = msg->objects[i].bPoint.p0.x;
		Point_temp.Y = msg->objects[i].bPoint.p0.y;
		Point_temp.Speed = msg->objects[i].relSpeed;
		PointCloud_temp.push_back(Point_temp);
		Point_temp.X = msg->objects[i].bPoint.p3.x;
		Point_temp.Y = msg->objects[i].bPoint.p3.y;
		Point_temp.Speed = msg->objects[i].relSpeed;
		PointCloud_temp.push_back(Point_temp);
		Point_temp.X = msg->objects[i].bPoint.p4.x;
		Point_temp.Y = msg->objects[i].bPoint.p4.y;
		Point_temp.Speed = msg->objects[i].relSpeed;
		PointCloud_temp.push_back(Point_temp);
		Point_temp.X = msg->objects[i].bPoint.p7.x;
		Point_temp.Y = msg->objects[i].bPoint.p7.y;
		Point_temp.Speed = msg->objects[i].relSpeed;
		PointCloud_temp.push_back(Point_temp);
		Point_temp.X = msg->objects[i].bPoint.p0.x + msg->objects[i].bPoint.p3.x + msg->objects[i].bPoint.p4.x + msg->objects[i].bPoint.p7.x;
		Point_temp.Y = msg->objects[i].bPoint.p0.y + msg->objects[i].bPoint.p3.y + msg->objects[i].bPoint.p4.y + msg->objects[i].bPoint.p7.y;
		Point_temp.Speed = msg->objects[i].relSpeed;
		PointCloud_temp.push_back(Point_temp);
	}
	#ifdef TEST
		BBox_Geofence.setPointCloud(PointCloud_temp,false,SLAM_x,SLAM_y,Heading);
	#else
		BBox_Geofence.setPointCloud(PointCloud_temp,true,SLAM_x,SLAM_y,Heading);
	#endif
}


void chatterCallbackPoly(const msgs::DynamicPath::ConstPtr& msg)
{
	vector<double> XX{msg->XP1_0, msg->XP1_1, msg->XP1_2, msg->XP1_3, msg->XP1_4, msg->XP1_5, msg->XP2_0, msg->XP2_1, msg->XP2_2, msg->XP2_3, msg->XP2_4, msg->XP2_5};
	vector<double> YY{msg->YP1_0, msg->YP1_1, msg->YP1_2, msg->YP1_3, msg->YP1_4, msg->YP1_5, msg->YP2_0, msg->YP2_1, msg->YP2_2, msg->YP2_3, msg->YP2_4, msg->YP2_5};
  	PCloud_Geofence.setPoly(XX,YY,6);
	BBox_Geofence.setPoly(XX,YY,6);
}

void callback_LidarAll(const sensor_msgs::PointCloud2::ConstPtr& msg)
{ 
       
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *cloud);
	
	Point Point_temp;
	vector<Point> PointCloud_temp;
        for (size_t i = 0; i < cloud->points.size (); ++i){        
                Point_temp.X = cloud->points[i].x;
                Point_temp.Y = cloud->points[i].y;
                //double z = cloud->points[i].z;
		//int intensity = cloud->points[i].intensity;
		Point_temp.Speed = 0.0;
		PointCloud_temp.push_back(Point_temp);               
	}
	PCloud_Geofence.setPointCloud(PointCloud_temp,true,SLAM_x,SLAM_y,Heading);
}






int main(int argc, char **argv){ 

	
	ros::init(argc, argv, "Geofence");
	ros::NodeHandle n;
	ros::Subscriber LidAllSub = n.subscribe("ring_edge_point_cloud", 1, callback_LidarAll);
	ros::Subscriber PCloudGeofenceSub = n.subscribe("dynamic_path_para", 1, chatterCallbackPoly);
	ros::Subscriber LTVSub = n.subscribe("localization_to_veh", 1, LocalizationToVehCallback);
	#ifdef TEST
		ros::Subscriber BBoxGeofenceSub = n.subscribe("abs_virBB_array", 1, chatterCallbackPCloud);
	#else
		ros::Subscriber BBoxGeofenceSub = n.subscribe("PathPredictionOutput/lidar", 1, chatterCallbackPCloud);
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
		cout << "========PCloud========" << endl;
		if(PCloud_Geofence.Calculator()==0){
			frame.can_id  = 0x590;
			cout << "Trigger: " << PCloud_Geofence.getTrigger() << " ";
 			cout << "Distance: " <<  setprecision(6) <<PCloud_Geofence.getDistance() << "\t";
			cout << "Speed: " << setprecision(6) <<PCloud_Geofence.getObjSpeed() << endl;
			cout << "(X,Y): " << "(" << PCloud_Geofence.getNearest_X() << "," << PCloud_Geofence.getNearest_Y() << ")" << endl;
			//cout << "Speed: " << PCloud_Geofence.Xpoly_one.size() << "\t" << PCloud_Geofence.Xpoly_two.size() << "\t" << PCloud_Geofence.Ypoly_one.size() << "\t" << PCloud_Geofence.Ypoly_two.size() << endl;
			//cout << "Pointcloud: " << PCloud_Geofence.PointCloud.size() << endl;
			frame.data[0] = (short int)(PCloud_Geofence.getDistance()*100);
			frame.data[1] = (short int)(PCloud_Geofence.getDistance()*100)>>8;
			frame.data[2] = (short int)(PCloud_Geofence.getObjSpeed()*100);
			frame.data[3] = (short int)(PCloud_Geofence.getObjSpeed()*100)>>8;
			frame.data[4] = (short int)(PCloud_Geofence.getNearest_X()*100);
			frame.data[5] = (short int)(PCloud_Geofence.getNearest_X()*100)>>8;
			frame.data[6] = (short int)(PCloud_Geofence.getNearest_Y()*100);
			frame.data[7] = (short int)(PCloud_Geofence.getNearest_Y()*100)>>8;
			nbytes = write(s, &frame, sizeof(struct can_frame));
		}
		else{
			cerr << "Please initialize all PCloud parameters first" << endl;
		}
		
		cout << "=========BBox=========" << endl;
		if(BBox_Geofence.Calculator()==0){
			frame.can_id  = 0x591;
			cout << "Trigger: " << BBox_Geofence.getTrigger() << " ";
 			cout << "Distance: " <<  setprecision(6) << BBox_Geofence.getDistance() << "\t";
			cout << "Speed: " << setprecision(6) << BBox_Geofence.getObjSpeed() << endl;
			cout << "(X,Y): " << "(" << BBox_Geofence.getNearest_X() << "," << PCloud_Geofence.getNearest_Y() << ")" << endl << endl;
			//cout << "Speed: " << PCloud_Geofence.Xpoly_one.size() << "\t" << PCloud_Geofence.Xpoly_two.size() << "\t" << PCloud_Geofence.Ypoly_one.size() << "\t" << PCloud_Geofence.Ypoly_two.size() << endl;
			//cout << "Pointcloud: " << PCloud_Geofence.PointCloud.size() << endl;
			frame.data[0] = (short int)(BBox_Geofence.getDistance()*100);
			frame.data[1] = (short int)(BBox_Geofence.getDistance()*100)>>8;
			frame.data[2] = (short int)(BBox_Geofence.getObjSpeed()*100);
			frame.data[3] = (short int)(BBox_Geofence.getObjSpeed()*100)>>8;
			frame.data[4] = (short int)(BBox_Geofence.getNearest_X()*100);
			frame.data[5] = (short int)(BBox_Geofence.getNearest_X()*100)>>8;
			frame.data[6] = (short int)(BBox_Geofence.getNearest_Y()*100);
			frame.data[7] = (short int)(BBox_Geofence.getNearest_Y()*100)>>8;
			nbytes = write(s, &frame, sizeof(struct can_frame));
		}
		else{
			cerr << "Please initialize all BBox parameters first" << endl;
		}
		loop_rate.sleep();	
	}
	close(s);
	return 0;
}
