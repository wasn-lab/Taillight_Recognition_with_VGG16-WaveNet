
#define CAN_DLC 8;
#define CAN_INTERFACE_NAME "can1"
#define can_id_start  0x000
const double NumOfID = 5;

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
#include "msgs/CurrentTrajInfo.h"
#include <iostream>
#include <cstdlib>
#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Bool.h"
//#include "std_msgs/String.h"
#include <msgs/BehaviorSceneRegister.h>


//For CAN BUS
#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <typeinfo>

using namespace std;

void chatterCallback_01(const msgs::DetectedObjectArray::ConstPtr& msg)
{

	int s;
	int nbytes;
	struct sockaddr_can addr;
	struct can_frame frame;
	struct ifreq ifr;

	const char *ifname = CAN_INTERFACE_NAME;

	if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
	{
		perror("Error while opening socket");
	}

	strcpy(ifr.ifr_name, ifname);
	ioctl(s, SIOCGIFINDEX, &ifr);

	addr.can_family  = AF_CAN;
	addr.can_ifindex = ifr.ifr_ifindex;

	printf("%s at index %d\n", ifname, ifr.ifr_ifindex);

	if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0)
	{
		perror("Error in socket bind");
	}
	
	frame.can_dlc = CAN_DLC ;
	
	int size = msg->objects.size();
	cout << "Size = " << size  << endl;
			
	for (int i =0 ; i<size ; i++)
	{
		
		cout << " Object_" << i << ": " << endl;
		frame.can_id  = can_id_start+(i*NumOfID) ; 
		frame.data[0] = (short int)(msg->objects[i].bPoint.p0.x*100);
		frame.data[1] = (short int)(msg->objects[i].bPoint.p0.x*100)>>8;
		frame.data[2] = (short int)(msg->objects[i].bPoint.p0.y*100);
		frame.data[3] = (short int)(msg->objects[i].bPoint.p0.y*100)>>8;
		frame.data[4] = (short int)(msg->objects[i].bPoint.p3.x*100);
		frame.data[5] = (short int)(msg->objects[i].bPoint.p3.x*100)>>8;
		frame.data[6] = (short int)(msg->objects[i].bPoint.p3.y*100);
		frame.data[7] = (short int)(msg->objects[i].bPoint.p3.y*100)>>8;
		nbytes = write(s, &frame, sizeof(struct can_frame));
		cout << "p0: (" << msg->objects[i].bPoint.p0.x << "," << msg->objects[i].bPoint.p0.y << ")" << endl;
		cout << "p3: (" << msg->objects[i].bPoint.p3.x << "," << msg->objects[i].bPoint.p3.y << ")" << endl;
		cout << "CAN_ID = " <<  frame.can_id << endl;

		frame.can_id  = can_id_start+(i*NumOfID)+1 ;
		frame.data[0] = (short int)(msg->objects[i].bPoint.p4.x*100);
		frame.data[1] = (short int)(msg->objects[i].bPoint.p4.x*100)>>8;
		frame.data[2] = (short int)(msg->objects[i].bPoint.p4.y*100);
		frame.data[3] = (short int)(msg->objects[i].bPoint.p4.y*100)>>8;
		frame.data[4] = (short int)(msg->objects[i].bPoint.p7.x*100);
		frame.data[5] = (short int)(msg->objects[i].bPoint.p7.x*100)>>8;
		frame.data[6] = (short int)(msg->objects[i].bPoint.p7.y*100);
		frame.data[7] = (short int)(msg->objects[i].bPoint.p7.y*100)>>8;
		nbytes = write(s, &frame, sizeof(struct can_frame));
		cout << "p4: (" << msg->objects[i].bPoint.p4.x << "," << msg->objects[i].bPoint.p4.y << ")" << endl;
		cout << "p7: (" << msg->objects[i].bPoint.p7.x << "," << msg->objects[i].bPoint.p7.y << ")" << endl;			
		cout << "CAN_ID = " <<  frame.can_id << endl;

		frame.can_id  = can_id_start+(i*NumOfID)+2 ;
		frame.data[0] = (short int)(msg->objects[i].track.relative_velocity.x*100);
		frame.data[1] = (short int)(msg->objects[i].track.relative_velocity.x*100)>>8;
		frame.data[2] = (short int)(msg->objects[i].track.relative_velocity.y*100);
		frame.data[3] = (short int)(msg->objects[i].track.relative_velocity.y*100)>>8;
		frame.data[4] = (short int)(size);
		frame.data[5] = (short int)(size)>>8;
		nbytes = write(s, &frame, sizeof(struct can_frame));
		cout << "Relative speed: (" << msg->objects[i].track.relative_velocity.x << "," << msg->objects[i].track.relative_velocity.y << ")" << endl;
		cout << "CAN_ID = " <<  frame.can_id << endl;

		//PP info
		if (msg->objects[i].track.is_ready_prediction==1)
		{
			frame.can_id  = can_id_start+(i*NumOfID)+3 ;
			frame.data[0] = (short int)(msg->objects[i].track.forecasts[4].position.x*100);
			frame.data[1] = (short int)(msg->objects[i].track.forecasts[4].position.x*100)>>8;
			frame.data[2] = (short int)(msg->objects[i].track.forecasts[4].position.y*100);
			frame.data[3] = (short int)(msg->objects[i].track.forecasts[4].position.y*100)>>8;
			frame.data[4] = (short int)(msg->objects[i].track.forecasts[9].position.x*100);
			frame.data[5] = (short int)(msg->objects[i].track.forecasts[9].position.x*100)>>8;
			frame.data[6] = (short int)(msg->objects[i].track.forecasts[9].position.y*100);
			frame.data[7] = (short int)(msg->objects[i].track.forecasts[9].position.y*100)>>8;
			nbytes = write(s, &frame, sizeof(struct can_frame));
			cout << "PP.P1: (" << msg->objects[i].track.forecasts[4].position.x << "," << msg->objects[i].track.forecasts[4].position.y << ")" << endl;
			cout << "PP.P2: (" << msg->objects[i].track.forecasts[9].position.x << "," << msg->objects[i].track.forecasts[9].position.y << ")" << endl;
			cout << "CAN_ID = " <<  frame.can_id << endl;
		
			frame.can_id  = can_id_start+(i*NumOfID)+4 ;
			frame.data[0] = (short int)(msg->objects[i].track.forecasts[14].position.x*100);
			frame.data[1] = (short int)(msg->objects[i].track.forecasts[14].position.x*100)>>8;
			frame.data[2] = (short int)(msg->objects[i].track.forecasts[14].position.y*100);
			frame.data[3] = (short int)(msg->objects[i].track.forecasts[14].position.y*100)>>8;
			frame.data[4] = (short int)(msg->objects[i].track.forecasts[19].position.x*100);
			frame.data[5] = (short int)(msg->objects[i].track.forecasts[19].position.x*100)>>8;
			frame.data[6] = (short int)(msg->objects[i].track.forecasts[19].position.y*100);
			frame.data[7] = (short int)(msg->objects[i].track.forecasts[19].position.y*100)>>8;
			nbytes = write(s, &frame, sizeof(struct can_frame));
			cout << "PP.P3: (" << msg->objects[i].track.forecasts[14].position.x << "," << msg->objects[i].track.forecasts[14].position.y << ")" << endl;
			cout << "PP.P4: (" << msg->objects[i].track.forecasts[19].position.x << "," << msg->objects[i].track.forecasts[19].position.y << ")" << endl;
			cout << "CAN_ID = " <<  frame.can_id << endl;
		}
					
	}
	close(s);
	printf("Wrote %d bytes\n", nbytes);
	//Close the SocketCAN

}

void chatterCallback_02(const std_msgs::Bool::ConstPtr& msg)
{

	int s;
	int nbytes;
	struct sockaddr_can addr;
	struct can_frame frame;
	struct ifreq ifr;

	const char *ifname = CAN_INTERFACE_NAME;

	if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
	{
		perror("Error while opening socket");
	}

	strcpy(ifr.ifr_name, ifname);
	ioctl(s, SIOCGIFINDEX, &ifr);

	addr.can_family  = AF_CAN;
	addr.can_ifindex = ifr.ifr_ifindex;

	printf("%s at index %d\n", ifname, ifr.ifr_ifindex);

	if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0)
	{
		perror("Error in socket bind");
	}
	
	frame.can_dlc = CAN_DLC;
	frame.can_id  = 0x050;
	frame.data[0] = (short int)(msg->data);
	frame.data[1] = (short int)(msg->data)>>8;
	nbytes = write(s, &frame, sizeof(struct can_frame));
	cout << "Driving state: " << int(msg->data) << endl;
	close(s);
	printf("Wrote %d bytes\n", nbytes);
	//Close the SocketCAN

}

void chatterCallback_03(const std_msgs::Bool::ConstPtr& msg)
{

	int s;
	int nbytes;
	struct sockaddr_can addr;
	struct can_frame frame;
	struct ifreq ifr;

	const char *ifname = CAN_INTERFACE_NAME;

	if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
	{
		perror("Error while opening socket");
	}

	strcpy(ifr.ifr_name, ifname);
	ioctl(s, SIOCGIFINDEX, &ifr);

	addr.can_family  = AF_CAN;
	addr.can_ifindex = ifr.ifr_ifindex;

	printf("%s at index %d\n", ifname, ifr.ifr_ifindex);

	if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0)
	{
		perror("Error in socket bind");
	}
	
	frame.can_dlc = CAN_DLC;
	frame.can_id  = 0x051;
	frame.data[0] = (short int)(msg->data);
	frame.data[1] = (short int)(msg->data)>>8;
	nbytes = write(s, &frame, sizeof(struct can_frame));
	cout << "System ready: " << int(msg->data) << endl;
	close(s);
	printf("Wrote %d bytes\n", nbytes);
	//Close the SocketCAN

}

void chatterCallback_04(const msgs::DetectedLightArray::ConstPtr& msg)
{
	uint color_temp=100;
	uint direction_temp=100;
	float distance_temp=2000;

	for(uint i=0;i<msg->lights.size();i++)
	{
		if(msg->lights[i].distance<distance_temp)
		{
			color_temp = msg->lights[i].color_light;
			direction_temp = msg->lights[i].direction;
		}
	}

	int s;
	int nbytes;
	struct sockaddr_can addr;
	struct can_frame frame;
	struct ifreq ifr;

	const char *ifname = CAN_INTERFACE_NAME;

	if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
	{
		perror("Error while opening socket");
	}

	strcpy(ifr.ifr_name, ifname);
	ioctl(s, SIOCGIFINDEX, &ifr);

	addr.can_family  = AF_CAN;
	addr.can_ifindex = ifr.ifr_ifindex;

	printf("%s at index %d\n", ifname, ifr.ifr_ifindex);

	if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0)
	{
		perror("Error in socket bind");
	}
	
	frame.can_dlc = CAN_DLC;
	frame.can_id  = 0x061;
	frame.data[0] = (short int)(color_temp);
	frame.data[1] = (short int)(color_temp)>>8;
	frame.data[2] = (short int)(direction_temp);
	frame.data[3] = (short int)(direction_temp)>>8;
	nbytes = write(s, &frame, sizeof(struct can_frame));
	cout << "Light sign: " << int(color_temp) << endl;
	cout << "Direction: " << int(direction_temp) << endl;
	close(s);
	printf("Wrote %d bytes\n", nbytes);
	//Close the SocketCAN

}

void chatterCallback_05(const msgs::Spat::ConstPtr& msg)
{
	int s;
	int nbytes;
	struct sockaddr_can addr;
	struct can_frame frame;
	struct ifreq ifr;

	const char *ifname = CAN_INTERFACE_NAME;

	if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
	{
		perror("Error while opening socket");
	}

	strcpy(ifr.ifr_name, ifname);
	ioctl(s, SIOCGIFINDEX, &ifr);

	addr.can_family  = AF_CAN;
	addr.can_ifindex = ifr.ifr_ifindex;

	printf("%s at index %d\n", ifname, ifr.ifr_ifindex);

	if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0)
	{
		perror("Error in socket bind");
	}
	
	frame.can_dlc = CAN_DLC;
	frame.can_id  = 0x062;
	frame.data[0] = (short int)(msg->spat_state);
	frame.data[1] = (short int)(msg->spat_state)>>8;
	frame.data[2] = (short int)(msg->spat_sec*100);
	frame.data[3] = (short int)(msg->spat_sec*100)>>8;
	frame.data[4] = (short int)(msg->signal_state);
	frame.data[5] = (short int)(msg->signal_state)>>8;
	frame.data[6] = (short int)(msg->index);
	frame.data[7] = (short int)(msg->index)>>8;
	nbytes = write(s, &frame, sizeof(struct can_frame));
	cout << "spat_state: " << int(msg->spat_state) << endl;
	cout << "spat_sec: " << double(msg->spat_sec) << endl;
	cout << "spat_status: " << int(msg->signal_state) << endl;
	cout << "spat_index: " << int(msg->index) << endl;
	close(s);
	printf("Wrote %d bytes\n", nbytes);
	//Close the SocketCAN

}

void chatterCallback_06(const msgs::CurrentTrajInfo::ConstPtr& msg)
{
	//LRturn
	int s;
	int nbytes;
	struct sockaddr_can addr;
	struct can_frame frame;
	struct ifreq ifr;

	const char *ifname = CAN_INTERFACE_NAME;

	if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
	{
		perror("Error while opening socket");
	}

	strcpy(ifr.ifr_name, ifname);
	ioctl(s, SIOCGIFINDEX, &ifr);

	addr.can_family  = AF_CAN;
	addr.can_ifindex = ifr.ifr_ifindex;

	printf("%s at index %d\n", ifname, ifr.ifr_ifindex);

	if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0)
	{
		perror("Error in socket bind");
	}
	
	frame.can_dlc = CAN_DLC;
	frame.can_id  = 0x063;
	frame.data[0] = (short int)(msg->LRturn);
	frame.data[1] = (short int)(msg->Curvature*1000);
	frame.data[2] = (short int)(msg->Curvature*1000)>>8;
	frame.data[3] = (short int)(msg->Distoturn*100);
	frame.data[4] = (short int)(msg->Distoturn*100)>>8;
	nbytes = write(s, &frame, sizeof(struct can_frame));
	cout << "LRturn: " << int(msg->LRturn) << endl;
	cout << "Curvature: " << double(msg->Curvature) << endl;
	cout << "Distoturn: " << int(msg->Distoturn) << endl;
	close(s);
	printf("Wrote %d bytes\n", nbytes);
	//Close the SocketCAN

	//Uphill
	int s_1;
	int nbytes_1;
	struct sockaddr_can addr_1;
	struct can_frame frame_1;
	struct ifreq ifr_1;

	const char *ifname_1 = CAN_INTERFACE_NAME;

	if((s_1 = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
	{
		perror("Error while opening socket");
	}

	strcpy(ifr_1.ifr_name, ifname_1);
	ioctl(s_1, SIOCGIFINDEX, &ifr_1);

	addr_1.can_family  = AF_CAN;
	addr_1.can_ifindex = ifr_1.ifr_ifindex;

	printf("%s at index %d\n", ifname_1, ifr_1.ifr_ifindex);

	if(bind(s_1, (struct sockaddr *)&addr_1, sizeof(addr_1)) < 0)
	{
		perror("Error in socket bind");
	}
	
	frame_1.can_dlc = CAN_DLC;
	frame_1.can_id  = 0x064;
	frame_1.data[0] = (short int)(msg->Uphill);
	frame_1.data[1] = (short int)(msg->MaxSlope*1000);
	frame_1.data[2] = (short int)(msg->MaxSlope*1000)>>8;
	frame_1.data[3] = (short int)(msg->Distouphill*100);
	frame_1.data[4] = (short int)(msg->Distouphill*100)>>8;
	nbytes_1 = write(s_1, &frame_1, sizeof(struct can_frame));
	cout << "Uphill: " << int(msg->Uphill) << endl;
	cout << "MaxSlope: " << double(msg->MaxSlope) << endl;
	cout << "Distouphill: " << int(msg->Distouphill) << endl;
	close(s_1);
	printf("Wrote %d bytes\n", nbytes_1);
	//Close the SocketCAN

}

void chatterCallback_07(const msgs::BehaviorSceneRegister::ConstPtr& msg)
{
	//LRturn
	int s;
	int nbytes;
	struct sockaddr_can addr;
	struct can_frame frame;
	struct ifreq ifr;

	const char *ifname = CAN_INTERFACE_NAME;

	if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
	{
		perror("Error while opening socket");
	}

	strcpy(ifr.ifr_name, ifname);
	ioctl(s, SIOCGIFINDEX, &ifr);

	addr.can_family  = AF_CAN;
	addr.can_ifindex = ifr.ifr_ifindex;

	printf("%s at index %d\n", ifname, ifr.ifr_ifindex);

	if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0)
	{
		perror("Error in socket bind");
	}
	
	frame.can_dlc = CAN_DLC;
	frame.can_id  = 0x065;
	frame.data[0] = (short int)(msg->StopZone);
	frame.data[1] = (short int)(msg->BusStopNum);
	frame.data[2] = (short int)(msg->Distance*1000);
	frame.data[3] = (short int)(msg->Distance*1000)>>8;
	nbytes = write(s, &frame, sizeof(struct can_frame));
	cout << "StopZone: " << int(msg->StopZone) << endl;
	cout << "BusStopNum: " << int(msg->BusStopNum) << endl;
	cout << "Stop_Distance: " << double(msg->Distance) << endl;
	close(s);
	printf("Wrote %d bytes\n", nbytes);
	//Close the SocketCAN
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "to_dspace");
  ros::NodeHandle n;
  //ros::Subscriber dSPACE_subscriber_01 = n.subscribe("PathPredictionOutput/lidar", 1, chatterCallback_01);
  ros::Subscriber dSPACE_subscriber_02 = n.subscribe("/ADV_op/req_run_stop", 1, chatterCallback_02);
  ros::Subscriber dSPACE_subscriber_03 = n.subscribe("/ADV_op/sys_ready", 1, chatterCallback_03);
  ros::Subscriber dSPACE_subscriber_04 = n.subscribe("LightResultOutput_ITRI_Campus", 1, chatterCallback_04);
  ros::Subscriber dSPACE_subscriber_05 = n.subscribe("/traffic", 1, chatterCallback_05);
  ros::Subscriber dSPACE_subscriber_06 = n.subscribe("/current_trajectory_info", 1, chatterCallback_06);
  ros::Subscriber dSPACE_subscriber_07 = n.subscribe("/bus_stop_register_info", 1, chatterCallback_07);
  ros::spin();
  return 0;
}


