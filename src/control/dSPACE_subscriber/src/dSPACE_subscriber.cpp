
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
#include <iostream>
#include <cstdlib>
#include "ros/ros.h"
//#include "std_msgs/String.h"


//For CAN BUS
#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <typeinfo>

using namespace std;

void chatterCallback(const msgs::DetectedObjectArray::ConstPtr& msg)
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
	
	//printf("Wrote %d bytes\n", nbytes);
	//Close the SocketCAN

}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "dSPACE_subscriber");
  ros::NodeHandle n;
  ros::Subscriber dSPACE_subscriber = n.subscribe("PathPredictionOutput/lidar", 1, chatterCallback);
  ros::spin();
  return 0;
}


