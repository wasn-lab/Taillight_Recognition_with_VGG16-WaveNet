
#define CAN_DLC 8;
#define CAN_INTERFACE_NAME "can1"

#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Bool.h"
#include "std_msgs/String.h"
#include "msgs/Flag_Info.h"

#include "std_msgs/Header.h"
#include <iostream>
#include <cstdlib>
#include <vector>


//For CAN BUS
#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <typeinfo>

using namespace std;

const int bus_number = 8; // 8 stops 
vector<int> bus_stop_flag(bus_number, 1);;
const vector<int> bus_stop_code = {1111,2222,3333,4444};

ros::Publisher publisher_01;

void chatterCallback_01(const std_msgs::String::ConstPtr& msg)
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

	int stop_one = 0;
	int stop_two = 0;
	for (char c: msg->data)
	{
		if (c >= '0' && c <= '9') {
			stop_one = stop_one * 10 + (c - '0');
		}
		else if(c == '#'){
			stop_two = stop_one;
			stop_one = 0;
		}
		else {
			stop_one = 0;
			stop_two = 0;
			std::cout << "Bad Input";
			return;
		}
	}
	std::cout << "Add stop: " << stop_two << "," << stop_one << '\n';

	for(int i=0;i<bus_stop_code.size();i++){
		if(bus_stop_code[i]==stop_one || bus_stop_code[i]==stop_two){
			bus_stop_flag[i] = 1;
		}
	}
	
	frame.can_dlc = CAN_DLC;
	frame.can_id  = 0x055;
	for(int i=0;i<8;i++){
		frame.data[i] = bus_stop_flag[i];
		cout << "stop" << i << ": " << bus_stop_flag[i] << endl;
	}
	nbytes = write(s, &frame, sizeof(struct can_frame));
	//printf("Wrote %d bytes\n", nbytes);
	//Close the SocketCAN
	close(s);

	msgs::Flag_Info msg_temp;
	msg_temp.Dspace_Flag01 = bus_stop_flag[0];
	msg_temp.Dspace_Flag02 = bus_stop_flag[1];
	msg_temp.Dspace_Flag03 = bus_stop_flag[2];
	msg_temp.Dspace_Flag04 = bus_stop_flag[3];
	msg_temp.Dspace_Flag05 = bus_stop_flag[4];
	msg_temp.Dspace_Flag06 = bus_stop_flag[5];
	msg_temp.Dspace_Flag07 = bus_stop_flag[6];
	msg_temp.Dspace_Flag08 = bus_stop_flag[7];
	publisher_01.publish(msg_temp);
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "bus_stop_info");
	ros::NodeHandle n;
	ros::Subscriber subscriber_01 = n.subscribe("/reserve/request", 1, chatterCallback_01);
	publisher_01 = n.advertise<msgs::Flag_Info>("/BusStop/Info", 1);
	ros::spin();
	return 0;
}

