
#define CAN_DLC 8;
#define CAN_INTERFACE_NAME "can1"

#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Bool.h"
#include "std_msgs/String.h"
#include "msgs/Flag_Info.h"
#include "msgs/StopInfoArray.h"
#include "msgs/StopInfo.h"

#include "std_msgs/Header.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>


//For CAN BUS
#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <typeinfo>

using namespace std;

const int stop_number = 8; // 8 stops 
vector<vector<int> > bus_stop_info(5);
int round_count=1;
const vector<int> bus_stop_code = {2001,2002,2003,2004,2005};

ros::Publisher publisher_01;



void send_can(){
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
	printf("%s at index %d\n", ifname, ifr.ifr_ifindex);
	if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0){
		perror("Error in socket bind");
	}
	frame.can_dlc = CAN_DLC;
	frame.can_id  = 0x055;
	for(int i=0;i<8;i++){
		frame.data[i] = (short int)(bus_stop_info[0][i]);
			cout << "stop" << i+1 << ": " << int(frame.data[i]) << endl;
	}
	nbytes = write(s, &frame, sizeof(struct can_frame));
	printf("Wrote %d bytes\n", nbytes);
	//Close the SocketCAN
	close(s);
}


void chatterCallback_01(const msgs::StopInfoArray::ConstPtr& msg)
{
	/*
	unsigned int in_round = msg->stops[0].round; //上車趟次
	unsigned int in_id = msg->stops[0].id; //上車站點ID
	unsigned int out_round = msg->stops[1].round; //下車趟次
	unsigned int out_id = msg->stops[1].id; //下車站點ID
	*/
	std::cout << "Current round:" << endl;
	for(uint i=0;i<msg->stops.size();i++){
		for(uint j=0;j<bus_stop_code.size();j++){
			if(bus_stop_code[j]==msg->stops[i].id && msg->stops[i].round>=round_count){
				bus_stop_info[msg->stops[i].round-round_count][j] = 1;
			}
			std::cout << "stop" << j+1 << ":" << bus_stop_info[0][j] << '\n';
		}
	}
	std::cout << "Next round:" << endl;
	for(uint j=0;j<bus_stop_code.size();j++){
		std::cout << "stop" << j+1 << ":" << bus_stop_info[1][j] << '\n';
	}

	/*
	for (string s: msg->data)
	{
		int temp = 0;
		int stop_round = 0;
		int stop_id = 0;
		for (string c:s)
		{
			if (c >= '0' && c <= '9') {
				temp = temp * 10 + (c - '0');
			}
			else if(c == '#'){
				stop_round = temp;
				temp = 0;
			}
			else {
				std::cout << "Bad Input";
				return;
			}
		}
		stop_id = temp;
		std::cout << "Round: " << stop_round << '\t' << "ID: "<< stop_id <<'\n';
		for(uint i=0;i<bus_stop_code.size();i++){
			if(bus_stop_code[i]==stop_id){
				bus_stop_flag[stop_round-round_count][i] = 1;
			}
			std::cout << "stop" << i << ":" << bus_stop_flag[0][i] << '\n';
		}
	}
	*/
	send_can();
	msgs::Flag_Info msg_temp;
	msg_temp.Dspace_Flag01 = bus_stop_info[0][0];
	msg_temp.Dspace_Flag02 = bus_stop_info[0][1];
	msg_temp.Dspace_Flag03 = bus_stop_info[0][2];
	msg_temp.Dspace_Flag04 = bus_stop_info[0][3];
	msg_temp.Dspace_Flag05 = bus_stop_info[0][4];
	msg_temp.Dspace_Flag06 = bus_stop_info[0][5];
	msg_temp.Dspace_Flag07 = bus_stop_info[0][6];
	msg_temp.Dspace_Flag08 = bus_stop_info[0][7];
	msg_temp.PX2_Flag01 = round_count;
	publisher_01.publish(msg_temp);
}

void chatterCallback_02(const msgs::Flag_Info::ConstPtr& msg)
{
	if(msg->Dspace_Flag02==2 && bus_stop_info[0][int(msg->Dspace_Flag01)-1]==1){
		bus_stop_info[0][int(msg->Dspace_Flag01)-1] = 0;
		double max = *max_element(bus_stop_info[0].begin(), bus_stop_info[0].end());
		if(max==0){
			
			round_count = round_count+1;
			for(uint i=0;i<(bus_stop_info.size()-1);i++){
				bus_stop_info[i] = bus_stop_info[i+1];
			}
			for(uint i=0;i<bus_stop_info.back().size();i++){
				bus_stop_info[bus_stop_info.size()-1][i] = 0;
			}
			std::cout << "Change to round: " << round_count << endl;
		}
		send_can();
	}
	msgs::Flag_Info msg_temp;
	msg_temp.Dspace_Flag01 = bus_stop_info[0][0];
	msg_temp.Dspace_Flag02 = bus_stop_info[0][1];
	msg_temp.Dspace_Flag03 = bus_stop_info[0][2];
	msg_temp.Dspace_Flag04 = bus_stop_info[0][3];
	msg_temp.Dspace_Flag05 = bus_stop_info[0][4];
	msg_temp.Dspace_Flag06 = bus_stop_info[0][5];
	msg_temp.Dspace_Flag07 = bus_stop_info[0][6];
	msg_temp.Dspace_Flag08 = bus_stop_info[0][7];
	msg_temp.PX2_Flag01 = round_count;
	publisher_01.publish(msg_temp);
}


int main(int argc, char **argv)
{
	// Initialize stop info global variable
	for ( int i = 0 ; i < 5 ; i++ ){
		bus_stop_info[i].resize(stop_number);	
	}		 
	ros::init(argc, argv, "bus_stop_info");
	ros::NodeHandle n;
	ros::Subscriber subscriber_01 = n.subscribe("/reserve/request", 1, chatterCallback_01);
	ros::Subscriber subscriber_02 = n.subscribe("/NextStop/Info", 1, chatterCallback_02);
	publisher_01 = n.advertise<msgs::Flag_Info>("/BusStop/Info", 1);
	ros::spin();
	return 0;
}


