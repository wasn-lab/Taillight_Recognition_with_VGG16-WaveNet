/*
 *   File: bus_vehinfo_receive_node.cpp
 *   Created on: 2018
 *   Author: Yen, Liang Hsi
 *   Institute: ITRI ICL U300
 */

//Can setup
#define CAN_DLC 8;
#define CAN_CHNNEL "can1"
const int NumOfReceiveID = 5;
const int NumOfTopic = 8;

#include "msgs/Flag_Info.h"
#include "msgs/DynamicPath.h"
#include "msgs/BackendInfo.h"
#include "std_msgs/Header.h"
#include "std_msgs/Float64.h"
#include <ros/ros.h>



#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <fstream>

// For CAN Bus
#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
using namespace std ;
msgs::DynamicPath msg123;


int ProcessFrame(const struct can_frame& frame, ros::Publisher* Publisher, msgs::DynamicPath &msg) {
    switch (frame.can_id) {
    case 0x601:
	{
	    msgs::Flag_Info msg_temp;
		msg_temp.Dspace_Flag01 = frame.data[0];
		msg_temp.Dspace_Flag02 = frame.data[1];
		msg_temp.Dspace_Flag03 = frame.data[2];
		msg_temp.Dspace_Flag04 = frame.data[3];
		msg_temp.Dspace_Flag05 = frame.data[4];
		msg_temp.Dspace_Flag06 = frame.data[5];
		msg_temp.Dspace_Flag07 = frame.data[6];
		msg_temp.Dspace_Flag08 = frame.data[7];

	    cout << " Flag01: " << msg_temp.Dspace_Flag01 << endl;
	    cout << " Flag02: " << msg_temp.Dspace_Flag02 << endl;
	    cout << " Flag03: " << msg_temp.Dspace_Flag03 << endl;
	    cout << " Flag04: " << msg_temp.Dspace_Flag04 << endl;
	    cout << " Flag05: " << msg_temp.Dspace_Flag05 << endl;
	    cout << " Flag06: " << msg_temp.Dspace_Flag06 << endl;
	    cout << " Flag07: " << msg_temp.Dspace_Flag07 << endl;
	    cout << " Flag08: " << msg_temp.Dspace_Flag08 << endl;

	    Publisher[0].publish(msg_temp);
	    return 1;
	}
    break;

	case 0x602:
	{
	    msgs::Flag_Info msg_temp;
		msg_temp.Dspace_Flag01 = frame.data[0];
		msg_temp.Dspace_Flag02 = frame.data[1];
		msg_temp.Dspace_Flag03 = frame.data[2];
		msg_temp.Dspace_Flag04 = frame.data[3];
		msg_temp.Dspace_Flag05 = frame.data[4];
		msg_temp.Dspace_Flag06 = frame.data[5];
		msg_temp.Dspace_Flag07 = frame.data[6];
		msg_temp.Dspace_Flag08 = frame.data[7];

	    cout << " Flag01: " << msg_temp.Dspace_Flag01 << endl;
	    cout << " Flag02: " << msg_temp.Dspace_Flag02 << endl;
	    cout << " Flag03: " << msg_temp.Dspace_Flag03 << endl;
	    cout << " Flag04: " << msg_temp.Dspace_Flag04 << endl;
	    cout << " Flag05: " << msg_temp.Dspace_Flag05 << endl;
	    cout << " Flag06: " << msg_temp.Dspace_Flag06 << endl;
	    cout << " Flag07: " << msg_temp.Dspace_Flag07 << endl;
	    cout << " Flag08: " << msg_temp.Dspace_Flag08 << endl;

	    Publisher[1].publish(msg_temp);
	    return 1;
	}
    break;

	case 0x603:
	{
	    msgs::Flag_Info msg_temp;
		msg_temp.Dspace_Flag01 = frame.data[0];
		msg_temp.Dspace_Flag02 = frame.data[1];
		msg_temp.Dspace_Flag03 = frame.data[2];
		msg_temp.Dspace_Flag04 = frame.data[3];
		msg_temp.Dspace_Flag05 = frame.data[4];
		msg_temp.Dspace_Flag06 = frame.data[5];
		msg_temp.Dspace_Flag07 = frame.data[6];
		msg_temp.Dspace_Flag08 = frame.data[7];

	    cout << " Flag01: " << msg_temp.Dspace_Flag01 << endl;
	    cout << " Flag02: " << msg_temp.Dspace_Flag02 << endl;
	    cout << " Flag03: " << msg_temp.Dspace_Flag03 << endl;
	    cout << " Flag04: " << msg_temp.Dspace_Flag04 << endl;
	    cout << " Flag05: " << msg_temp.Dspace_Flag05 << endl;
	    cout << " Flag06: " << msg_temp.Dspace_Flag06 << endl;
	    cout << " Flag07: " << msg_temp.Dspace_Flag07 << endl;
	    cout << " Flag08: " << msg_temp.Dspace_Flag08 << endl;

	    Publisher[2].publish(msg_temp);
	    return 1;
	}
    break;

    case 0x610:
	{
	    msgs::Flag_Info msg_temp;
		msg_temp.Dspace_Flag01 = frame.data[0];
		msg_temp.Dspace_Flag02 = frame.data[1];
		msg_temp.Dspace_Flag03 = frame.data[2];
		msg_temp.Dspace_Flag04 = frame.data[3];
		msg_temp.Dspace_Flag05 = frame.data[4];
		msg_temp.Dspace_Flag06 = frame.data[5];
		msg_temp.Dspace_Flag07 = frame.data[6];
		msg_temp.Dspace_Flag08 = frame.data[7];

	    cout << " Next Stop: " << msg_temp.Dspace_Flag01 << endl;
	    cout << " Stop status: " << msg_temp.Dspace_Flag02 << endl;
	    Publisher[5].publish(msg_temp);
	    return 1;
	}
    break;

    case 0x351:
	{
        std_msgs::Float64 speed_kph;
        std_msgs::Float64 speed_ms ;
	    msgs::Flag_Info msg_temp;
        speed_kph.data = frame.data[0];
        Publisher[6].publish(speed_kph);
        cout << "speed_kph: " << speed_kph.data << endl;
        speed_ms.data = speed_kph.data/3.6;
        Publisher[7].publish(speed_ms);
        cout << "speed_ms: " << speed_ms.data << endl;
		msg_temp.Dspace_Flag01 = frame.data[0];
		msg_temp.Dspace_Flag02 = frame.data[1];
		msg_temp.Dspace_Flag03 = frame.data[2];
		msg_temp.Dspace_Flag04 = frame.data[3];
		msg_temp.Dspace_Flag05 = frame.data[4];
		msg_temp.Dspace_Flag06 = frame.data[5];
		msg_temp.Dspace_Flag07 = frame.data[6];
		msg_temp.Dspace_Flag08 = frame.data[7];
	    Publisher[3].publish(msg_temp);
	    return 1;
	}
    break;

    case 0x301:
	{

	    return 1;
	}
    break;

    case 0x302:
	{

	    return 1;
	}
    break;

    case 0x303:
	{
	    return 1;
	}
    break;

    case 0x304:
	{
	    return 1;
	}
    break;
    
    case 0x350:
	{
	    return 1;
	}
    break;

    default:
		{
		    // Should never get here if the receive filters were set up correctly
		    std::cerr << "Unexpected CAN ID: 0x"
		                << std::hex   << std::uppercase
		                << std::setw(3) << std::setfill('0')
		                << frame.can_id << std::endl;
		    std::cerr.copyfmt(std::ios(nullptr));
			return -1;
		}
    }
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "from_dspace");
    ros::NodeHandle n;
    //ros::Publisher Publisher01 = n.advertise<msgs::Flag_Info>("Flag_Info01", 1);
	//ros::Publisher Publisher02 = n.advertise<msgs::Flag_Info>("Flag_Info02", 1);
	//ros::Publisher Publisher03 = n.advertise<msgs::Flag_Info>("Flag_Info03", 1);
	ros::Publisher Publisher[NumOfTopic];
	Publisher[0] = n.advertise<msgs::Flag_Info>("Flag_Info01", 1);
	Publisher[1] = n.advertise<msgs::Flag_Info>("Flag_Info02", 1);
	Publisher[2] = n.advertise<msgs::Flag_Info>("Flag_Info03", 1);
    Publisher[3] = n.advertise<msgs::Flag_Info>("Flag_Info04", 1);
    Publisher[4] = n.advertise<msgs::Flag_Info>("Flag_Info05", 1);
    Publisher[5] = n.advertise<msgs::Flag_Info>("/NextStop/Info", 1);
    Publisher[6] = n.advertise<msgs::Flag_Info>("/Ego_speed/kph", 1);
    Publisher[7] = n.advertise<msgs::Flag_Info>("/Ego_speed/ms", 1);
	//Publisher[3] = n.advertise<msgs::DynamicPath>("dynamic_path_para_test", 1);
	//uint32_t seq = 0;
    ros::Publisher Publisher_BD;
    Publisher_BD = n.advertise<msgs::BackendInfo>("Backend/Info", 1);

    int rc;
	struct can_filter filter[NumOfReceiveID];
    for(int i=0 ;i <NumOfReceiveID; i++)
    {
        filter[i].can_mask = CAN_SFF_MASK;
    }
	filter[0].can_id = 0x601;
	filter[1].can_id = 0x602;
	filter[2].can_id = 0x603;
    filter[3].can_id = 0x610;
    filter[4].can_id = 0x351;

    /*
	filter[3].can_id = 0x3A0;
	filter[4].can_id = 0x3A1;
	filter[5].can_id = 0x3A2;
	filter[6].can_id = 0x3A3;
	filter[7].can_id = 0x3A4;
	filter[8].can_id = 0x3A5;
	filter[9].can_id = 0x3A6;
	filter[10].can_id = 0x3A7;
	filter[11].can_id = 0x3A8;
	filter[12].can_id = 0x3A9;
	filter[13].can_id = 0x3B0;
	filter[14].can_id = 0x3B1;
    */

    int s;
    const char *ifname = CAN_CHNNEL;
    int nbytes;
    struct sockaddr_can addr;
    struct can_frame frame;
    struct ifreq ifr;


    if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) {
        perror("Error while opening socket");
        return -1;
    }

    rc = setsockopt(s, SOL_CAN_RAW, CAN_RAW_FILTER, &filter, sizeof(filter));
    if (-1 == rc)
    {
        std::perror("setsockopt filter");
    }

    strcpy(ifr.ifr_name, ifname);
    ioctl(s, SIOCGIFINDEX, &ifr);

    addr.can_family  = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    printf("%s at index %d\n", ifname, ifr.ifr_ifindex);

    if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Error in socket bind");
        return -2;
    }

    ros::Rate rate(10);
    while(ros::ok())
    {
    	// For msgs that need more than one CAN_ID
    	//msgs::DynamicPath msg;
        //msg123.header.stamp = ros::Time::now();
        //msg123.header.frame_id = "dynamicpath";
        //msg123.header.seq = seq++;
        for (int i =0; i <NumOfReceiveID; i++)
        {
            nbytes = read(s, &frame, sizeof(struct can_frame));
            printf("Read %d bytes\n", nbytes);
            ProcessFrame(frame, Publisher, msg123);
        }
        msgs::BackendInfo msg123;
        Publisher_BD.publish(msg123);
        rate.sleep();
    }
    return 0;
}

