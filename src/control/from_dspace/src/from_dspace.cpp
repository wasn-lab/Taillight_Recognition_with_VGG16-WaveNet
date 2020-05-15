/*
 *   File: bus_vehinfo_receive_node.cpp
 *   Created on: 2018
 *   Author: Yen, Liang Hsi
 *   Institute: ITRI ICL U300
 */

//Can setup
#define CAN_DLC 8;
#define CAN_CHNNEL "can1"
const int NumOfReceiveID = 4;
const int NumOfTopic = 4;

#include "msgs/Flag_Info.h"
#include "msgs/DynamicPath.h"
#include "msgs/BackendInfo.h"
#include "std_msgs/Header.h"
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
	    Publisher[3].publish(msg_temp);
	    return 1;
	}
    break;
    
    /*
	case 0x3A0:
	{
        int XP1_0_tmp;
        int XP1_1_tmp;
        XP1_0_tmp = frame.data[0] | frame.data[1] << 8 | frame.data[2] << 16 | frame.data[3] << 24;
        XP1_1_tmp = frame.data[4] | frame.data[5] << 8 | frame.data[6] << 16 | frame.data[7] << 24;
        msg.XP1_0 = XP1_0_tmp;
        msg.XP1_1 = XP1_1_tmp;
        msg.XP1_0 /=1000000.0;
        msg.XP1_1 /=1000000.0;
        std::cout <<  "Got 0x3A0: " <<
        " msg.XP1_0: " << std::setprecision(10) << msg.XP1_0 << " "<<
        " msg.XP1_1: " << std::setprecision(10) << msg.XP1_1 << " " << std::endl;
	}
    break;
    case 0x3A1:
 	{

        int XP1_2_tmp;
        int XP1_3_tmp;
        XP1_2_tmp = frame.data[0] | frame.data[1] << 8 | frame.data[2] << 16 | frame.data[3] << 24;
        XP1_3_tmp = frame.data[4] | frame.data[5] << 8 | frame.data[6] << 16 | frame.data[7] << 24;
        msg.XP1_2 = XP1_2_tmp;
        msg.XP1_3 = XP1_3_tmp;
        msg.XP1_2 /=1000000.0;
        msg.XP1_3 /=1000000.0;
        std::cout <<  "Got 0x3A1: " <<
        " msg.XP1_2: " << std::setprecision(10) << msg.XP1_2 << " "<<
        " msg.XP1_3: " << std::setprecision(10) << msg.XP1_3 << " " << std::endl;
	}
    break;
    case 0x3A2:
	{
       
		int XP1_4_tmp;
        int XP1_5_tmp;
        XP1_4_tmp = frame.data[0] | frame.data[1] << 8 | frame.data[2] << 16 | frame.data[3] << 24;
        XP1_5_tmp = frame.data[4] | frame.data[5] << 8 | frame.data[6] << 16 | frame.data[7] << 24;
        msg.XP1_4 = XP1_4_tmp;
        msg.XP1_5 = XP1_5_tmp;
        msg.XP1_4 /=1000000.0;
        msg.XP1_5 /=1000000.0;
        std::cout <<  "Got 0x3A2: " <<
        " msg.XP1_4: " << std::setprecision(10) << msg.XP1_4 << " "<<
        " msg.XP1_5: " << std::setprecision(10) << msg.XP1_5 << " " << std::endl;
	}
    break;

    case 0x3A3:
	{
        int YP1_0_tmp;
        int YP1_1_tmp;
        YP1_0_tmp = frame.data[0] | frame.data[1] << 8 | frame.data[2] << 16 | frame.data[3] << 24;
        YP1_1_tmp = frame.data[4] | frame.data[5] << 8 | frame.data[6] << 16 | frame.data[7] << 24;
        msg.YP1_0 = YP1_0_tmp;
        msg.YP1_1 = YP1_1_tmp;
        msg.YP1_0 /=1000000.0;
        msg.YP1_1 /=1000000.0;
        std::cout <<  "Got 0x3A3: " <<
        " msg.YP1_0: " << std::setprecision(10) << msg.YP1_0 << " "<<
        " msg.YP1_1: " << std::setprecision(10) << msg.YP1_1 << " " << std::endl;
	}
    break;
    case 0x3A4:
	{
       
		int YP1_2_tmp;
        int YP1_3_tmp;
        YP1_2_tmp = frame.data[0] | frame.data[1] << 8 | frame.data[2] << 16 | frame.data[3] << 24;
        YP1_3_tmp = frame.data[4] | frame.data[5] << 8 | frame.data[6] << 16 | frame.data[7] << 24;
        msg.YP1_2 = YP1_2_tmp;
        msg.YP1_3 = YP1_3_tmp;
        msg.YP1_2 /=1000000.0;
        msg.YP1_3 /=1000000.0;
        std::cout <<  "Got 0x3A4: " <<
        " msg.YP1_2: " << std::setprecision(10) << msg.YP1_2 << " "<<
        " msg.YP1_3: " << std::setprecision(10) << msg.YP1_3 << " " << std::endl;
	}
    break;
    case 0x3A5:
	{       
		int YP1_4_tmp;
        int YP1_5_tmp;
        YP1_4_tmp = frame.data[0] | frame.data[1] << 8 | frame.data[2] << 16 | frame.data[3] << 24;
        YP1_5_tmp = frame.data[4] | frame.data[5] << 8 | frame.data[6] << 16 | frame.data[7] << 24;
        msg.YP1_4 = YP1_4_tmp;
        msg.YP1_5 = YP1_5_tmp;
        msg.YP1_4 /=1000000.0;
        msg.YP1_5 /=1000000.0;
        std::cout <<  "Got 0x3A5: " <<
        " msg.YP1_4: " << std::setprecision(10) << msg.YP1_4 << " "<<
        " msg.YP1_5: " << std::setprecision(10) << msg.YP1_5 << " " << std::endl;
	}
    break;

    case 0x3A6:
	{      
		int XP2_0_tmp;
        int XP2_1_tmp;
        XP2_0_tmp = frame.data[0] | frame.data[1] << 8 | frame.data[2] << 16 | frame.data[3] << 24;
        XP2_1_tmp = frame.data[4] | frame.data[5] << 8 | frame.data[6] << 16 | frame.data[7] << 24;
        msg.XP2_0 = XP2_0_tmp;
        msg.XP2_1 = XP2_1_tmp;
        msg.XP2_0 /=1000000.0;
        msg.XP2_1 /=1000000.0;
        std::cout <<  "Got 0x3A6: " <<
        " msg.XP2_0: " << std::setprecision(10) << msg.XP2_0 << " "<<
        " msg.XP2_1: " << std::setprecision(10) << msg.XP2_1 << " " << std::endl;
	}
    break;
    case 0x3A7:
	{
        int XP2_2_tmp;
        int XP2_3_tmp;
        XP2_2_tmp = frame.data[0] | frame.data[1] << 8 | frame.data[2] << 16 | frame.data[3] << 24;
        XP2_3_tmp = frame.data[4] | frame.data[5] << 8 | frame.data[6] << 16 | frame.data[7] << 24;
        msg.XP2_2 = XP2_2_tmp;
        msg.XP2_3 = XP2_3_tmp;
        msg.XP2_2 /=1000000.0;
        msg.XP2_3 /=1000000.0;
        std::cout <<  "Got 0x3A7: " <<
        " msg.XP2_2: " << std::setprecision(10) << msg.XP2_2 << " "<<
        " msg.XP2_3: " << std::setprecision(10) << msg.XP2_3 << " " << std::endl;
	}
    break;
    case 0x3A8:
	{
        int XP2_4_tmp;
        int XP2_5_tmp;
        XP2_4_tmp = frame.data[0] | frame.data[1] << 8 | frame.data[2] << 16 | frame.data[3] << 24;
        XP2_5_tmp = frame.data[4] | frame.data[5] << 8 | frame.data[6] << 16 | frame.data[7] << 24;
        msg.XP2_4 = XP2_4_tmp;
        msg.XP2_5 = XP2_5_tmp;
        msg.XP2_4 /=1000000.0;
        msg.XP2_5 /=1000000.0;
        std::cout <<  "Got 0x3A8: " <<
        " msg.XP2_4: " << std::setprecision(10) << msg.XP2_4 << " "<<
        " msg.XP2_5: " << std::setprecision(10) << msg.XP2_5 << " " << std::endl;
	}
    break;
    case 0x3A9:
	{
        int YP2_0_tmp;
        int YP2_1_tmp;
        YP2_0_tmp = frame.data[0] | frame.data[1] << 8 | frame.data[2] << 16 | frame.data[3] << 24;
        YP2_1_tmp = frame.data[4] | frame.data[5] << 8 | frame.data[6] << 16 | frame.data[7] << 24;
        msg.YP2_0 = YP2_0_tmp;
        msg.YP2_1 = YP2_1_tmp;
        msg.YP2_0 /=1000000.0;
        msg.YP2_1 /=1000000.0;
        std::cout <<  "Got 0x3A9: " <<
        " msg.YP2_0: " << std::setprecision(10) << msg.YP2_0 << " "<<
        " msg.YP2_1: " << std::setprecision(10) << msg.YP2_1 << " " << std::endl;
	}
    break;
    case 0x3B0:
	{
        int YP2_2_tmp;
        int YP2_3_tmp;
        YP2_2_tmp = frame.data[0] | frame.data[1] << 8 | frame.data[2] << 16 | frame.data[3] << 24;
        YP2_3_tmp = frame.data[4] | frame.data[5] << 8 | frame.data[6] << 16 | frame.data[7] << 24;
        msg.YP2_2 = YP2_2_tmp;
        msg.YP2_3 = YP2_3_tmp;
        msg.YP2_2 /=1000000.0;
        msg.YP2_3 /=1000000.0;
        std::cout <<  "Got 0x3B0: " <<
        " msg.YP2_2: " << std::setprecision(10) << msg.YP2_2 << " "<<
        " msg.YP2_3: " << std::setprecision(10) << msg.YP2_3 << " " << std::endl;
	}
    break;
    case 0x3B1:
	{
	    int YP2_4_tmp;
	    int YP2_5_tmp;
	    YP2_4_tmp = frame.data[0] | frame.data[1] << 8 | frame.data[2] << 16 | frame.data[3] << 24;
	    YP2_5_tmp = frame.data[4] | frame.data[5] << 8 | frame.data[6] << 16 | frame.data[7] << 24;
	    msg.YP2_4 = YP2_4_tmp;
	    msg.YP2_5 = YP2_5_tmp;
	    msg.YP2_4 /=1000000.0;
	    msg.YP2_5 /=1000000.0;
	    std::cout <<  "Got 0x3B1: " <<
	    " msg.YP2_4: " << std::setprecision(10) << msg.YP2_4 << " "<<
	    " msg.YP2_5: " << std::setprecision(10) << msg.YP2_5 << " " << std::endl;
		
	}
    break;
    */
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
    Publisher[3] = n.advertise<msgs::Flag_Info>("/NextStop/Info", 1);
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

