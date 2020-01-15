/*
 *   File: bus_vehinfo_receive_node.cpp
 *   Created on: 2018
 *   Author: Yen, Liang Hsi
 *   Institute: ITRI ICL U300
 */

#define CAN_ID 0x603;
#define CAN_DLC 8;
#include "msgs/Flag_Info.h"
#include "std_msgs/Header.h"
#include <ros/ros.h>
#include "std_msgs/Header.h"


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <iostream>

// For CAN Bus
#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>

using namespace std ;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Flag_Info03");
    ros::NodeHandle nodeHandle1;
    ros::Publisher Dspace_Flag_pub = nodeHandle1.advertise<msgs::Flag_Info>("Flag_Info03", 1);
    int rc;
    struct can_filter filter[1];
    filter[0].can_id   = CAN_ID;
    filter[0].can_mask = CAN_SFF_MASK;


    int s;
    int nbytes;
    struct sockaddr_can addr;
    struct can_frame frame;
    struct ifreq ifr;

    const char *ifname = "can1";

    if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
    {
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

    if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        perror("Error in socket bind");
        return -2;
    }

    ros::Rate rate(10);

    while(ros::ok())
    {
        msgs::Flag_Info msg;
        nbytes = read(s, &frame, sizeof(struct can_frame));

    	msg.Dspace_Flag01 = frame.data[0];
    	msg.Dspace_Flag02 = frame.data[1];
	    msg.Dspace_Flag03 = frame.data[2];
	    msg.Dspace_Flag04 = frame.data[3];
	    msg.Dspace_Flag05 = frame.data[4];
	    msg.Dspace_Flag06 = frame.data[5];
	    msg.Dspace_Flag07 = frame.data[6];
	    msg.Dspace_Flag08 = frame.data[7];
	
        cout << " Flag01: " << msg.Dspace_Flag01 << endl;
        cout << " Flag02: " << msg.Dspace_Flag02 << endl;
        cout << " Flag03: " << msg.Dspace_Flag03 << endl;
        cout << " Flag04: " << msg.Dspace_Flag04 << endl;
        cout << " Flag05: " << msg.Dspace_Flag05 << endl;
        cout << " Flag06: " << msg.Dspace_Flag06 << endl;
        cout << " Flag07: " << msg.Dspace_Flag07 << endl;
        cout << " Flag08: " << msg.Dspace_Flag08 << endl;

        Dspace_Flag_pub.publish(msg);
        ros::spinOnce();
        //rate.sleep();
    }

    return 0;
}

