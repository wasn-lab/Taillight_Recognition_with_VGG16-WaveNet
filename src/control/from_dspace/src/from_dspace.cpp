/*
 *   File: bus_vehinfo_receive_node.cpp
 *   Created on: 2018
 *   Author: Yen, Liang Hsi
 *   Institute: ITRI ICL U300
 */

//Can setup
#define CAN_DLC 8;
// #define CAN_CHNNEL "can1"
const int NumOfReceiveID = 10;
const int NumOfTopic = 8;

#include "msgs/Flag_Info.h"
#include "msgs/DynamicPath.h"
#include "msgs/BackendInfo.h"
#include "std_msgs/Header.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Int8MultiArray.h"
#include "msgs/VehInfo.h"
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
msgs::VehInfo msg_VehInfo;
msgs::BackendInfo msg_Backend;



int ProcessFrame(const struct can_frame& frame, ros::Publisher* Publisher) {
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
        double steer_angle_fb;
		msg_temp.Dspace_Flag01 = frame.data[0];
		msg_temp.Dspace_Flag02 = frame.data[1] / 10.0;
		msg_temp.Dspace_Flag03 = frame.data[2] / 10.0;
        steer_angle_fb = frame.data[3] | frame.data[4] << 8;
		msg_temp.Dspace_Flag04 = steer_angle_fb / 10.0 - 1000.0;
		msg_temp.Dspace_Flag05 = frame.data[5];
		msg_temp.Dspace_Flag06 = frame.data[6];
		msg_temp.Dspace_Flag07 = frame.data[7];
	    Publisher[3].publish(msg_temp);
	    return 1;
	}
    break;

    // case 0x301:
    // {
    //     int ego_x_tmp;
    //     int ego_y_tmp;
    //     short ego_z_tmp;

    //     //bus_vehinfo2_receive::VehInfo msg;
    //     ego_x_tmp = frame.data[0] << 8| frame.data[1] << 16 | frame.data[2] << 24;
    //     ego_y_tmp = frame.data[3] << 8| frame.data[4] << 16 | frame.data[5] << 24;
    //     ego_z_tmp = frame.data[6] | frame.data[7]<< 8;

    //     msg_VehInfo.ego_x = ego_x_tmp/256;
    //     msg_VehInfo.ego_y = ego_y_tmp/256;
    //     msg_VehInfo.ego_z = ego_z_tmp;

    //     msg_VehInfo.ego_x /=100;
    //     msg_VehInfo.ego_y /=100;
    //     msg_VehInfo.ego_z /=100;

    //     std::cout <<  "Got 0x301: " <<
    //     " msg_VehInfo.ego_x: " << msg_VehInfo.ego_x << " "<<
    //     " msg_VehInfo.ego_y: " << msg_VehInfo.ego_y << " " <<
    //     " msg_VehInfo.ego_z: " << msg_VehInfo.ego_z << " " << std::endl;
    //     return 1;
    // }
    // break;

    // case 0x302:
    // {
    //     short road_id_tmp;
    //     short lane_width_tmp;
    //     short yaw_rate_tmp;

    //     road_id_tmp = frame.data[0] | frame.data[1] << 8;
    //     lane_width_tmp = frame.data[2];
    //     yaw_rate_tmp = frame.data[3] | frame.data[4] << 8;

    //     msg_VehInfo.road_id = road_id_tmp;
    //     msg_VehInfo.lanewidth = lane_width_tmp;
    //     msg_VehInfo.lanewidth /=10;
    //     msg_VehInfo.yaw_rate = yaw_rate_tmp;
    //     msg_VehInfo.yaw_rate /= 10;

    //     std::cout << "Got 0x302: " <<
    //     "road_id: " << msg_VehInfo.road_id <<  " " <<
    //     "lane_width: " << msg_VehInfo.lanewidth <<  " " <<
    //     "yaw_rate: " << msg_VehInfo.yaw_rate <<  " " << std::endl;
    //     return 1;
    // }
    // break;

    // case 0x303:
    // {
    //     int ukf_ego_x_tmp;
    //     int ukf_ego_y_tmp;
    //     short ukf_ego_heading_tmp;

    //     ukf_ego_x_tmp = frame.data[0]<<8 | frame.data[1] << 16 | frame.data[2] << 24;
    //     ukf_ego_y_tmp = frame.data[3] <<8 | frame.data[4] << 16 | frame.data[5] << 24;
    //     ukf_ego_heading_tmp = frame.data[6] | frame.data[7]<< 8;

    //     msg_VehInfo.ukf_ego_x = ukf_ego_x_tmp/256;
    //     msg_VehInfo.ukf_ego_y = ukf_ego_y_tmp/256;
    //     msg_VehInfo.ukf_ego_heading = ukf_ego_heading_tmp;

    //     msg_VehInfo.ukf_ego_x /= 100;
    //     msg_VehInfo.ukf_ego_y /= 100;
    //     msg_VehInfo.ukf_ego_heading /= 10;
    //     std::cout << "Got 0x303:" <<
    //     "ukf_ego_x: " << msg_VehInfo.ukf_ego_x <<  " " <<
    //     "ukf_ego_y: " << msg_VehInfo.ukf_ego_y <<  " " <<
    //     "ukf_ego_heading: " << msg_VehInfo.ukf_ego_heading <<  " " <<std::endl;
    //     return 1;
    // }
    // break;

    // case 0x304:
    // {
    //     bool gps_fault_flag_tmp;
    //     short ego_heading_tmp;
    //     short ego_speed_tmp;

    //     gps_fault_flag_tmp = frame.data[0];
    //     ego_heading_tmp = frame.data[1] | frame.data[2]<< 8;
    //     ego_speed_tmp = frame.data[3] | frame.data[4] << 8;

    //     msg_VehInfo.gps_fault_flag = gps_fault_flag_tmp;
    //     msg_VehInfo.ego_heading = ego_heading_tmp;
    //     msg_VehInfo.ego_speed = ego_speed_tmp;

    //     msg_VehInfo.ego_heading /= 10;
    //     msg_VehInfo.ego_speed /= 100;

    //     std::cout << "Got 0x304:" <<
    //     "gps_fault_flag: " << unsigned(msg_VehInfo.gps_fault_flag) <<  " " <<
    //     "ego_heading: " << msg_VehInfo.ego_heading <<  " " <<
    //     "ego_speed: " << msg_VehInfo.ego_speed <<  " " <<std::endl;
    //     return 1;
    // }
    // break;
    
    // case 0x350:
	// {
	//     return 1;
	// }
    // break;

    case 0x620:
    {
        bool gps_fault_flag_tmp;
        short ego_heading_tmp;
        short ego_speed_tmp;
        msg_Backend.door = frame.data[6]&1;
        msg_Backend.headlight = (frame.data[6]>>1)&1;
        msg_Backend.left_turn_light = (frame.data[6]>>2)&1;
        msg_Backend.right_turn_light = (frame.data[6]>>3)&1;
        msg_Backend.air_conditioner = (frame.data[6]>>4)&1;
        msg_Backend.estop = (frame.data[6]>>5)&1;
        msg_Backend.wiper = (frame.data[6]>>6)&1;
        msg_Backend.hand_brake = (frame.data[6]>>7)&1;
        msg_Backend.indoor_light = frame.data[7]&1;
        msg_Backend.gross_power = (frame.data[7]>>1)&1;
        msg_Backend.ACC_state = (frame.data[7]>>2)&1;

        std::cout << "Got 0x620" << endl;
        return 1;
    }
    break;

    case 0x621:
    {


        msg_Backend.mileage = frame.data[0] | frame.data[1]<< 8 | frame.data[2]<< 16 | frame.data[3]<< 24 ;
        msg_Backend.speed = frame.data[4];
        msg_Backend.air_pressure = frame.data[5];
        // 電門踏板 = frame.data[6];
        // 速度命令 = frame.data[7];

        std::cout << "Got 0x621" << endl;

        return 1;
    }
    break;

    case 0x622:
    {

        msg_Backend.gross_current = frame.data[0] | frame.data[1]<< 8;
        msg_Backend.highest_voltage = frame.data[2] | frame.data[3]<< 8;
        msg_Backend.highest_voltage = msg_Backend.highest_voltage/100;
        msg_Backend.highest_number = frame.data[4];
        msg_Backend.lowest_volage = frame.data[5] | frame.data[6]<< 8;
        msg_Backend.lowest_volage = msg_Backend.lowest_volage/100;
        msg_Backend.lowest_number = frame.data[7];

        std::cout << "Got 0x622" << endl;
        return 1;
    }
    break;

    case 0x623:
    {
        msg_Backend.voltage_deviation = frame.data[0];
        msg_Backend.voltage_deviation = msg_Backend.voltage_deviation/100;
        msg_Backend.highest_temperature = frame.data[1] | frame.data[2]<< 8;
        msg_Backend.highest_temp_location = frame.data[3];
        msg_Backend.gross_voltage = frame.data[4] | frame.data[5]<< 8;
        // 控制電壓 = frame.data[6];
        msg_Backend.battery = frame.data[7];

        std::cout << "Got 0x623" << endl;
        return 1;
    }
    break;

    case 0x624:
    {
        msg_Backend.motor_temperature = frame.data[0] | frame.data[1]<< 8;
        msg_Backend.motor_temperature = msg_Backend.motor_temperature/10;
        // life signal = (frame.data[7]>>7)&1;

        std::cout << "Got 0x624" << endl;
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

    std::string can_name_ = "can1";
    ros::param::get(ros::this_node::getName()+"/can_name", can_name_);

	ros::Publisher Publisher[NumOfTopic];
	Publisher[0] = n.advertise<msgs::Flag_Info>("Flag_Info01", 100);
	Publisher[1] = n.advertise<msgs::Flag_Info>("Flag_Info02", 100);
	Publisher[2] = n.advertise<msgs::Flag_Info>("Flag_Info03", 100);
    Publisher[3] = n.advertise<msgs::Flag_Info>("Flag_Info04", 100);
    Publisher[4] = n.advertise<msgs::Flag_Info>("Flag_Info05", 100);
    Publisher[5] = n.advertise<msgs::Flag_Info>("/NextStop/Info", 100);
    Publisher[6] = n.advertise<std_msgs::Float64>("/Ego_speed/kph", 100);
    Publisher[7] = n.advertise<std_msgs::Float64>("/Ego_speed/ms", 100);

    ros::Publisher Publisher_Backend;
    Publisher_Backend = n.advertise<msgs::BackendInfo>("Backend/Info", 100);
    // ros::Publisher vehinfo_pub;
    // vehinfo_pub = n.advertise<msgs::VehInfo>("veh_info", 10);

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
    // filter[4].can_id = 0x301;
    // filter[5].can_id = 0x302;
    // filter[6].can_id = 0x303;
    // filter[7].can_id = 0x304;
    // filter[8].can_id = 0x350;
    filter[4].can_id = 0x351;
    filter[5].can_id = 0x620;
    filter[6].can_id = 0x621;
    filter[7].can_id = 0x622;
    filter[8].can_id = 0x623;
    filter[9].can_id = 0x624;
    

    int s;
    const char *ifname = can_name_.c_str();//CAN_CHNNEL;
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
            printf("from_dspace : Read %d bytes\n", nbytes);
            ProcessFrame(frame, Publisher);
        }
        Publisher_Backend.publish(msg_Backend);
        //vehinfo_pub.publish(msg_VehInfo);
        rate.sleep();
    }
    return 0;
}

