/*
 *   File: vehinfo_pub.cpp
 *   Created on: Sep  2018
 *   Author: Bo Chun Xu
 *	 Institute: ITRI ICL U300
 */


#include <ros/ros.h>
#include "std_msgs/Header.h"
// #include "vehinfo_pub/VehInfo.h"
#include "msgs/VehInfo.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <iostream>

#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>

#include <linux/can.h>
#include <linux/can/raw.h>


#define CAN_DLC 8;

msgs::VehInfo msg;

void processFrame(const struct can_frame& frame, msgs::VehInfo &msg) {
        switch (frame.can_id) {
        case 0x301:
        {
                int ego_x_tmp;
                int ego_y_tmp;
                short ego_z_tmp;


                //bus_vehinfo2_receive::VehInfo msg;
                ego_x_tmp = frame.data[0] << 8| frame.data[1] << 16 | frame.data[2] << 24;
                ego_y_tmp = frame.data[3] << 8| frame.data[4] << 16 | frame.data[5] << 24;
                ego_z_tmp = frame.data[6] | frame.data[7]<< 8;

                msg.ego_x = ego_x_tmp/256;
                msg.ego_y = ego_y_tmp/256;
                msg.ego_z = ego_z_tmp;

                msg.ego_x /=100;
                msg.ego_y /=100;
                msg.ego_z /=100;

                std::cout <<  "Got 0x301: " <<
                " msg.ego_x: " << msg.ego_x << " "<<
                " msg.ego_y: " << msg.ego_y << " " <<
                " msg.ego_z: " << msg.ego_z << " " << std::endl;

        }
        break;
        case 0x302:
        {
                short road_id_tmp;
                short lane_width_tmp;
                short yaw_rate_tmp;

                road_id_tmp = frame.data[0] | frame.data[1] << 8;
                lane_width_tmp = frame.data[2];
                yaw_rate_tmp = frame.data[3] | frame.data[4] << 8;

                msg.road_id = road_id_tmp;
                msg.lanewidth = lane_width_tmp;
                msg.lanewidth /=10;
                msg.yaw_rate = yaw_rate_tmp;
                msg.yaw_rate /= 10;

                std::cout << "Got 0x302: " <<
                "road_id: " << msg.road_id <<  " " <<
                "lane_width: " << msg.lanewidth <<  " " <<
                "yaw_rate: " << msg.yaw_rate <<  " " << std::endl;

        }
        break;

        case 0x303:
        {
                int ukf_ego_x_tmp;
                int ukf_ego_y_tmp;
                short ukf_ego_heading_tmp;

                ukf_ego_x_tmp = frame.data[0]<<8 | frame.data[1] << 16 | frame.data[2] << 24;
                ukf_ego_y_tmp = frame.data[3] <<8 | frame.data[4] << 16 | frame.data[5] << 24;
                ukf_ego_heading_tmp = frame.data[6] | frame.data[7]<< 8;

                msg.ukf_ego_x = ukf_ego_x_tmp/256;
                msg.ukf_ego_y = ukf_ego_y_tmp/256;
                msg.ukf_ego_heading = ukf_ego_heading_tmp;

                msg.ukf_ego_x /= 100;
                msg.ukf_ego_y /= 100;
                msg.ukf_ego_heading /= 10;
                std::cout << "Got 0x303:" <<
                "ukf_ego_x: " << msg.ukf_ego_x <<  " " <<
                "ukf_ego_y: " << msg.ukf_ego_y <<  " " <<
                "ukf_ego_heading: " << msg.ukf_ego_heading <<  " " <<std::endl;

        }
        break;

        case 0x304:
        {
                bool gps_fault_flag_tmp;
                short ego_heading_tmp;
                short ego_speed_tmp;

                gps_fault_flag_tmp = frame.data[0];
                ego_heading_tmp = frame.data[1] | frame.data[2]<< 8;
                ego_speed_tmp = frame.data[3] | frame.data[4] << 8;

                msg.gps_fault_flag = gps_fault_flag_tmp;
                msg.ego_heading = ego_heading_tmp;
                msg.ego_speed = ego_speed_tmp;

                msg.ego_heading /= 10;
                msg.ego_speed /= 100;

                std::cout << "Got 0x304:" <<
                "gps_fault_flag: " << unsigned(msg.gps_fault_flag) <<  " " <<
                "ego_heading: " << msg.ego_heading <<  " " <<
                "ego_speed: " << msg.ego_speed <<  " " <<std::endl;
        }
        break;

        case 0x310:
        {
                short path1_to_v_x_tmp;
                short path1_to_v_y_tmp;
                short path2_to_v_x_tmp;
                short path2_to_v_y_tmp;


                path1_to_v_x_tmp = frame.data[0] | frame.data[1]<< 8;
                path1_to_v_y_tmp = frame.data[2] | frame.data[3] << 8;
                path2_to_v_x_tmp = frame.data[4] | frame.data[5]<< 8;
                path2_to_v_y_tmp = frame.data[6] | frame.data[7] << 8;


                msg.path1_to_v_x = path1_to_v_x_tmp/100;
                msg.path1_to_v_y = path1_to_v_y_tmp/100;
                msg.path1_to_v_x = path2_to_v_x_tmp/100;
                msg.path1_to_v_y = path2_to_v_y_tmp/100;


                std::cout << "Got 0x310:" <<
                "path1_to_v_x: " << msg.path1_to_v_x <<  " " <<
                "path1_to_v_y: " << msg.path1_to_v_y <<  " " <<
                "path2_to_v_x: " << msg.path2_to_v_x <<  " " <<
                "path2_to_v_y: " << msg.path2_to_v_y <<  " " <<std::endl;
        }
        break;
        case 0x311:
        {
                short path3_to_v_x_tmp;
                short path3_to_v_y_tmp;
                short path4_to_v_x_tmp;
                short path4_to_v_y_tmp;


                path3_to_v_x_tmp = frame.data[0] | frame.data[1]<< 8;
                path3_to_v_y_tmp = frame.data[2] | frame.data[3] << 8;
                path4_to_v_x_tmp = frame.data[4] | frame.data[5]<< 8;
                path4_to_v_y_tmp = frame.data[6] | frame.data[7] << 8;


                msg.path3_to_v_x = path3_to_v_x_tmp/100;
                msg.path3_to_v_y = path3_to_v_y_tmp/100;
                msg.path4_to_v_x = path4_to_v_x_tmp/100;
                msg.path4_to_v_y = path4_to_v_y_tmp/100;


                std::cout << "Got 0x311:" <<
                "path3_to_v_x: " << msg.path3_to_v_x <<  " " <<
                "path3_to_v_y: " << msg.path3_to_v_y <<  " " <<
                "path4_to_v_x: " << msg.path4_to_v_x <<  " " <<
                "path4_to_v_y: " << msg.path4_to_v_y <<  " " <<std::endl;
        }
        break;
        case 0x312:
        {
                short path5_to_v_x_tmp;
                short path5_to_v_y_tmp;
                short path6_to_v_x_tmp;
                short path6_to_v_y_tmp;


                path5_to_v_x_tmp = frame.data[0] | frame.data[1]<< 8;
                path5_to_v_y_tmp = frame.data[2] | frame.data[3] << 8;
                path6_to_v_x_tmp = frame.data[4] | frame.data[5]<< 8;
                path6_to_v_y_tmp = frame.data[6] | frame.data[7] << 8;


                msg.path5_to_v_x = path5_to_v_x_tmp/100;
                msg.path5_to_v_y = path5_to_v_y_tmp/100;
                msg.path6_to_v_x = path6_to_v_x_tmp/100;
                msg.path6_to_v_y = path6_to_v_y_tmp/100;


                std::cout << "Got 0x312:" <<
                "path5_to_v_x: " << msg.path5_to_v_x <<  " " <<
                "path5_to_v_y: " << msg.path5_to_v_y <<  " " <<
                "path6_to_v_x: " << msg.path6_to_v_x <<  " " <<
                "path6_to_v_y: " << msg.path6_to_v_y <<  " " <<std::endl;
        }
        break;


        default:
                // Should never get here if the receive filters were set up correctly
                std::cerr << "Unexpected CAN ID: 0x"
                          << std::hex   << std::uppercase
                          << std::setw(3) << std::setfill('0')
                          << frame.can_id << std::endl;
                std::cerr.copyfmt(std::ios(nullptr));
                break;
        }
}



int main(int argc, char **argv)
{
        ros::init(argc, argv, "vehinfo_pub");
        ros::NodeHandle nodeHandle1;
        uint32_t seq = 0;

        ros::Publisher vehinfo_pub = nodeHandle1.advertise<msgs::VehInfo>("veh_info", 1);
        int rc;
        int s;
        struct can_filter filter[7];
        filter[0].can_id   = 0x301;
        filter[0].can_mask = CAN_SFF_MASK;
        filter[1].can_id   = 0x302;
        filter[1].can_mask = CAN_SFF_MASK;
        filter[2].can_id   = 0x303;
        filter[2].can_mask = CAN_SFF_MASK;
        filter[3].can_id   = 0x304;
        filter[3].can_mask = CAN_SFF_MASK;

        filter[4].can_id   = 0x310;
        filter[4].can_mask = CAN_SFF_MASK;
        filter[5].can_id   = 0x311;
        filter[5].can_mask = CAN_SFF_MASK;
        filter[6].can_id   = 0x312;
        filter[6].can_mask = CAN_SFF_MASK;

        int cnt = 0;

        int nbytes;
        struct sockaddr_can addr;
        struct can_frame frame;
        struct ifreq ifr;

        const char *ifname = "can1";

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

        // frame.can_id  = CAN_ID;	//
        // frame.can_dlc = CAN_DLC;	//receive byte

        /*
           struct can_frame {
            canid_t can_id;  // 32 bit CAN_ID + EFF/RTR/ERR flags
            __u8    can_dlc; // frame payload length in byte (0 .. CAN_MAX_DLEN)
            __u8    __pad;   // padding
            __u8    __res0;  // reserved / padding
            __u8    __res1;  // reserved / padding
            __u8    data[CAN_MAX_DLEN]-+ __attribute__((aligned(8)));
           };
         */
        ros::Rate rate(100);
        while(ros::ok())
        {
		//bool check_idx[7] = {0};
                //vehinfo_pub::VehInfo msg;

                //nbytes = read(s, &frame, sizeof(struct can_frame));

            msg.header.stamp = ros::Time::now();
            msg.header.frame_id = "vehinfo";
            msg.header.seq = seq++;

            for (int i =0; i < 7; i++)
            {
                    nbytes = read(s, &frame, CANFD_MTU);
                    processFrame(frame, msg);
                    std::cout << "-----------------------------------------------" << std::endl;
            }
		

			std::cout << "published.."<< std::endl;
    		vehinfo_pub.publish(msg);


            ros::spinOnce();
        	rate.sleep();
        }

        return 0;
}
