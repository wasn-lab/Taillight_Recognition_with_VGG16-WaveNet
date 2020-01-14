
#include <ros/ros.h>
#include "std_msgs/Header.h"

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
#define SendToDspace "can1"
#define ReveiveFromXBW "can0"
const int NumOfReceiveID = 10;

int ProcessFrame(const struct can_frame& frame) {
        switch (frame.can_id) {
        
        case 0x130:
        {
            int s;
            int nbytes;
            struct sockaddr_can addr;
            struct can_frame frameout;
            struct ifreq ifr;
            const char *ifname = SendToDspace;

            if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
            {
                perror("Error while opening socket");
                return -1;
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

            frameout.can_id  = "0xa30" ;
            frameout.can_dlc = CAN_DLC ;
            frameout.data[0] = frame.data[0] ;
            frameout.data[1] = frame.data[1] ;
            frameout.data[2] = frame.data[2] ;
            frameout.data[3] = frame.data[3] ;
            frameout.data[4] = frame.data[4] ;
            frameout.data[5] = frame.data[5] ;
            frameout.data[6] = frame.data[6] ;
            frameout.data[7] = frame.data[7] ;
            
            cout << " Receivced 0x130, Sent 0xa30 " << endl;

            nbytes = write(s, &frameout, sizeof(struct can_frame));
            printf("Wrote %d bytes\n", nbytes);
            //Close the SocketCAN
            close(s);
            return nbytes;
        }
        break;

        case 0x230:
        {
            int s;
            int nbytes;
            struct sockaddr_can addr;
            struct can_frame frameout;
            struct ifreq ifr;
            const char *ifname = SendToDspace;

            if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
            {
                perror("Error while opening socket");
                return -1;
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

            frameout.can_id  = "0xb30" ;
            frameout.can_dlc = CAN_DLC ;
            frameout.data[0] = frame.data[0] ;
            frameout.data[1] = frame.data[1] ;
            frameout.data[2] = frame.data[2] ;
            frameout.data[3] = frame.data[3] ;
            frameout.data[4] = frame.data[4] ;
            frameout.data[5] = frame.data[5] ;
            frameout.data[6] = frame.data[6] ;
            frameout.data[7] = frame.data[7] ;
            
            cout << " Receivced 0x230, Sent 0xb30 " << endl;

            nbytes = write(s, &frameout, sizeof(struct can_frame));
            printf("Wrote %d bytes\n", nbytes);
            //Close the SocketCAN
            close(s);
            return nbytes;
        }
        break;

        case 0x231:
        {
            int s;
            int nbytes;
            struct sockaddr_can addr;
            struct can_frame frameout;
            struct ifreq ifr;
            const char *ifname = SendToDspace;

            if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
            {
                perror("Error while opening socket");
                return -1;
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

            frameout.can_id  = "0xb310" ;
            frameout.can_dlc = CAN_DLC ;
            frameout.data[0] = frame.data[0] ;
            frameout.data[1] = frame.data[1] ;
            frameout.data[2] = frame.data[2] ;
            frameout.data[3] = frame.data[3] ;
            frameout.data[4] = frame.data[4] ;
            frameout.data[5] = frame.data[5] ;
            frameout.data[6] = frame.data[6] ;
            frameout.data[7] = frame.data[7] ;
            
            cout << " Receivced 0x231, Sent 0xb31 " << endl;

            nbytes = write(s, &frameout, sizeof(struct can_frame));
            printf("Wrote %d bytes\n", nbytes);
            //Close the SocketCAN
            close(s);
            return nbytes;
        }
        break;

        case 0x232:
        {
            int s;
            int nbytes;
            struct sockaddr_can addr;
            struct can_frame frameout;
            struct ifreq ifr;
            const char *ifname = SendToDspace;

            if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
            {
                perror("Error while opening socket");
                return -1;
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

            frameout.can_id  = "0xb32" ;
            frameout.can_dlc = CAN_DLC ;
            frameout.data[0] = frame.data[0] ;
            frameout.data[1] = frame.data[1] ;
            frameout.data[2] = frame.data[2] ;
            frameout.data[3] = frame.data[3] ;
            frameout.data[4] = frame.data[4] ;
            frameout.data[5] = frame.data[5] ;
            frameout.data[6] = frame.data[6] ;
            frameout.data[7] = frame.data[7] ;
            
            cout << " Receivced 0x232, Sent 0xb32 " << endl;

            nbytes = write(s, &frameout, sizeof(struct can_frame));
            printf("Wrote %d bytes\n", nbytes);
            //Close the SocketCAN
            close(s);
            return nbytes;
        }
        break;

        case 0x120:
        {
            int s;
            int nbytes;
            struct sockaddr_can addr;
            struct can_frame frameout;
            struct ifreq ifr;
            const char *ifname = SendToDspace;

            if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
            {
                perror("Error while opening socket");
                return -1;
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

            frameout.can_id  = "0xa20" ;
            frameout.can_dlc = CAN_DLC ;
            frameout.data[0] = frame.data[0] ;
            frameout.data[1] = frame.data[1] ;
            frameout.data[2] = frame.data[2] ;
            frameout.data[3] = frame.data[3] ;
            frameout.data[4] = frame.data[4] ;
            frameout.data[5] = frame.data[5] ;
            frameout.data[6] = frame.data[6] ;
            frameout.data[7] = frame.data[7] ;
            
            cout << " Receivced 0x120, Sent 0xa20 " << endl;

            nbytes = write(s, &frameout, sizeof(struct can_frame));
            printf("Wrote %d bytes\n", nbytes);
            //Close the SocketCAN
            close(s);
            return nbytes;
        }
        break;

        case 0x121:
        {
            int s;
            int nbytes;
            struct sockaddr_can addr;
            struct can_frame frameout;
            struct ifreq ifr;
            const char *ifname = SendToDspace;

            if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
            {
                perror("Error while opening socket");
                return -1;
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

            frameout.can_id  = "0xa12" ;
            frameout.can_dlc = CAN_DLC ;
            frameout.data[0] = frame.data[0] ;
            frameout.data[1] = frame.data[1] ;
            frameout.data[2] = frame.data[2] ;
            frameout.data[3] = frame.data[3] ;
            frameout.data[4] = frame.data[4] ;
            frameout.data[5] = frame.data[5] ;
            frameout.data[6] = frame.data[6] ;
            frameout.data[7] = frame.data[7] ;
            
            cout << " Receivced 0x121, Sent 0xa21 " << endl;

            nbytes = write(s, &frameout, sizeof(struct can_frame));
            printf("Wrote %d bytes\n", nbytes);
            //Close the SocketCAN
            close(s);
            return nbytes;
        }
        break;

        case 0x220:
        {
            int s;
            int nbytes;
            struct sockaddr_can addr;
            struct can_frame frameout;
            struct ifreq ifr;
            const char *ifname = SendToDspace;

            if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
            {
                perror("Error while opening socket");
                return -1;
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

            frameout.can_id  = "0xb20" ;
            frameout.can_dlc = CAN_DLC ;
            frameout.data[0] = frame.data[0] ;
            frameout.data[1] = frame.data[1] ;
            frameout.data[2] = frame.data[2] ;
            frameout.data[3] = frame.data[3] ;
            frameout.data[4] = frame.data[4] ;
            frameout.data[5] = frame.data[5] ;
            frameout.data[6] = frame.data[6] ;
            frameout.data[7] = frame.data[7] ;
            
            cout << " Receivced 0x220, Sent 0xb20 " << endl;

            nbytes = write(s, &frameout, sizeof(struct can_frame));
            printf("Wrote %d bytes\n", nbytes);
            //Close the SocketCAN
            close(s);
            return nbytes;
        }
        break;

        case 0x221:
        {
            int s;
            int nbytes;
            struct sockaddr_can addr;
            struct can_frame frameout;
            struct ifreq ifr;
            const char *ifname = SendToDspace;

            if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
            {
                perror("Error while opening socket");
                return -1;
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

            frameout.can_id  = "0xb21" ;
            frameout.can_dlc = CAN_DLC ;
            frameout.data[0] = frame.data[0] ;
            frameout.data[1] = frame.data[1] ;
            frameout.data[2] = frame.data[2] ;
            frameout.data[3] = frame.data[3] ;
            frameout.data[4] = frame.data[4] ;
            frameout.data[5] = frame.data[5] ;
            frameout.data[6] = frame.data[6] ;
            frameout.data[7] = frame.data[7] ;
            
            cout << " Receivced 0x221, Sent 0xb21 " << endl;

            nbytes = write(s, &frameout, sizeof(struct can_frame));
            printf("Wrote %d bytes\n", nbytes);
            //Close the SocketCAN
            close(s);
            return nbytes;
        }
        break;

        case 0x110:
        {
            int s;
            int nbytes;
            struct sockaddr_can addr;
            struct can_frame frameout;
            struct ifreq ifr;
            const char *ifname = SendToDspace;

            if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
            {
                perror("Error while opening socket");
                return -1;
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

            frameout.can_id  = "0xa10" ;
            frameout.can_dlc = CAN_DLC ;
            frameout.data[0] = frame.data[0] ;
            frameout.data[1] = frame.data[1] ;
            frameout.data[2] = frame.data[2] ;
            frameout.data[3] = frame.data[3] ;
            frameout.data[4] = frame.data[4] ;
            frameout.data[5] = frame.data[5] ;
            frameout.data[6] = frame.data[6] ;
            frameout.data[7] = frame.data[7] ;
            
            cout << " Receivced 0x110, Sent 0xa10 " << endl;

            nbytes = write(s, &frameout, sizeof(struct can_frame));
            printf("Wrote %d bytes\n", nbytes);
            //Close the SocketCAN
            close(s);
            return nbytes;
        }
        break;

        case 0x211:
        {
            int s;
            int nbytes;
            struct sockaddr_can addr;
            struct can_frame frameout;
            struct ifreq ifr;
            const char *ifname = SendToDspace;

            if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
            {
                perror("Error while opening socket");
                return -1;
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

            frameout.can_id  = "0xb11" ;
            frameout.can_dlc = CAN_DLC ;
            frameout.data[0] = frame.data[0] ;
            frameout.data[1] = frame.data[1] ;
            frameout.data[2] = frame.data[2] ;
            frameout.data[3] = frame.data[3] ;
            frameout.data[4] = frame.data[4] ;
            frameout.data[5] = frame.data[5] ;
            frameout.data[6] = frame.data[6] ;
            frameout.data[7] = frame.data[7] ;
            
            cout << " Receivced 0x211, Sent 0xb11 " << endl;

            nbytes = write(s, &frameout, sizeof(struct can_frame));
            printf("Wrote %d bytes\n", nbytes);
            //Close the SocketCAN
            close(s);
            return nbytes;
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

        int rc;
        int s;
        struct can_filter filter[NumOfReceiveID];
        for(int i=0 ;i <NumOfReceiveID; i++)
        {
            filter[i].can_mask = CAN_SFF_MASK;
        }
        filter[0].can_id   = 0x130;
        filter[1].can_id   = 0x230;
        filter[2].can_id   = 0x231;
        filter[3].can_id   = 0x232;
        filter[4].can_id   = 0x120;
        filter[5].can_id   = 0x121;
        filter[6].can_id   = 0x220;
        filter[7].can_id   = 0x221;
        filter[8].can_id   = 0x110;
        filter[9].can_id   = 0x211;

        int cnt = 0;
        int nbytes;
        struct sockaddr_can addr;
        struct can_frame frame;
        struct ifreq ifr;
        const char *ifname = ReveiveFromXBW;

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

        ros::Rate rate(100);
        while(1)
        {
            //nbytes = read(s, &frame, sizeof(struct can_frame));
            //
            for (int i =0; i <NumOfReceiveID; i++)
            {
                nbytes = read(s, &frame, CANFD_MTU);
                ProcessFrame(frame);
                //std::cout << "i = " << i << std::endl;
            }
            rate.sleep();
        }
        return 0;
}
