#include "ros/ros.h"
#include "std_msgs/Header.h"
#include "std_msgs/String.h"
#include "msgs/Rad.h"
#include "msgs/PointXYZV.h"
#include <cstring>
#include <visualization_msgs/Marker.h>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <netinet/in.h> /* for htons() */

#include <linux/if_packet.h>
#include <linux/if_ether.h> /* for ETH_P_CAN */
#include <linux/can.h>      /* for struct can_frame */

//For CAN BUS
#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#define CAN_DLC 8;
#define CAN_INTERFACE_NAME "can1"





void callbackRadFront(const msgs::Rad::ConstPtr &msg)
{
  

    std_msgs::Header h = msg->radHeader;
    //printf("h.stamp.sec: %d\n", h.stamp.sec);
    //printf("h.stamp.nsec: %d\n", h.stamp.nsec);
    //printf("h.seq: %d\n", h.seq);
  int s;
  int nbytes;
  struct sockaddr_can addr;
  struct can_frame frame;
  struct ifreq ifr;

  const char *ifname = CAN_INTERFACE_NAME;

  if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
  {
    perror("Error while opening socket");
    return ;
  }


  strcpy(ifr.ifr_name, ifname);
  ioctl(s, SIOCGIFINDEX, &ifr);

  addr.can_family  = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;

  //printf("%s at index %d\n", ifname, ifr.ifr_ifindex);

  if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0)
  {
    perror("Error in socket bind");
    return;
  }
    
    frame.can_id  = 0x632 ;
    frame.can_dlc = CAN_DLC; 
   
    float min = 100;
    float x = 1000;
    float y = 1000;
    float speed = 0;
    for (int i = 0; i < msg->radPoint.size(); i++)
    {	
 	
	if(msg->radPoint[i].speed<70 && msg->radPoint[i].y<1.2 && msg->radPoint[i].y>-1.2 )
        {
	    if((msg->radPoint[i].x) < min)
            {
		min = (msg->radPoint[i].x);
	        x = (msg->radPoint[i].x);
	        y = msg->radPoint[i].y;
	        speed = msg->radPoint[i].speed;
	    }
	     
        }
	
    }

    
    if(min<100){
        ROS_INFO("radPoint(x, y, speed)=(%8.4f, %8.4f, %8.4f)", x, y, speed);   
	
	frame.data[0] = int(speed*100)&0xff ;
        frame.data[1] = int(speed*100)>>8 ;
        frame.data[2] = int(speed*100)>>16 ;
        frame.data[3] = int(speed*100)>>24 ;
	frame.data[4] = int(x*10)&0xff ;
        frame.data[5] = int(x*10)>>8 ;
        frame.data[6] = int(x*10)>>16 ;
        frame.data[7] = int(x*10)>>24 ;
   	nbytes = write(s, &frame, sizeof(struct can_frame));
    	printf("Wrote %d bytes", nbytes);
	ROS_INFO("(x,speed)=(%8.4f, %8.4f)", x, speed);
        
        
    }
        
    close(s);
}


int main(int argc, char **argv)
{

    ros::init(argc, argv, "RadFrontSub");  
    ros::NodeHandle n;
    ros::Subscriber RadFrontSub = n.subscribe("RadFront", 1, callbackRadFront);
    ros::Rate rate(100);
    while (ros::ok())
    {
        ros::spinOnce();
	rate.sleep();
    }
    return 0;
}
