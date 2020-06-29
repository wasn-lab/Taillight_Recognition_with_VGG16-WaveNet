#include "ros/ros.h"
#include "std_msgs/Header.h"
#include "std_msgs/String.h"
#include "msgs/Rad.h"
#include "msgs/PointXYZV.h"
#include <cstring>

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

void callbackRadFront(const msgs::Rad::ConstPtr& msg)
{
  std_msgs::Header h = msg->radHeader;
  printf("h.stamp.sec: %d\n", h.stamp.sec);
  printf("h.stamp.nsec: %d\n", h.stamp.nsec);
  printf("h.seq: %d\n", h.seq);

  /*
      for (int i = 0; i < msg->radPoint.size(); i++)
      {
    if(msg->radPoint[i].speed<70)
    {
            ROS_INFO("radPoint(x, y, z, speed)=(%8.4f, %8.4f, %8.4f, %8.4f)", (float)msg->radPoint[i].x,
    (float)msg->radPoint[i].y, (float)msg->radPoint[i].z, (float)msg->radPoint[i].speed);
    }
          //ROS_INFO("radPoint(x, y, z, speed)=(%8.4f, %8.4f, %8.4f, %8.4f)", (float)msg->radPoint[i].x,
    (float)msg->radPoint[i].y, (float)msg->radPoint[i].z, (float)msg->radPoint[i].speed);
      }
  */

  float min = 100;
  float x = 0;
  float y = 0;
  float speed = 0;
  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    if (msg->radPoint[i].speed < 70 && msg->radPoint[i].y < 2 && msg->radPoint[i].y > -2)
    {
      if (msg->radPoint[i].x < min)
      {
        min = msg->radPoint[i].x;
        x = msg->radPoint[i].x;
        y = msg->radPoint[i].y;
        speed = msg->radPoint[i].speed;
      }
    }
    // ROS_INFO("radPoint(x, y, z, speed)=(%8.4f, %8.4f, %8.4f, %8.4f)", (float)msg->radPoint[i].x,
    // (float)msg->radPoint[i].y, (float)msg->radPoint[i].z, (float)msg->radPoint[i].speed);
  }
  ROS_INFO("radPoint(x, y, speed)=(%8.4f, %8.4f, %8.4f)", x, y, speed);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "RadFrontSub");
  ros::NodeHandle n;
  ros::Subscriber RadFrontSub = n.subscribe("RadFront", 1, callbackRadFront);

  while (ros::ok())
  {
    ros::spinOnce();
  }
  return 0;
}
