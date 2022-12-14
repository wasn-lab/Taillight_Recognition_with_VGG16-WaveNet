#include "ros/ros.h"
#include "std_msgs/Header.h"
#include "std_msgs/String.h"
#include "msgs/Rad.h"
#include "msgs/PointXYZV.h"
#include <string>
#include <cstring>
#include <cmath>

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

void delphi_radar_parsing(struct can_frame frame, float* x, float* y, float* z, float* speed);
void onInit(ros::NodeHandle nh, ros::NodeHandle n);

int debug_message = 0;
int delphi_raw_message = 0;
char *ifname;

int main(int argc, char** argv)
{
  uint32_t seq = 0;

  int s;
  struct can_frame frame[64];
  struct can_frame can_frame_tmp;

  int nbytes, i;
  static struct ifreq ifr;
  static struct sockaddr_ll sll;
  int ifindex;
  int send_one_frame = 0;
  int count = 0;

  ros::init(argc, argv, "RadDelphi");
  ros::NodeHandle n;
  ros::NodeHandle nh("~");

  ros::Publisher RadFrontPub = n.advertise<msgs::Rad>("DelphiFront", 1);
  ros::Rate loop_rate(20);
  
  onInit(nh, n);

  s = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (s < 0)
  {
    perror("socket");
    return 1;
  }

  if (strcmp(ifname, "any") == 0)
  {
    ifindex = 0;
  }
  else
  {
    strcpy(ifr.ifr_name, ifname);
    ioctl(s, SIOCGIFINDEX, &ifr);
    ifindex = ifr.ifr_ifindex;
  }

  sll.sll_family = AF_PACKET;
  sll.sll_ifindex = ifindex;
  sll.sll_protocol = htons(ETH_P_CAN);

  if (bind(s, (struct sockaddr*)&sll, sizeof(sll)) < 0)
  {
    perror("bind");
    return 1;
  }
  else
  {
    std::cout << "Create success !!" << std::endl;
  }

  float x, y, z, speed;

  while (ros::ok())
  {
    msgs::Rad rad;
    rad.radHeader.stamp = ros::Time::now();
    rad.radHeader.seq = seq++;
    msgs::PointXYZV point;

    // double radar_angle;
    // double temp[3], radar_xyz[3], camera_xyz[3];
    count = 0;
    for (i = 0; i < 64; i++)
    {
      while (1)
      {
        nbytes = read(s, &can_frame_tmp, sizeof(struct can_frame));
        if (can_frame_tmp.can_id == (0x500 + i))
        {
          memcpy(&frame[i], &can_frame_tmp, sizeof(struct can_frame));
          count++;
          break;
        }
      }
    }
    if (debug_message)
    {
      printf("*******  Delphi count = %d  *******\n", count);
    }

    std_msgs::Header h = rad.radHeader;
    // printf("h.seq: %d, h.stamp: %d.%d\n", h.seq, h.stamp.sec, h.stamp.nsec);

    for (i = 0; i < count; i++)
    {
      x = 0, y = 0, z = 0, speed = 0;
      delphi_radar_parsing(frame[i], &x, &y, &z, &speed);
      point.x = x;
      x = x + 0.5;  // Align with lidar origin
      point.y = y;
      point.z = z;
      point.speed = speed;
      rad.radPoint.push_back(point);
      // ROS_INFO("radPoint(x, y, z, speed)=(%8.4f, %8.4f, %8.4f, %8.4f)", point.x, point.y, point.z, point.speed);
    }
    RadFrontPub.publish(rad);

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}

void onInit(ros::NodeHandle nh, ros::NodeHandle n)
{
  nh.param("/debug_message", debug_message, 0);
  nh.param("/delphi_raw_message", delphi_raw_message, 0);

  std::string ifname_temp = "can0";
  nh.getParam("ifname", ifname_temp);
  ifname = (char *)malloc(sizeof(char) * (ifname_temp.length()+1));
  strcpy(ifname, ifname_temp.c_str());
  std::cout << std::endl << std::endl << "++++++++++ ifname(delphi) = " << ifname << " ++++++++++" << std::endl;

}

void delphi_radar_parsing(struct can_frame frame, float* x, float* y, float* z, float* speed)
{
  unsigned int Range, Angle, tmp;
  float fRange, fAngle;
  int i, sign;
  if ((frame.can_id >= 0x500) && (frame.can_id <= 0x53f))
  {
    Range = ((frame.data[2] & 0x07) << 8) | frame.data[3];
    fRange = Range * 0.1;

    sign = frame.data[1] & 0x10;
    if (sign == 0)
    {
      sign = 1;
    }
    else
    {
      sign = -1;
    }
    Angle = ((frame.data[1] & 0x0f) << 5) | (frame.data[2] >> 3);
    if (sign == 1)
    {
      fAngle = Angle * 0.1;
    }
    else
    {
      fAngle = Angle * 0.1 - 51.2;
    }

    sign = frame.data[6] & 0x20;
    if (sign == 0)
    {
      sign = 1;
    }
    else
    {
      sign = -1;
    }
    tmp = ((frame.data[6] & 0x1f) << 8) | frame.data[7];
    if (sign == 1)
    {
      *speed = tmp * 0.01;
    }
    else
    {
      *speed = tmp * 0.01 - 81.92;
    }

    if (fRange != 0)
    {
      *x = fRange * cos(fAngle / 180 * M_PI) - 0.4;
      *y = fRange * sin(fAngle / 180 * M_PI);
      *z = 0.2;
      int mode = (frame.data[6] & 0xC0) >> 6;
      if (delphi_raw_message)
      {
        printf("[%04X] %02X %02X %02X %02X %02X %02X %02X %02X \n", frame.can_id, frame.data[0], frame.data[1],
               frame.data[2], frame.data[3], frame.data[4], frame.data[5], frame.data[6], frame.data[7]);
        printf("       x : %f, y : %f, speed : %f\n", *x, *y, *speed);
        printf("       fRange = %4.1f  ,fAngle = %4.1f \n", fRange, fAngle);
        printf("       Coming : %d, change : %d, width : %d, mode : %d\n", frame.data[0] & 0x01,
               (frame.data[0] & 0x02) >> 1, ((frame.data[4] & 0x3C) >> 2), ((frame.data[6] & 0xC0) >> 6));
      }
      if (mode == 0)
      {
        *speed = 85;
      }
    }
  }
}
