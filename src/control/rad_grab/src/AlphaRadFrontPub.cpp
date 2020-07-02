#include "ros/ros.h"
#include "std_msgs/Header.h"
#include "std_msgs/String.h"
#include "msgs/Rad.h"
#include "msgs/PointXYZV.h"
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

int main(int argc, char** argv)
{
  uint32_t seq = 0;

  int s;
  struct can_frame frame[64];
  struct can_frame can_frame_tmp;

  int nbytes, i;
  static struct ifreq ifr;
  static struct sockaddr_ll sll;
  char* ifname = "can1";
  int ifindex;
  int opt;
  int send_one_frame = 0;
  int count = 0;

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

  std::cout << "@@@@@@@@@@@@@" << std::endl;
  ros::init(argc, argv, "AlphaRadFrontPub");
  ros::NodeHandle n;
  ros::Publisher RadFrontPub = n.advertise<msgs::Rad>("RadFrontAlpha", 1);
  ros::Rate loop_rate(20);  // ros::Rate loop_rate(30);
  srand(ros::Time::now().toSec());

  float x, y, z, speed;
  while (ros::ok())
  {

  std::cout << "#################" << std::endl;
    count = 0;
    while (1)
    {

  std::cout << "$$$$$$$$$$$$$$$$$$$" << std::endl;
      nbytes = read(s, &can_frame_tmp, sizeof(struct can_frame));

  std::cout << "$$$$$$$$$$$$$$$$$$$" << std::endl;
      if (can_frame_tmp.can_id == 0xC1)
      {
        memcpy(&frame[i], &can_frame_tmp, sizeof(struct can_frame));

        printf("********************  count = %d  ******************\n", count);
        count++;
        break;
      }

  std::cout << "$$$$$$$$$$$$$$$$$$$" << std::endl;
    }
  }

  ros::spinOnce();
  loop_rate.sleep();

  return 0;
}
