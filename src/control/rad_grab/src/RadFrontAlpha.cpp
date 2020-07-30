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

int alpha_radar_parsing(struct can_frame frame);

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
  int line_count = 0;

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
  // // test frame x = -0.3 y = 1.4 p = 34
  // struct can_frame t_frame;
  // t_frame.can_id = 0xC1;
  // t_frame.can_dlc = 8;
  // t_frame.data[0] = 0x40;
  // t_frame.data[1] = 0x7b;
  // t_frame.data[2] = 0xfd;
  // t_frame.data[3] = 0x00;
  // t_frame.data[4] = 0x74;
  // t_frame.data[5] = 0x40;
  // t_frame.data[6] = 0x00;
  // t_frame.data[7] = 0x00;

  // alpha_radar_parsing(t_frame);

  struct can_frame s_frame;
  s_frame.can_id = 0xC2;
  s_frame.can_dlc = 8;
  s_frame.data[0] = 0x61;
  s_frame.data[1] = 0x72;
  s_frame.data[2] = 0x20;
  s_frame.data[3] = 0x31;
  s_frame.data[4] = 0x20;
  s_frame.data[5] = 0x32;
  s_frame.data[6] = 0x20;
  s_frame.data[7] = 0x32;
  int s_result = write(s, &s_frame, sizeof(s_frame));

  s_frame.can_id = 0xC3;

  s_result = write(s, &s_frame, sizeof(s_frame));

  if (s_result != sizeof(s_frame))
  {
    printf("Error\n!");
  }

  ros::init(argc, argv, "AlphaRadFrontPub");
  ros::NodeHandle n;
  ros::Publisher RadFrontPub = n.advertise<msgs::Rad>("RadFrontAlpha", 1);
  ros::Rate loop_rate(20);

  float x, y, z, speed;
  while (ros::ok())
  {
    count = 0;
    std::cout << ros::Time::now() << std::endl;

    while (1)
    {
      nbytes = read(s, &can_frame_tmp, sizeof(struct can_frame));
      printf("[%04X] %d %02X %02X %02X %02X %02X %02X %02X %02X \n", can_frame_tmp.can_id, can_frame_tmp.can_dlc,
             can_frame_tmp.data[0], can_frame_tmp.data[1], can_frame_tmp.data[2], can_frame_tmp.data[3],
             can_frame_tmp.data[4], can_frame_tmp.data[5], can_frame_tmp.data[6], can_frame_tmp.data[7]);
      int state = alpha_radar_parsing(can_frame_tmp);
      count++;

      if (state > 0)
      {
        line_count = count;
        while (line_count < 11)
        {
          line_count++;
          std::cout << std::endl;
        }
        break;
      }
    }

    printf("********************  count = %d  ******************\n", count);
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}

int alpha_radar_parsing(struct can_frame frame)
{
  int id;
  int state;
  int trackid;
  int cx;
  int cy;
  float x;
  float y;
  int p;
  int cvx;
  int cvy;
  float vx;
  float vy;

  id = (frame.data[0] & 0xfc) >> 2;

  state = frame.data[0] & 0x03;

  trackid = (frame.data[1] & 0xfc) >> 2;

  cx = (frame.data[1] & 0x02) >> 1;

  cy = (frame.data[1] & 0x80) >> 7;

  cvx = (frame.data[5] & 0x20) >> 5;

  cvy = (frame.data[6] & 0x08) >> 3;

  if (cx == 1)
  {
    cx = -1;
    x = (0x1ff - (((frame.data[1] & 0x03) << 8 | frame.data[2]) & 0x1ff) + 1) * 0.1 * cx;
  }
  else
  {
    x = (((frame.data[1] & 0x03) << 8 | frame.data[2]) & 0x1ff) * 0.1;
  }

  if (cy == 1)
  {
    cy = -1;
    y = ((0xfff - ((frame.data[3] << 5 | (frame.data[4] & 0xf8) >> 3) & 0xfff)) + 1) * 0.1 * cy;
  }
  else
  {
    y = ((frame.data[3] << 5 | (frame.data[4] & 0xf8) >> 3) & 0xfff) * 0.1;
  }

  if (cvx == 1)
  {
    cvx = -1;
    vx = (0x1ff - (((frame.data[5] & 0x3f) << 4 | (frame.data[6] & 0xf0) >> 4) & 0x1ff) + 1) * 0.1 * cvx;
  }
  else
  {
    vx = (((frame.data[5] & 0x3f) << 4 | (frame.data[6] & 0xf0) >> 4) & 0x1ff) * 0.1;
  }

  if (cvy == 1)
  {
    cvy = -1;
    vy = (0xeff - (((frame.data[6] & 0x0f) << 8 | frame.data[7]) & 0xeff) + 1) * 0.1 * cvy;
  }
  else
  {
    vy = (((frame.data[6] & 0x0f) << 8 | frame.data[7]) & 0xeff) * 0.1;
  }

  p = (((frame.data[4] & 0x07) << 2) | ((frame.data[5] & 0xc0) >> 6)) * 2;

  std::cout << "id : " << id << ", state : " << state << ", track : " << trackid << ", p : " << p << ", x : " << x
            << ", y : " << y << ", vx : " << vx << ", vy : " << vy << std::endl;

  return state;
}
