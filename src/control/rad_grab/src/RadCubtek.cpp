#include "ros/ros.h"
#include "std_msgs/Header.h"
#include "std_msgs/String.h"
#include "msgs/Rad.h"
#include "msgs/PointXYZV.h"
#include "msgs/RadObjectArray.h"
#include "msgs/RadObject.h"
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
#include <linux/can/raw.h>

using namespace std;

void onInit(ros::NodeHandle nh, ros::NodeHandle n);
void turnRadarOn(int s, int type);
void radarParsing(struct can_frame first, struct can_frame second, msgs::RadObject* rad_obj);

int debug_message = 0;
int radar_object_num = 82;
struct can_frame current_frame;
ros::Publisher RadPub;

int main(int argc, char** argv)
{
  uint32_t seq = 0;

  int s;
  int rc;
  // target 1 ~ target 40 each has 2 frame
  struct can_frame frame[radar_object_num];
  struct can_frame can_frame_tmp;

  int nbytes, i;
  static struct ifreq ifr;
  static struct sockaddr_ll sll;
  char* ifname = "can1";
  int ifindex;
  int send_one_frame = 0;
  int count = 0;

  ros::init(argc, argv, "RadCubtek");
  ros::NodeHandle n;
  ros::NodeHandle nh("~");
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

  msgs::RadObjectArray arrays;

  while (ros::ok())
  {
    std::cout << "111111111111111111111!!" << std::endl;
    arrays.objects.clear();
    arrays.header.stamp = ros::Time::now();
    arrays.header.seq = seq++;

    count = 0;

    for (i = 0; i < radar_object_num; i++)
    {
      // while (1)
      // {
      nbytes = read(s, &can_frame_tmp, sizeof(struct can_frame));
      printf("[%04X] %02X %02X %02X %02X %02X %02X %02X %02X \n", can_frame_tmp.can_id, can_frame_tmp.data[0],
             can_frame_tmp.data[1], can_frame_tmp.data[2], can_frame_tmp.data[3], can_frame_tmp.data[4], can_frame_tmp.data[5],
             can_frame_tmp.data[6], can_frame_tmp.data[7]);

      // printf("[%04X]",can_frame_tmp.can_id);
      //     nbytes = read(s, &can_frame_tmp, sizeof(struct can_frame));
      //     if (can_frame_tmp.can_id == (0x503 + i))
      //     {

      // std::cout << "3333333333333333!!" << std::endl;
      //       memcpy(&frame[i], &can_frame_tmp, sizeof(struct can_frame));
      //       count++;
      //       break;
      //     }
      // }
    }

    std::cout << "4444444444444444444444!!" << std::endl;

    for (i = 0; i < count; i + 2)
    {
      msgs::RadObject object;
      radarParsing(frame[i], frame[i + 1], &object);
    }
    // RadPub.publish(rad);

    if (debug_message)
    {
      printf("[%04X] **********  count = %d  **********\n", can_frame_tmp.can_id, count);
    }

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}

void onInit(ros::NodeHandle nh, ros::NodeHandle n)
{
  nh.param("/debug_message", debug_message, 0);
}

void radarParsing(struct can_frame first, struct can_frame second, msgs::RadObject* rad_obj)
{
  int px;
  int py;
  int vx;
  int vy;
  int track_id;

  // 0 : unclassified, 1 : standing, 2 : stopped, 3 : moving, 4 : oncoming, 5 : flyover
  int state;

  // 0 : unknow, 1 : pedestrian, 2 : bike, 3 : car, 4 : truck
  int type;

  // 0 : 25% 1 : 50%, 2 : 75%, 3 : 99%
  int prob_of_exist;

  int accel_x;
  int path_flag;
  int acc_flag;
  int aeb_flag;

  // for debug use

  printf("[%04X] %02X %02X %02X %02X %02X %02X %02X %02X \n", first.can_id, first.data[0], first.data[1], first.data[2],
         first.data[3], first.data[4], first.data[5], first.data[6], first.data[7]);
  printf("[%04X] %02X %02X %02X %02X %02X %02X %02X %02X \n", second.can_id, second.data[0], second.data[1],
         second.data[2], second.data[3], second.data[4], second.data[5], second.data[6], second.data[7]);

  // start to parse data

  px = (first.data[0] << 4) | ((first.data[1] & 0xf0) >> 4);
  py = ((first.data[1] & 0x0f) << 8) | (first.data[2]);
  vx = (first.data[3] << 4) | ((first.data[4] & 0xf0) >> 4);
  vy = ((first.data[4] & 0x0f) << 8) | (first.data[5]);
  track_id = first.data[6];

  accel_x = (second.data[0] << 4) | ((second.data[1] & 0xf0) >> 4);

  path_flag = (first.data[7] & 0x08) >> 3;
  acc_flag = (first.data[7] & 0x80) >> 7;
  aeb_flag = (first.data[7] & 0x40) >> 6;

  // std::cout << "id : " << id << ", state : " << state << ", track : " << trackid << ", p : " << p << ", x : " << x
  //           << ", y : " << y << ", vx : " << vx << ", vy : " << vy << std::endl;

  // fill data to msg
  rad_obj->path_flag = path_flag;
  rad_obj->aeb_flag = aeb_flag;
  rad_obj->acc_flag = acc_flag;
}
