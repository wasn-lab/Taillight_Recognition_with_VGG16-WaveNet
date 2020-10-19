#include "ros/ros.h"
#include "std_msgs/Header.h"
#include "std_msgs/String.h"
#include "msgs/Rad.h"
#include "msgs/PointXYZV.h"
#include "msgs/RadObjectArray.h"
#include "msgs/RadObject.h"
#include <cstring>
#include <cmath>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <ctime>
#include <chrono>

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
int radar_object_num = 16;
int radar_object_data = radar_object_num * 2;
struct can_frame current_frame;
ros::Publisher RadPub;

int main(int argc, char** argv)
{
  uint32_t seq = 0;

  int s;
  int rc;
  // target 1 ~ target 15 each has 2 frame
  struct can_frame frame[radar_object_data];
  struct can_frame can_frame_tmp;

  int nbytes, i;
  static struct ifreq ifr;
  static struct sockaddr_ll sll;
  char* ifname = "can1";
  int ifindex;
  int send_one_frame = 0;
  vector<int> can_data;

  ros::init(argc, argv, "RadCubtek");
  ros::NodeHandle n;
  ros::NodeHandle nh("~");

  RadPub = n.advertise<msgs::RadObjectArray>("CubtekFront", 1);

  onInit(nh, n);

  // Add filter object to socket
  struct can_filter rfilter[radar_object_data + 1];
  for (int i = 0; i < radar_object_data; i++)
  {
    rfilter[i].can_mask = CAN_SFF_MASK;
    rfilter[i].can_id = (0x503 + i);
  }

  // Add filter cubtek radar header to socket
  rfilter[radar_object_data].can_mask = CAN_SFF_MASK;
  rfilter[radar_object_data].can_id = (0x500);

  s = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (s < 0)
  {
    perror("socket");
    return 1;
  }

  rc = setsockopt(s, SOL_CAN_RAW, CAN_RAW_FILTER, &rfilter, sizeof(rfilter));  //設定規則
  if (-1 == rc)
  {
    perror("setsockopt filter error ");
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

  msgs::RadObjectArray radArray;
  radArray.header.frame_id = "base_link";

  ros::Rate loop_rate(18);
  int no_obj = 0;
  int print_count = 100;

  while (ros::ok())
  {
    radArray.objects.clear();
    radArray.header.stamp = ros::Time::now();
    radArray.header.seq = seq++;

    nbytes = read(s, &can_frame_tmp, sizeof(struct can_frame));

    if (can_frame_tmp.can_id == 0x500)
    {
      no_obj = can_frame_tmp.data[0] >> 2;
    }

    if (no_obj > 0)
    {
      for (int i = 0; i < no_obj * 2; i++)
      {
        nbytes = read(s, &can_frame_tmp, sizeof(struct can_frame));
        int loc = can_frame_tmp.can_id - 0x503;
        can_data.push_back(loc);
        memcpy(&frame[loc], &can_frame_tmp, sizeof(struct can_frame));
      }

      int max = *max_element(can_data.begin(), can_data.end());
      can_data.clear();

      for (i = 0; i < max + 1; i += 2)
      {
        msgs::RadObject object;
        radarParsing(frame[i], frame[i + 1], &object);
        radArray.objects.push_back(object);
      }
    }

    RadPub.publish(radArray);

    print_count++;
    if (print_count > 60)
    {
      std::cout << "========= cubtek no_obj : " << no_obj << std::endl;
      print_count = 0;
    }

    ros::spinOnce();

    // auto start = std::chrono::system_clock::now();
    // std::time_t end_time = std::chrono::system_clock::to_time_t(start);
    // cout << ctime(&end_time) << endl;
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
  float px;
  float py;
  float vx;
  float vy;
  int track_id;

  // 0 : unclassified, 1 : standing, 2 : stopped, 3 : moving, 4 : oncoming, 5 : flyover
  int state;

  // 0 : unknow, 1 : pedestrian, 2 : bike, 3 : car, 4 : truck
  int type;

  // 0 : 25% 1 : 50%, 2 : 75%, 3 : 99%
  int prob_of_exist;

  float accel_x;
  int path_flag;
  int acc_flag;
  int aeb_flag;

  // start to parse data

  px = ((first.data[0] << 4) | ((first.data[1] & 0xf0) >> 4)) * 0.125;
  py = (((first.data[1] & 0x0f) << 8) | (first.data[2])) * 0.125 - 128;
  vx = ((first.data[3] << 4) | ((first.data[4] & 0xf0) >> 4)) * 0.05 - 102;
  vy = (((first.data[4] & 0x0f) << 8) | (first.data[5])) * 0.05 - 102;
  track_id = first.data[6];

  accel_x = ((second.data[0] << 4) | ((second.data[1] & 0xf0) >> 4)) * 0.04 - 40;

  path_flag = (first.data[7] & 0x08) >> 3;
  acc_flag = (first.data[7] & 0x80) >> 7;
  aeb_flag = (first.data[7] & 0x40) >> 6;

  // for debug use
  if (debug_message)
  {
    printf("1. [%04X] %02X %02X %02X %02X %02X %02X %02X %02X \n", first.can_id, first.data[0], first.data[1],
           first.data[2], first.data[3], first.data[4], first.data[5], first.data[6], first.data[7]);
    printf("2. [%04X] %02X %02X %02X %02X %02X %02X %02X %02X \n", second.can_id, second.data[0], second.data[1],
           second.data[2], second.data[3], second.data[4], second.data[5], second.data[6], second.data[7]);
    std::cout << "px : " << px << ", py : " << py << ", vx : " << vx << ", vy : " << vy << ", track_id : " << track_id
              << ", accel_x : " << accel_x << ", path_flag : " << path_flag << ", acc_flag : " << acc_flag << std::endl;
  }

  // fill data to msg
  rad_obj->px = px;
  rad_obj->py = py;
  rad_obj->vx = vx;
  rad_obj->vy = vy;
  rad_obj->track_id = track_id;
  rad_obj->path_flag = path_flag;
  rad_obj->aeb_flag = aeb_flag;
  rad_obj->acc_flag = acc_flag;
}
