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
using namespace std;

void onInit(ros::NodeHandle nh, ros::NodeHandle n);
int alpha_radar_parsing(struct can_frame frame, msgs::PointXYZV* point);

vector<double> Alpha_Front_Center_Param;
vector<double> Alpha_Front_Left_Param;
vector<double> Alpha_Front_Right_Param;
vector<double> Alpha_Side_Left_Param;
vector<double> Alpha_Side_Right_Param;
vector<double> Alpha_Back_Left_Param;
vector<double> Alpha_Back_Right_Param;
vector<double> Zero_Param(6, 0.0);

struct can_frame current_frame;
ros::Publisher RadPub;

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

  // TODO : Add filter to socket
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

  // ============ turn alpha radar on ===============
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

  ros::init(argc, argv, "RadAlpha");
  ros::NodeHandle n;
  ros::NodeHandle nh("~");
  ros::Rate loop_rate(20);

  onInit(nh, n);

  int err_count = 0;
  msgs::Rad rad;

  while (ros::ok())
  {
    rad.radHeader.stamp = ros::Time::now();
    rad.radHeader.seq = seq++;
    count = 0;
    err_count = 0;
    msgs::PointXYZV point;
    while (1)
    {
      nbytes = read(s, &can_frame_tmp, sizeof(struct can_frame));
      // printf("[%04X] %d %02X %02X %02X %02X %02X %02X %02X %02X \n", can_frame_tmp.can_id, can_frame_tmp.can_dlc,
      //        can_frame_tmp.data[0], can_frame_tmp.data[1], can_frame_tmp.data[2], can_frame_tmp.data[3],
      //        can_frame_tmp.data[4], can_frame_tmp.data[5], can_frame_tmp.data[6], can_frame_tmp.data[7]);

      printf("[%04X] [%04X] \n", can_frame_tmp.can_id, current_frame.can_id);

      if (can_frame_tmp.can_id == current_frame.can_id)
      {
        int state = alpha_radar_parsing(can_frame_tmp, &point);
        count++;
        rad.radPoint.push_back(point);
        // 0:Have the next data, 1:Last data, 2:No object, 3:Reserved
        if (state > 0)
        {
          RadPub.publish(rad);
          rad.radPoint.clear();
          err_count = 0;
          break;
        }
      }
      else
      {
        cout << "================== error ====================\n" << endl;
        err_count++;
        break;
      }
    }
    if (err_count > 0)
    {
      ros::shutdown();
    }
    printf("********************  count = %d  ******************\n", count);
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}

void onInit(ros::NodeHandle nh, ros::NodeHandle n)
{
  if (!ros::param::has("/Alpha_Front_Center_Param"))
  {
    nh.setParam("Alpha_Front_Center_Param", Zero_Param);
    nh.setParam("Alpha_Front_Left_Param", Zero_Param);
    nh.setParam("Alpha_Front_Right_Param", Zero_Param);
    nh.setParam("Alpha_Side_Left_Param", Zero_Param);
    nh.setParam("Alpha_Side_Right_Param", Zero_Param);
    nh.setParam("Alpha_Back_Left_Param", Zero_Param);
    nh.setParam("Alpha_Back_Right_Param", Zero_Param);
    cout << "NO STITCHING PARAMETER INPUT!" << endl;
    cout << "Now is using [0,0,0,0,0,0] as stitching parameter!" << endl;
  }
  else
  {
    nh.param("/Alpha_Front_Center_Param", Alpha_Front_Center_Param, vector<double>());
    nh.param("/Alpha_Front_Left_Param", Alpha_Front_Center_Param, vector<double>());
    nh.param("/Alpha_Front_Right_Param", Alpha_Front_Center_Param, vector<double>());
    nh.param("/Alpha_Side_Left_Param", Alpha_Front_Center_Param, vector<double>());
    nh.param("/Alpha_Side_Right_Param", Alpha_Front_Center_Param, vector<double>());
    nh.param("/Alpha_Back_Left_Param", Alpha_Front_Center_Param, vector<double>());
    nh.param("/Alpha_Back_Right_Param", Alpha_Front_Center_Param, vector<double>());
    cout << "STITCHING PARAMETER FIND!" << endl;
  }

  int filter_id = 0;
  nh.getParam("filter_id", filter_id);

  switch (filter_id)
  {
    case 1:
      current_frame.can_id = 0xC1;
      RadPub = n.advertise<msgs::Rad>("RadFrontAlpha", 1);
      break;
    case 2:
      current_frame.can_id = 0xC2;
      RadPub = n.advertise<msgs::Rad>("RadFrontLeft", 1);
      break;
    case 3:
      current_frame.can_id = 0xC3;
      RadPub = n.advertise<msgs::Rad>("RadFrontRight", 1);
      break;
    case 4:
      current_frame.can_id = 0xC4;
      RadPub = n.advertise<msgs::Rad>("RadSideLeft", 1);
      break;
    case 5:
      current_frame.can_id = 0xC5;
      RadPub = n.advertise<msgs::Rad>("RadSideRight", 1);
      break;
    case 6:
      current_frame.can_id = 0xC6;
      RadPub = n.advertise<msgs::Rad>("RadBackLeft", 1);
      break;
    case 7:
      current_frame.can_id = 0xC7;
      RadPub = n.advertise<msgs::Rad>("RadBackRight", 1);
      break;
    default:
      cout << "NO FILTER FOUND!" << endl;
      break;
  }
}

int alpha_radar_parsing(struct can_frame frame, msgs::PointXYZV* point)
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

  // std::cout << "id : " << id << ", state : " << state << ", track : " << trackid << ", p : " << p << ", x : " << x
  //           << ", y : " << y << ", vx : " << vx << ", vy : " << vy << std::endl;

  // fill data to msg
  point->x = x;
  point->y = y;
  point->z = -1;
  point->speed = vy;

  return state;
}
