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
#include <linux/can/raw.h>

using namespace std;

void onInit(ros::NodeHandle nh, ros::NodeHandle n);
void turnRadarOn(int s, int type);
int radarParsing(struct can_frame frame, msgs::PointXYZV* point);
void frontRadFilter(msgs::Rad* rad);
void cornerRadFilter(msgs::Rad* rad);
int debug_message = 0;
int alpha_raw_message = 0;
char *ifname; // YD_TEST

struct can_frame current_frame;
ros::Publisher RadPub;
int filter_distance = 6;
int filter_min = 4;

// y: 往前 , x: 右正
// 1: front center, 2: front left, 3: front right,
// 4: side left, 5: side right,
// 6: back left, 7: back right
//
//            2__1__3
//            4|   |5
//             |   |
//             |   |
//             |   |
//             |   |
//            6|___|7
//
//
// front radar must be exist for the 電阻

int main(int argc, char** argv)
{
  uint32_t seq = 0;

  int s;
  int rc;
  struct can_frame frame[64];
  struct can_frame can_frame_tmp;

  int nbytes, i;
  static struct ifreq ifr;
  static struct sockaddr_ll sll;
//  char* ifname = "can1";	// YD_TEST
  int ifindex;
  int opt;
  int send_one_frame = 0;
  int count = 0;

  ros::init(argc, argv, "RadAlpha");
  ros::NodeHandle n;
  ros::NodeHandle nh("~");
  ros::Rate loop_rate(20);

  onInit(nh, n);
  if (current_frame.can_id == 0x00)
  {
    perror("can filter error ");
    return 1;
  }

  // Add filter to socket
  struct can_filter rfilter[1];
  rfilter[0].can_id = current_frame.can_id;
  rfilter[0].can_mask = CAN_SFF_MASK;  //#define CAN_SFF_MASK 0x000007FFU

  s = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (s < 0)
  {
    perror("socke error ");
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
    perror("bind error ");
    return 1;
  }

  if (current_frame.can_id == 0xC1)
  {
    // the front radar has different enable code
    turnRadarOn(s, 1);
  }
  else
  {
    turnRadarOn(s, 2);
  }

  int err_count = 0;
  msgs::Rad rad;
  rad.radHeader.frame_id = "radar";

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

      if (can_frame_tmp.can_id == current_frame.can_id)
      {
        int state = radarParsing(can_frame_tmp, &point);
        count++;
        rad.radPoint.push_back(point);
        // 0:Have the next data, 1:Last data, 2:No object, 3:Reserved
        if (state > 0)
        {
          // if (current_frame.can_id == 0xC1)
          // {
          //   frontRadFilter(&rad);
          // }
          // else
          // {
          //   cornerRadFilter(&rad);
          // }
          RadPub.publish(rad);
          count = (int)rad.radPoint.size();
          rad.radPoint.clear();
          err_count = 0;
          break;
        }
      }
      else
      {
        cout << "================== error ====================" << endl;
        err_count++;
        break;
      }
    }
    if (err_count > 0)
    {
      ros::shutdown();
    }

    if (debug_message)
    {
      printf("========= alpha [%04X] count : %d\n", can_frame_tmp.can_id, count);
    }
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}

void onInit(ros::NodeHandle nh, ros::NodeHandle n)
{
  nh.param("/debug_message", debug_message, 0);
  nh.param("/alpha_raw_message", alpha_raw_message, 0);

  // YD_TEST
  string ifname_temp = "any";
  nh.getParam("ifname", ifname_temp);
  ifname = (char *)malloc(sizeof(char) * (ifname_temp.length()+1));
  strcpy(ifname, ifname_temp.c_str());
  cout << endl << endl << "++++++++++ ifname(alpha) = " << ifname << " ++++++++++" << endl;
  // YD_TEST

  int filter_id = 0;
  nh.getParam("filter_id", filter_id);

  cout << "============id============  " << filter_id << endl;

  switch (filter_id)
  {
    case 1:
      current_frame.can_id = 0xC1;
      RadPub = n.advertise<msgs::Rad>("AlphaFrontCenter", 1);
      break;
    case 2:
      current_frame.can_id = 0xC2;
      RadPub = n.advertise<msgs::Rad>("AlphaFrontLeft", 1);
      break;
    case 3:
      current_frame.can_id = 0xC3;
      RadPub = n.advertise<msgs::Rad>("AlphaFrontRight", 1);
      break;
    case 4:
      current_frame.can_id = 0xC4;
      RadPub = n.advertise<msgs::Rad>("AlphaSideLeft", 1);
      break;
    case 5:
      current_frame.can_id = 0xC5;
      RadPub = n.advertise<msgs::Rad>("AlphaSideRight", 1);
      break;
    case 6:
      current_frame.can_id = 0xC6;
      RadPub = n.advertise<msgs::Rad>("AlphaBackLeft", 1);
      break;
    case 7:
      current_frame.can_id = 0xC7;
      RadPub = n.advertise<msgs::Rad>("AlphaBackRight", 1);
      break;
    default:
      current_frame.can_id = 0x00;
      cout << "NO filter_id FOUND!" << endl;
      break;
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
  // radarParsing(t_frame);
}

void turnRadarOn(int s, int type)
{
  cout << "============ radar on ============  " << current_frame.can_id << endl;

  struct can_frame s_frame;
  s_frame.can_id = current_frame.can_id;
  s_frame.can_dlc = 8;

  switch (type)
  {
    // 1: front radar, 2: corner radar
    case 1:
      s_frame.data[0] = 0x61;
      s_frame.data[1] = 0x72;
      s_frame.data[2] = 0x20;
      s_frame.data[3] = 0x31;
      s_frame.data[4] = 0x20;
      s_frame.data[5] = 0x36;
      s_frame.data[6] = 0x20;
      s_frame.data[7] = 0x32;
      break;
    case 2:
      s_frame.data[0] = 0x61;
      s_frame.data[1] = 0x72;
      s_frame.data[2] = 0x20;
      s_frame.data[3] = 0x31;
      s_frame.data[4] = 0x20;
      s_frame.data[5] = 0x32;
      s_frame.data[6] = 0x20;
      s_frame.data[7] = 0x32;
      break;
    default:
      break;
  }

  int s_result = write(s, &s_frame, sizeof(s_frame));

  if (s_result != sizeof(s_frame))
  {
    printf("Error\n!");
  }
}

void frontRadFilter(msgs::Rad* rad)
{
  vector<msgs::PointXYZV> temp;
  float closed_object = 200;

  for (int i = 0; i < rad->radPoint.size(); i++)
  {
    if (rad->radPoint[i].y < closed_object)
    {
      if (rad->radPoint[i].y > filter_min)
      {
        closed_object = rad->radPoint[i].y;
      }
    }
  }

  for (int i = 0; i < rad->radPoint.size(); i++)
  {
    if (rad->radPoint[i].y < closed_object + filter_distance)
    {
      temp.push_back(rad->radPoint[i]);
    }
  }

  rad->radPoint.clear();

  for (int i = 0; i < temp.size(); i++)
  {
    rad->radPoint.push_back(temp[i]);
  }
}

void cornerRadFilter(msgs::Rad* rad)
{
  // clear 0.0 的radar point
  vector<msgs::PointXYZV> temp;
  float closed_object = 0.1;

  for (int i = 0; i < rad->radPoint.size(); i++)
  {
    if (abs(rad->radPoint[i].x) > closed_object)
    {
      temp.push_back(rad->radPoint[i]);
    }
  }

  rad->radPoint.clear();

  for (int i = 0; i < temp.size(); i++)
  {
    rad->radPoint.push_back(temp[i]);
  }
}

int radarParsing(struct can_frame frame, msgs::PointXYZV* point)
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

  if (alpha_raw_message)
  {
    std::cout << "id : " << id << ", state : " << state << ", track : " << trackid << ", p : " << p << ", x : " << x
              << ", y : " << y << ", vx : " << vx << ", vy : " << vy << std::endl;
  }

  // fill data to msg
  point->x = -x;
  point->y = y;
  point->z = -1;
  point->speed = vy;

  return state;
}
