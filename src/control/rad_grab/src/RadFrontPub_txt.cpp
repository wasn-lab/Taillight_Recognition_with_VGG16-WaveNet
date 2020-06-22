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

void delphi_radar_parsing(uint8_t data[8], float* x, float* y, float* z, float* speed);

int main(int argc, char** argv)
{
  uint32_t seq = 0;

  ros::init(argc, argv, "RadFrontPub");
  ros::NodeHandle n;
  ros::Publisher RadFrontPub = n.advertise<msgs::Rad>("RadFront", 1);
  ros::Rate loop_rate(20);  // ros::Rate loop_rate(30);
  srand(ros::Time::now().toSec());

  float x, y, z, speed;
  FILE* txt = fopen("20190507_radar.txt", "r");

  char *pch, *buffer, tmp[20], can_id[4], ch;
  size_t size = 0;
  ssize_t read;
  int cur_timesec[2];
  if (read = getline(&buffer, &size, txt) != -1)
  {
    strncpy(&ch, buffer, 1);
    // printf("%c\n", ch);
    if (strcmp("s", &ch) == 0)
    {
      pch = strtok(buffer, ", ");
      pch = strtok(NULL, " ");
      std::strcpy(tmp, pch);
      cur_timesec[0] = atoi(tmp);
      pch = strtok(NULL, ", ");
      std::strcpy(tmp, pch + 14);
      cur_timesec[1] = atoi(tmp);
      // printf("timesec: %d.%d\n", cur_timesec[0], cur_timesec[1]);
    }
  }
  uint8_t data[8];
  int count;
  while (ros::ok())
  {
    msgs::Rad rad;
    rad.radHeader.stamp.sec = cur_timesec[0];
    rad.radHeader.stamp.nsec = cur_timesec[1];
    rad.radHeader.seq = seq++;
    msgs::PointXYZV point;
    std_msgs::Header h = rad.radHeader;
    // printf("h.seq: %d, h.stamp: %d.%d\n", h.seq, h.stamp.sec, h.stamp.nsec);

    count = 0;
    while (read = getline(&buffer, &size, txt) != -1)
    {
      strncpy(&ch, buffer, 1);
      // printf("%c\n", ch);
      if (strcmp("s", &ch) == 0)
      {
        pch = strtok(buffer, ", ");
        pch = strtok(NULL, " ");
        std::strcpy(tmp, pch);
        cur_timesec[0] = atoi(tmp);
        pch = strtok(NULL, ", ");
        std::strcpy(tmp, pch + 14);
        cur_timesec[1] = atoi(tmp);
        // printf("timesec: %d.%d\n", cur_timesec[0], cur_timesec[1]);
        break;
      }
      if (strcmp("[", &ch) == 0)
      {
        printf("%s\n", buffer);
        pch = strtok(buffer, "] ");
        std::strcpy(can_id, pch + 1);
        // printf("[%s] ", can_id);
        for (int i = 0; i < 8; i++)
        {
          pch = strtok(NULL, " ");
          sscanf(pch, "%2hhx", &data[i]);
          // printf("%02X ", data[i]);
        }
        // printf("\n");

        x = 0, y = 0, z = 0, speed = 0;
        delphi_radar_parsing(data, &x, &y, &z, &speed);
        point.x = x;
        point.y = y;
        point.z = z;
        point.speed = speed;
        rad.radPoint.push_back(point);
        ROS_INFO("radPoint(x, y, z, speed)=(%8.4f, %8.4f, %8.4f, %8.4f)", point.x, point.y, point.z, point.speed);
        count++;
      }
    }

    printf("********************  count = %d  ******************\n", count);
    RadFrontPub.publish(rad);

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}

void delphi_radar_parsing(uint8_t data[8], float* x, float* y, float* z, float* speed)
{
  unsigned int Range, Angle, tmp;
  float fRange, fAngle;
  int i, sign;

  Range = ((data[2] & 0x07) << 8) | data[3];
  fRange = Range * 0.1;

  sign = data[1] & 0x10;
  if (sign == 0)
  {
    sign = 1;
  }
  else
  {
    sign = -1;
  }
  Angle = ((data[1] & 0x0f) << 5) | (data[2] >> 3);
  if (sign == 1)
  {
    fAngle = Angle * 0.1;
  }
  else
  {
    fAngle = Angle * 0.1 - 51.2;
  }

  sign = data[1] & 0x20;
  if (sign == 0)
  {
    sign = 1;
  }
  else
  {
    sign = -1;
  }
  tmp = ((data[6] & 0x1f) << 8) | data[7];
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
    // printf("\n[%04X] %02X %02X %02X %02X %02X %02X %02X %02X \n", can_id, data[0], data[1], data[2], data[3],
    // data[4], data[5], data[6], data[7]); printf("fRange = %4.1f  ,fAngle = %4.1f \n", fRange, fAngle);
    *x = -fRange * cos(fAngle / 180 * M_PI);
    *y = fRange * sin(fAngle / 180 * M_PI);
    *z = 0.5;
  }
}