#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

// For ROS
#include "ros/ros.h"
#include "std_msgs/Header.h"
#include "std_msgs/String.h"
#include "msgs/Rad.h"
#include "msgs/BoxPoint.h"
#include "msgs/DynamicPath.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/PathPrediction.h"
#include "msgs/PointXY.h"
#include "msgs/PointXYZ.h"
#include "msgs/PointXYZV.h"
#include "msgs/TrackInfo.h"
#include "sensor_msgs/Imu.h"
#include "msgs/LocalizationToVeh.h"
#include <cstring>
#include <visualization_msgs/Marker.h>

// For PCL
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace std;


ros::Publisher RadFrontPub;

double imu_angular_velocity_z = 0;
int do_rotate = 0;

void callbackRadFront(const msgs::Rad::ConstPtr& msg)
{
  int m_rotate = do_rotate;
  msgs::Rad rad;
  msgs::PointXYZV point;

  //             _____
  //             |   |
  //             |   |
  //             |   |
  //             |   |
  //             |   |
  //             |___|
  //
  //

  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    int m_NeedAdd = 0;
    double x = abs(msg->radPoint[i].x);
    double y = abs(msg->radPoint[i].y);
    double speed = abs(msg->radPoint[i].speed);

    if (speed > 80 || x > 80)
    {
      continue;
    }

    if (m_rotate == 1)
    {
      if (y < 1.6 && speed > 4)
      {
        m_NeedAdd = 1;
      }
      else if (y < 4 && y >= 1.6 && speed > 2.2)
      {
        m_NeedAdd = 1;
      }
      else if (y < 7 && y >= 4 && speed > 3)
      {
        m_NeedAdd = 1;
      }
      else if (y < 15 && y >= 7 && speed > 1)
      {
        m_NeedAdd = 1;
      }
    }
    else
    {
      // 直線先不做判斷，先讓AEB work
      m_NeedAdd = 1;
      // if (y < 1.2 && speed > 3)
      // {
      //   m_NeedAdd = 1;
      // }
      // else if (y < 15 && y >= 1.2 && speed > 0)
      // {
      //   m_NeedAdd = 1;
      // }
    }

    if (m_NeedAdd == 1)
    {
      point.x = msg->radPoint[i].x;
      point.y = msg->radPoint[i].y;
      point.z = 0;
      point.speed = msg->radPoint[i].speed;

      // Debug
      cout << "X: " << point.x << ", Y: " << point.y << ", Speed: " << point.speed << endl;

      rad.radPoint.push_back(point);
    }
  }
  std::cout << "Radar Data : " << rad.radPoint.size() << std::endl;
  rad.radHeader.stamp = msg->radHeader.stamp;
  rad.radHeader.seq = msg->radHeader.seq;
  RadFrontPub.publish(rad);
}

void callbackIMU(const sensor_msgs::Imu::ConstPtr& input)
{
  imu_angular_velocity_z = input->angular_velocity.z;
  if (abs(imu_angular_velocity_z) > 0.05)
  {
    do_rotate = 1;
  }
  else
  {
    do_rotate = 0;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "Rad_Filter");
  ros::NodeHandle n;
  ros::Subscriber RadFrontSub = n.subscribe("RadFrontDelphi", 1, callbackRadFront);
  ros::Subscriber IMURadSub = n.subscribe("imu_data_rad", 1, callbackIMU);

  RadFrontPub = n.advertise<msgs::Rad>("RadFront", 1);

  ros::Rate rate(20);
  while (ros::ok())
  {
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}
