#include "std_msgs/Header.h"
#include "msgs/BoxPoint.h"
#include "msgs/DynamicPath.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/PathPrediction.h"
#include "msgs/PointXY.h"
#include "msgs/PointXYZ.h"
#include "msgs/PointXYZV.h"
#include "msgs/TrackInfo.h"
#include "msgs/LocalizationToVeh.h"
#include "msgs/MMTPInfo.h"
#include "ros/ros.h"
#include <visualization_msgs/Marker.h>
#include "msgs/Rad.h"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <iostream>
using namespace std;

ros::Publisher BBox_pub;
float m_radio = 0.25;
float m_height = m_radio * 2;
float m_depth = m_radio * 2;

void callbackRadFront(const msgs::Rad::ConstPtr& msg)
{
  std_msgs::Header h = msg->radHeader;
  ros::NodeHandle n;
  msgs::DetectedObjectArray BBox;
  msgs::DetectedObject Box_temp;
  BBox.header = msg->radHeader;
  Box_temp.header = msg->radHeader;
  Box_temp.header.frame_id = "lidar";

  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    if (msg->radPoint[i].speed < 80)
    {
      Box_temp.relSpeed = msg->radPoint[i].speed;

      Box_temp.bPoint.p0.x = msg->radPoint[i].x - m_radio;
      Box_temp.bPoint.p0.y = -msg->radPoint[i].y - m_radio;
      Box_temp.bPoint.p0.z = msg->radPoint[i].z;

      Box_temp.bPoint.p1.x = msg->radPoint[i].x - m_radio;
      Box_temp.bPoint.p1.y = -msg->radPoint[i].y;
      Box_temp.bPoint.p1.z = msg->radPoint[i].z + m_height;

      Box_temp.bPoint.p2.x = msg->radPoint[i].x + m_radio;
      Box_temp.bPoint.p2.y = -msg->radPoint[i].y;
      Box_temp.bPoint.p2.z = msg->radPoint[i].z + m_height;

      Box_temp.bPoint.p3.x = msg->radPoint[i].x + m_radio;
      Box_temp.bPoint.p3.y = -msg->radPoint[i].y;
      Box_temp.bPoint.p3.z = msg->radPoint[i].z;

      Box_temp.bPoint.p4.x = msg->radPoint[i].x - m_radio;
      Box_temp.bPoint.p4.y = -msg->radPoint[i].y + m_depth;
      Box_temp.bPoint.p4.z = msg->radPoint[i].z;

      Box_temp.bPoint.p5.x = msg->radPoint[i].x - m_radio;
      Box_temp.bPoint.p5.y = -msg->radPoint[i].y + m_depth;
      Box_temp.bPoint.p5.z = msg->radPoint[i].z + m_height;

      Box_temp.bPoint.p6.x = msg->radPoint[i].x + m_radio;
      Box_temp.bPoint.p6.y = -msg->radPoint[i].y + m_depth;
      Box_temp.bPoint.p6.z = msg->radPoint[i].z + m_height;

      Box_temp.bPoint.p7.x = msg->radPoint[i].x + m_radio;
      Box_temp.bPoint.p7.y = -msg->radPoint[i].y + m_depth;
      Box_temp.bPoint.p7.z = msg->radPoint[i].z;

      BBox.objects.push_back(Box_temp);

      // cout << "x = " << Box_temp.bPoint.p0.x << ", y = " << Box_temp.bPoint.p0.y << ", speed = " << Box_temp.relSpeed
      //  << endl;
    }
  }
  cout << "Number of objects = " << BBox.objects.size() << endl;
  BBox_pub.publish(BBox);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "RadFrontSub_BBox");
  ros::NodeHandle n;
  ros::Subscriber RadFrontSub = n.subscribe("RadFront", 1, callbackRadFront);
  BBox_pub = n.advertise<msgs::DetectedObjectArray>("RadarDetection", 1);
  ros::Rate rate(100);
  while (ros::ok())
  {
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}
