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
#include "msgs/LocalizationToVeh.h"
#include <cstring>
#include <visualization_msgs/Marker.h>

// For PCL
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

ros::Publisher PCloud_pub;
ros::Publisher PCloud_Alpha_pub;
ros::Publisher PCloud_Alpha_left_pub;
ros::Publisher PCloud_Alpha_right_pub;

void callbackRadFront(const msgs::Rad::ConstPtr& msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointXYZI temp;

  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    if (msg->radPoint[i].speed < 80)
    {
      temp.x = msg->radPoint[i].x;
      temp.y = -msg->radPoint[i].y;
      cloud->points.push_back(temp);
    }
  }
  sensor_msgs::PointCloud2 msgtemp;
  pcl::toROSMsg(*cloud, msgtemp);
  msgtemp.header = msg->radHeader;
  msgtemp.header.seq = msg->radHeader.seq;
  msgtemp.header.frame_id = "base_link";
  PCloud_pub.publish(msgtemp);
}

void callbackRadFrontAlpha(const msgs::Rad::ConstPtr& msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointXYZI temp;

  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    temp.x = msg->radPoint[i].x;
    temp.y = -msg->radPoint[i].y;
    cloud->points.push_back(temp);
  }
  sensor_msgs::PointCloud2 msgtemp;
  pcl::toROSMsg(*cloud, msgtemp);
  msgtemp.header = msg->radHeader;
  msgtemp.header.seq = msg->radHeader.seq;
  msgtemp.header.frame_id = "alpha_front";
  PCloud_pub.publish(msgtemp);
}

void callbackRadFrontLeft(const msgs::Rad::ConstPtr& msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointXYZI temp;

  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    temp.x = msg->radPoint[i].x;
    temp.y = -msg->radPoint[i].y;
    cloud->points.push_back(temp);
  }
  sensor_msgs::PointCloud2 msgtemp;
  pcl::toROSMsg(*cloud, msgtemp);
  msgtemp.header = msg->radHeader;
  msgtemp.header.seq = msg->radHeader.seq;
  msgtemp.header.frame_id = "alpha_front_left";
  PCloud_pub.publish(msgtemp);
}

void callbackRadFrontRight(const msgs::Rad::ConstPtr& msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointXYZI temp;

  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    temp.x = msg->radPoint[i].x;
    temp.y = -msg->radPoint[i].y;
    cloud->points.push_back(temp);
  }
  sensor_msgs::PointCloud2 msgtemp;
  pcl::toROSMsg(*cloud, msgtemp);
  msgtemp.header = msg->radHeader;
  msgtemp.header.seq = msg->radHeader.seq;
  msgtemp.header.frame_id = "alpha_front_right";
  PCloud_pub.publish(msgtemp);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "RadFrontSub_PCloud");
  ros::NodeHandle n;

  ros::Subscriber RadFrontSub = n.subscribe("RadFront", 1, callbackRadFront);
  ros::Subscriber RadFrontAlphaSub = n.subscribe("RadFrontAlpha", 1, callbackRadFrontAlpha);
  ros::Subscriber RadFrontLeftSub = n.subscribe("RadFrontLeft", 1, callbackRadFrontLeft);
  ros::Subscriber RadFrontRightSub = n.subscribe("RadFrontRight", 1, callbackRadFrontRight);

  PCloud_pub = n.advertise<sensor_msgs::PointCloud2>("radar_point_cloud", 1);
  PCloud_Alpha_pub = n.advertise<sensor_msgs::PointCloud2>("radar_alpha", 1);
  PCloud_Alpha_left_pub = n.advertise<sensor_msgs::PointCloud2>("radar_alpha_left", 1);
  PCloud_Alpha_right_pub = n.advertise<sensor_msgs::PointCloud2>("radar_alpha_right", 1);

  ros::Rate rate(20);
  while (ros::ok())
  {
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}
