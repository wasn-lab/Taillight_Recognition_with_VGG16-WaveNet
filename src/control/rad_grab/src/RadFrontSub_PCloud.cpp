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

// radfront , 正規路線
ros::Publisher pcloud_pub;

// alpha test
ros::Publisher alpha_front_center_pub;
ros::Publisher alpha_front_left_pub;
ros::Publisher alpha_front_right_pub;
ros::Publisher alpha_side_left_pub;
ros::Publisher alpha_side_right_pub;
ros::Publisher alpha_back_left_pub;
ros::Publisher alpha_back_right_pub;

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
  pcloud_pub.publish(msgtemp);
}

void callbackFrontCenter(const msgs::Rad::ConstPtr& msg)
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
  msgtemp.header.frame_id = "rad_fc";
  alpha_front_center_pub.publish(msgtemp);
}

void callbackFrontLeft(const msgs::Rad::ConstPtr& msg)
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
  msgtemp.header.frame_id = "rad_fl";
  alpha_front_left_pub.publish(msgtemp);
}

void callbackFrontRight(const msgs::Rad::ConstPtr& msg)
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
  msgtemp.header.frame_id = "rad_fr";
  alpha_front_right_pub.publish(msgtemp);
}
void callbackSideLeft(const msgs::Rad::ConstPtr& msg)
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
  msgtemp.header.frame_id = "rad_sl";
  alpha_side_left_pub.publish(msgtemp);
}

void callbackSideRight(const msgs::Rad::ConstPtr& msg)
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
  msgtemp.header.frame_id = "rad_sr";
  alpha_side_right_pub.publish(msgtemp);
}
void callbackBackLeft(const msgs::Rad::ConstPtr& msg)
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
  msgtemp.header.frame_id = "rad_bl";
  alpha_back_left_pub.publish(msgtemp);
}

void callbackBackRight(const msgs::Rad::ConstPtr& msg)
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
  msgtemp.header.frame_id = "rad_br";
  alpha_back_right_pub.publish(msgtemp);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "RadFrontSub_PCloud");
  ros::NodeHandle n;

  ros::Subscriber RadFrontSub = n.subscribe("RadFront", 1, callbackRadFront);
  ros::Subscriber RadFrontCenterSub = n.subscribe("AlphaFrontCenter", 1, callbackFrontCenter);
  ros::Subscriber RadFrontLeftSub = n.subscribe("AlphaFrontLeft", 1, callbackFrontLeft);
  ros::Subscriber RadFrontRightSub = n.subscribe("AlphaFrontRight", 1, callbackFrontRight);
  ros::Subscriber RadSideLeftSub = n.subscribe("AlphaSideLeft", 1, callbackSideLeft);
  ros::Subscriber RadSideRightSub = n.subscribe("AlphaSideRight", 1, callbackSideRight);
  ros::Subscriber RadBackLeftSub = n.subscribe("AlphaBackLeft", 1, callbackBackLeft);
  ros::Subscriber RadBackRightSub = n.subscribe("AlphaBackRight", 1, callbackBackRight);

  pcloud_pub = n.advertise<sensor_msgs::PointCloud2>("radar_point_cloud", 1);

  alpha_front_center_pub = n.advertise<sensor_msgs::PointCloud2>("rad_front_center", 1);
  alpha_front_left_pub = n.advertise<sensor_msgs::PointCloud2>("rad_front_left", 1);
  alpha_front_right_pub = n.advertise<sensor_msgs::PointCloud2>("rad_front_right", 1);
  alpha_side_left_pub = n.advertise<sensor_msgs::PointCloud2>("rad_side_left", 1);
  alpha_side_right_pub = n.advertise<sensor_msgs::PointCloud2>("rad_side_right", 1);
  alpha_back_left_pub = n.advertise<sensor_msgs::PointCloud2>("rad_back_left", 1);
  alpha_back_right_pub = n.advertise<sensor_msgs::PointCloud2>("rad_back_right", 1);

  ros::Rate rate(20);
  while (ros::ok())
  {
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}
