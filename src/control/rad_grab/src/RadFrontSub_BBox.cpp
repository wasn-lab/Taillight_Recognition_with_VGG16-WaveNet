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
#include <string.h>
#include <visualization_msgs/Marker.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
using namespace std;

ros::Publisher BBox_pub;



void callbackRadFront(const msgs::Rad::ConstPtr &msg)
{
 

  std_msgs::Header h = msg->radHeader;
	ros::NodeHandle n;  
	uint32_t shape = visualization_msgs::Marker::SPHERE;
  msgs::DetectedObjectArray BBox;
	msgs::DetectedObject Box_temp;
	//BBox.objects.resize(msg->radPoint.size());
	
	for (int i = 0; i < msg->radPoint.size(); i++)
	{
		if(msg->radPoint[i].speed<80){
				Box_temp.relSpeed = msg->radPoint[i].speed;
				Box_temp.bPoint.p0.x = msg->radPoint[i].x;
				Box_temp.bPoint.p0.y = -msg->radPoint[i].y;
				Box_temp.bPoint.p3.x = msg->radPoint[i].x;
				Box_temp.bPoint.p3.y = -msg->radPoint[i].y;
				Box_temp.bPoint.p4.x = msg->radPoint[i].x;
				Box_temp.bPoint.p4.y = -msg->radPoint[i].y;
				Box_temp.bPoint.p7.x = msg->radPoint[i].x;
				Box_temp.bPoint.p7.y = -msg->radPoint[i].y;
				BBox.objects.push_back(Box_temp);
				cout << "x = "<< Box_temp.bPoint.p0.x << endl;
				cout << "y = "<< Box_temp.bPoint.p0.y << endl;
				cout << "speed = "<< Box_temp.relSpeed << endl;
			}
		
	}
	cout << "Number of objects = "<< BBox.objects.size() << endl;	
	BBox_pub.publish(BBox);
}


int main(int argc, char **argv)
{

  ros::init(argc, argv, "RadFrontSub_BBox");  
  ros::NodeHandle n;
  ros::Subscriber RadFrontSub = n.subscribe("RadFront", 1, callbackRadFront);
	BBox_pub = n.advertise<msgs::DetectedObjectArray>("PathPredictionOutput/radar", 1); 
  ros::Rate rate(100);
  while (ros::ok())
  {
    ros::spinOnce();
		rate.sleep();
  }
  return 0;
}
