// ROS message
#include "ros/ros.h"
#include "std_msgs/Header.h"
#include <ros/package.h>
#include <iostream>

#include <msgs/DetectedObjectArray.h>
#include "camera_params.h"  // include camera topic name

// Subscriber: 8 cams
ros::Subscriber CamObjFR;
ros::Subscriber CamObjFC;
ros::Subscriber CamObjFL;
ros::Subscriber CamObjFT;

ros::Subscriber CamObjRF;
ros::Subscriber CamObjRB;

ros::Subscriber CamObjLF;
ros::Subscriber CamObjLB;

ros::Subscriber CamObjBT;

// Publisher: 1, represent all cams
ros::Publisher CamObjAll;

std_msgs::Header HeaderAll;

std::vector<msgs::DetectedObject> arrCamObjFR;
std::vector<msgs::DetectedObject> arrCamObjFC;
std::vector<msgs::DetectedObject> arrCamObjFL;
std::vector<msgs::DetectedObject> arrCamObjFT;
std::vector<msgs::DetectedObject> arrCamObjRF;
std::vector<msgs::DetectedObject> arrCamObjRB;
std::vector<msgs::DetectedObject> arrCamObjLF;
std::vector<msgs::DetectedObject> arrCamObjLB;
std::vector<msgs::DetectedObject> arrCamObjBT;

