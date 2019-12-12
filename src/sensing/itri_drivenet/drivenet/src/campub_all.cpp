#include "ros/ros.h"
#include "std_msgs/Header.h"
#include <ros/package.h>

#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <boost/thread.hpp>
#include <pthread.h>
#include <thread>
#include <future>
#include <mutex>

#include "drivenet/drivenet_60_b1.h"

#include <msgs/DetectedObjectArray.h>

using namespace DriveNet;

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

// msgs::DetectedObjectArray arrCamObjFR;
// msgs::DetectedObjectArray arrCamObjFC;
// msgs::DetectedObjectArray arrCamObjFL;
// msgs::DetectedObjectArray arrCamObjFT;
// msgs::DetectedObjectArray arrCamObjRF;
// msgs::DetectedObjectArray arrCamObjRB;
// msgs::DetectedObjectArray arrCamObjLF;
// msgs::DetectedObjectArray arrCamObjLB;
// msgs::DetectedObjectArray arrCamObjBT;

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


void callback_CamObjFR(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  arrCamObjFR = DetectMsg->objects;
}

void callback_CamObjFC(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  HeaderAll = DetectMsg->header;
  arrCamObjFC = DetectMsg->objects;
}

void callback_CamObjFL(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  
  arrCamObjFL = DetectMsg->objects;
}

void callback_CamObjFT(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  
  arrCamObjFT = DetectMsg->objects;
}

void callback_CamObjRF(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  
  arrCamObjRF = DetectMsg->objects;
}

void callback_CamObjRB(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  
  arrCamObjRB = DetectMsg->objects;
}

void callback_CamObjLF(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  
  arrCamObjLF = DetectMsg->objects;
}

void callback_CamObjLB(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  
  arrCamObjLB = DetectMsg->objects;
}

void callback_CamObjBT(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  
  arrCamObjBT = DetectMsg->objects;
}

void collectRepub()
{
  // arrCamObjAll = arrCamObjBT + arrCamObjLB;
  msgs::DetectedObjectArray arrCamObjAll;
  arrCamObjAll.header = HeaderAll;

  for(size_t i = 0; i < arrCamObjFR.size(); i++)
  {
    arrCamObjAll.objects.push_back(arrCamObjFR[i]);
  }

  for(size_t i = 0; i < arrCamObjFC.size(); i++)
  {
    arrCamObjAll.objects.push_back(arrCamObjFC[i]);
  }

  for(size_t i = 0; i < arrCamObjFL.size(); i++)
  {
    arrCamObjAll.objects.push_back(arrCamObjFL[i]);
  }

  for(size_t i = 0; i < arrCamObjFT.size(); i++)
  {
    arrCamObjAll.objects.push_back(arrCamObjFT[i]);
  }

  for(size_t i = 0; i < arrCamObjRF.size(); i++)
  {
    arrCamObjAll.objects.push_back(arrCamObjRF[i]);
  }

  for(size_t i = 0; i < arrCamObjRB.size(); i++)
  {
    arrCamObjAll.objects.push_back(arrCamObjRB[i]);
  }

  for(size_t i = 0; i < arrCamObjLF.size(); i++)
  {
    arrCamObjAll.objects.push_back(arrCamObjLF[i]);
  }

  for(size_t i = 0; i < arrCamObjLB.size(); i++)
  {
    arrCamObjAll.objects.push_back(arrCamObjLB[i]);
  }

  for(size_t i = 0; i < arrCamObjBT.size(); i++)
  {
    arrCamObjAll.objects.push_back(arrCamObjBT[i]);
  }
  
  ROS_INFO_STREAM("HEADER: " << arrCamObjAll.header);  

  CamObjAll.publish(arrCamObjAll);
}



int main(int argc, char** argv)
{
  ros::init(argc, argv, "CamPub");
  ros::NodeHandle nh;

  // ros::param::get(ros::this_node::getName() + "/car_id", car_id);
  // cam60_0_topicName = camera::topics[camera::id::right_60];
  // cam60_0 = nh.subscribe(cam60_0_topicName + std::string("/compressed"), 1, callback_60_0_decode);

  // pub60_0 = nh.advertise<msgs::DetectedObjectArray>("/CamObjFrontRight", 8);

  // Subscribe msgs
  CamObjFR = nh.subscribe("/CamObjFrontRight", 1, callback_CamObjFR);
  CamObjFC = nh.subscribe("/CamObjFrontCenter", 1, callback_CamObjFC);
  CamObjFL = nh.subscribe("/CamObjFrontLeft", 1, callback_CamObjFL);
  CamObjFT = nh.subscribe("/CamObjFrontTop", 1, callback_CamObjFT);
  CamObjRF = nh.subscribe("/CamObjRightFront", 1, callback_CamObjRF);
  CamObjRB = nh.subscribe("/CamObjRightBack", 1, callback_CamObjRB);
  CamObjLF = nh.subscribe("/CamObjLeftFront", 1, callback_CamObjLF);
  CamObjLB = nh.subscribe("/CamObjLeftBack", 1, callback_CamObjLB);
  CamObjBT = nh.subscribe("/CamObjBackTop", 1, callback_CamObjBT);

  CamObjAll = nh.advertise<msgs::DetectedObjectArray>("/CamObjAll", 8);

  ros::Rate loop_rate(30);

  while (ros::ok())
  {
    collectRepub();
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
