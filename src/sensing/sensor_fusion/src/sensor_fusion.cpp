#include <iostream>
#include <signal.h>
#include "ros/ros.h"
#include <msgs/DetectedObjectArray.h>
#include <msgs/DetectedObject.h>

/************************************************************************/
std_msgs::Header lidarHeader;
std_msgs::Header cam60_0_Header;
std_msgs::Header cam60_1_Header;
std_msgs::Header cam60_2_Header;
std_msgs::Header cam30_0_Header;
std_msgs::Header cam30_1_Header;
std_msgs::Header cam30_2_Header;
std_msgs::Header cam120_0_Header;
std_msgs::Header cam120_1_Header;
std_msgs::Header cam120_2_Header;
/************************************************************************/
msgs::DetectedObjectArray msgLidarObj;
msgs::DetectedObjectArray msgCam60_0_Obj;
msgs::DetectedObjectArray msgCam60_1_Obj;
msgs::DetectedObjectArray msgCam60_2_Obj;
msgs::DetectedObjectArray msgCam30_0_Obj;
msgs::DetectedObjectArray msgCam30_1_Obj;
msgs::DetectedObjectArray msgCam30_2_Obj;
msgs::DetectedObjectArray msgCam120_0_Obj;
msgs::DetectedObjectArray msgCam120_1_Obj;
msgs::DetectedObjectArray msgCam120_2_Obj;
/************************************************************************/
uint32_t seq = 0;
msgs::DetectedObjectArray msgFusionObj;
ros::Publisher fusion_pub;
void fuseDetectedObjects();
/************************************************************************/
std::vector<msgs::DetectedObject> vDetectedObjectDF;
std::vector<msgs::DetectedObject> vDetectedObjectLID;
std::vector<msgs::DetectedObject> vDetectedObjectCAM_60_0;
std::vector<msgs::DetectedObject> vDetectedObjectCAM_60_1;
std::vector<msgs::DetectedObject> vDetectedObjectCAM_60_2;
std::vector<msgs::DetectedObject> vDetectedObjectTemp;
std::vector<msgs::DetectedObject> vDetectedObjectCAM_30_1;
std::vector<msgs::DetectedObject> vDetectedObjectCAM_120_1;
/************************************************************************/
msgs::DetectedObjectArray msgLidar_60_0_Obj;
msgs::DetectedObjectArray msgLidar_60_1_Obj;
msgs::DetectedObjectArray msgLidar_60_2_Obj;
msgs::DetectedObjectArray msgLidar_30_0_Obj;
msgs::DetectedObjectArray msgLidar_30_1_Obj;
msgs::DetectedObjectArray msgLidar_30_2_Obj;
msgs::DetectedObjectArray msgLidar_120_0_Obj;
msgs::DetectedObjectArray msgLidar_120_1_Obj;
msgs::DetectedObjectArray msgLidar_120_2_Obj;
msgs::DetectedObjectArray msgLidar_others_Obj;
msgs::DetectedObjectArray msgLidar_rear_Obj;
msgs::DetectedObjectArray msgLidar_frontshort;
/**************************************************************************/

void MySigintHandler(int sig)
{
  ROS_INFO("****** MySigintHandler ******");
  if (sig == SIGINT)
  {
    ROS_INFO("END SensorFusion");
    ros::shutdown();
  }
}

void LidarDetectionCb(const msgs::DetectedObjectArray::ConstPtr& lidar_obj_array)
{
  msgLidarObj.header = lidar_obj_array->header;

  std::vector<msgs::DetectedObject>().swap(msgLidarObj.objects);
  msgLidarObj.objects.reserve(lidar_obj_array->objects.size());

  for (const auto& obj : lidar_obj_array->objects)
  {
    msgLidarObj.objects.push_back(obj);
  }

  fuseDetectedObjects();
}

void callback_camera_main(const msgs::DetectedObjectArray::ConstPtr& cam_obj_array,
                          msgs::DetectedObjectArray& msg_cam_obj)
{
  msg_cam_obj.header = cam_obj_array->header;

  std::vector<msgs::DetectedObject>().swap(msg_cam_obj.objects);
  msg_cam_obj.objects.reserve(cam_obj_array->objects.size());

  for (const auto& obj : cam_obj_array->objects)
  {
    if (obj.distance >= 0)
    {
      msg_cam_obj.objects.push_back(obj);
    }
  }
}

void cam60_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_0_ObjArray)
{
  callback_camera_main(Cam60_0_ObjArray, msgCam60_0_Obj);
}

void cam60_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_1_ObjArray)
{
  callback_camera_main(Cam60_1_ObjArray, msgCam60_1_Obj);
}

void cam60_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_2_ObjArray)
{
  callback_camera_main(Cam60_2_ObjArray, msgCam60_2_Obj);
}

void cam30_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_0_ObjArray)
{
  callback_camera_main(Cam30_0_ObjArray, msgCam30_0_Obj);
}

void cam30_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_1_ObjArray)
{
  callback_camera_main(Cam30_1_ObjArray, msgCam30_1_Obj);
}

void cam30_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_2_ObjArray)
{
  callback_camera_main(Cam30_2_ObjArray, msgCam30_2_Obj);
}

void cam120_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_0_ObjArray)
{
  callback_camera_main(Cam120_0_ObjArray, msgCam120_0_Obj);
}

void cam120_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_1_ObjArray)
{
  callback_camera_main(Cam120_1_ObjArray, msgCam120_1_Obj);
}

void cam120_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_2_ObjArray)
{
  callback_camera_main(Cam120_2_ObjArray, msgCam120_2_Obj);
}

void fuseDetectedObjects()
{
  std::cout << "**************** do_fusion ****************" << std::endl;

  std::vector<msgs::DetectedObject>().swap(vDetectedObjectDF);
  std::vector<msgs::DetectedObject>().swap(vDetectedObjectLID);
  std::vector<msgs::DetectedObject>().swap(vDetectedObjectCAM_60_0);
  std::vector<msgs::DetectedObject>().swap(vDetectedObjectCAM_60_1);
  std::vector<msgs::DetectedObject>().swap(vDetectedObjectCAM_60_2);
  std::vector<msgs::DetectedObject>().swap(vDetectedObjectCAM_30_1);
  std::vector<msgs::DetectedObject>().swap(vDetectedObjectCAM_120_1);

  /************************************************************************/

  for (auto& obj : msgLidarObj.objects)
  {
    obj.header = msgLidarObj.header;
    vDetectedObjectLID.push_back(obj);
  }
  std::cout << "num_lidar_objs = " << vDetectedObjectLID.size() << std::endl;

  /************************************************************************/

  for (auto& obj : msgCam60_0_Obj.objects)
  {
    obj.header = msgCam60_0_Obj.header;
    vDetectedObjectCAM_60_0.push_back(obj);
  }
  std::cout << "num_cam60_0_objs = " << vDetectedObjectCAM_60_0.size() << std::endl;

  /************************************************************************/

  for (auto& obj : msgCam60_1_Obj.objects)
  {
    obj.header = msgCam60_1_Obj.header;
    vDetectedObjectCAM_60_1.push_back(obj);
  }
  std::cout << "num_cam60_1_objs = " << vDetectedObjectCAM_60_1.size() << std::endl;

  /************************************************************************/

  for (auto& obj : msgCam60_2_Obj.objects)
  {
    obj.header = msgCam60_2_Obj.header;
    vDetectedObjectCAM_60_2.push_back(obj);
  }
  std::cout << "num_cam60_2_objs = " << vDetectedObjectCAM_60_2.size() << std::endl;

  /************************************************************************/

  for (auto& obj : msgCam30_1_Obj.objects)
  {
    obj.header = msgCam30_1_Obj.header;
    vDetectedObjectCAM_30_1.push_back(obj);
  }
  std::cout << "num_cam30_0_objs = " << vDetectedObjectCAM_30_1.size() << std::endl;

  /************************************************************************/

  for (auto& obj : msgCam120_1_Obj.objects)
  {
    obj.header = msgCam120_1_Obj.header;
    vDetectedObjectCAM_120_1.push_back(obj);
  }
  std::cout << "num_cam120_0_objs = " << vDetectedObjectCAM_120_1.size() << std::endl;

  /************************************************************************/

  for (const auto& obj : vDetectedObjectLID)
  {
    vDetectedObjectDF.push_back(obj);
  }

  for (const auto& obj : vDetectedObjectCAM_60_0)
  {
    vDetectedObjectDF.push_back(obj);
  }

  for (const auto& obj : vDetectedObjectCAM_60_1)
  {
    vDetectedObjectDF.push_back(obj);
  }

  for (const auto& obj : vDetectedObjectCAM_60_2)
  {
    vDetectedObjectDF.push_back(obj);
  }

  for (const auto& obj : vDetectedObjectCAM_30_1)
  {
    vDetectedObjectDF.push_back(obj);
  }

  for (const auto& obj : vDetectedObjectCAM_120_1)
  {
    vDetectedObjectDF.push_back(obj);
  }

  std::cout << "num_total_objs = " << vDetectedObjectDF.size() << std::endl;

  msgFusionObj.header.stamp = msgLidarObj.header.stamp;
  msgFusionObj.header.frame_id = "lidar";
  msgFusionObj.header.seq = seq++;
  std::vector<msgs::DetectedObject>().swap(msgFusionObj.objects);
  msgFusionObj.objects.assign(vDetectedObjectDF.begin(), vDetectedObjectDF.end());

  fusion_pub.publish(msgFusionObj);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sensor_fusion");
  ros::NodeHandle nh;

  ros::Subscriber lidar_det_sub = nh.subscribe("/LidarDetection", 1, LidarDetectionCb);
  ros::Subscriber cam_F_right_sub = nh.subscribe("/CamObjFrontRight", 1, cam60_0_DetectionCb);
  ros::Subscriber cam_F_center_sub = nh.subscribe("/CamObjFrontCenter", 1, cam60_1_DetectionCb);
  ros::Subscriber cam_F_left_sub = nh.subscribe("/CamObjFrontLeft", 1, cam60_2_DetectionCb);

  fusion_pub = nh.advertise<msgs::DetectedObjectArray>("SensorFusion", 2);

  signal(SIGINT, MySigintHandler);

  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();
}
