#ifndef __FUSION_H__
#define __FUSION_H__

#include <iostream>
#include <signal.h>
#include "ros/ros.h"
#include <msgs/DetectedObjectArray.h>
#include <msgs/DetectedObject.h>

#include "Hungarian.h"

#define DEBUG 0
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
unsigned int seq = 0;
msgs::DetectedObjectArray msgFusionObj;
ros::Publisher fusion_pub;
void fuseDetectedObjects();
constexpr double FUSE_RANGE_SED = 9.;  // 3 x 3
constexpr double FUSE_INVALID = 10000.;
std::vector<std::vector<double> > distance_table_;
void init_distance_table(std::vector<msgs::DetectedObject>& objs1, std::vector<msgs::DetectedObject>& objs2);
void associate_data(std::vector<msgs::DetectedObject>& objs1, std::vector<msgs::DetectedObject>& objs2);
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

void MySigintHandler(int sig);
void fuseDetectedObjects();
void get_obj_center(double& obj_cx, double obj_cy, const msgs::DetectedObject& obj);
void init_distance_table(std::vector<msgs::DetectedObject>& objs1, std::vector<msgs::DetectedObject>& objs2);
void associate_data(std::vector<msgs::DetectedObject>& objs1, std::vector<msgs::DetectedObject>& objs2);
int main(int argc, char** argv);

void LidarDetectionCb(const msgs::DetectedObjectArray::ConstPtr& lidar_obj_array);
void callback_camera_main(const msgs::DetectedObjectArray::ConstPtr& cam_obj_array,
                          msgs::DetectedObjectArray& msg_cam_obj);
void cam60_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_0_ObjArray);
void cam60_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_1_ObjArray);
void cam60_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_2_ObjArray);
void cam30_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_0_ObjArray);
void cam30_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_1_ObjArray);
void cam30_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_2_ObjArray);
void cam120_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_0_ObjArray);
void cam120_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_1_ObjArray);
void cam120_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_2_ObjArray);

#endif  // __FUSION_H__