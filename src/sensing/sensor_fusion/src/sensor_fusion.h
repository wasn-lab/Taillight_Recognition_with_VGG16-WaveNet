#ifndef __FUSION_H__
#define __FUSION_H__

#include <iostream>
#include <csignal>
#include "ros/ros.h"
#include <msgs/DetectedObjectArray.h>
#include <msgs/DetectedObject.h>

#include "Hungarian.h"

#define DEBUG 0
/************************************************************************/
ros::Subscriber lidar_sub;
ros::Subscriber camera_sub;
ros::Publisher fusion_pub;
/************************************************************************/
unsigned int seq = 0;
msgs::DetectedObjectArray lidar_msg;
msgs::DetectedObjectArray camera_msg;
msgs::DetectedObjectArray fusion_msg;
/************************************************************************/
std::vector<msgs::DetectedObject> lidar_objects;
std::vector<msgs::DetectedObject> camera_objs;
std::vector<msgs::DetectedObject> fusion_objects;
/************************************************************************/
void callback_lidar(const msgs::DetectedObjectArray::ConstPtr& lidar_obj_array);
void callback_camera_main(const msgs::DetectedObjectArray::ConstPtr& camera_obj_array,
                          msgs::DetectedObjectArray& camera_msg);
void callback_camera(const msgs::DetectedObjectArray::ConstPtr& camera_obj_array);
/************************************************************************/
void fuseDetectedObjects();
constexpr double FUSE_RANGE_SED = 9.;  // 3 x 3
constexpr double FUSE_INVALID = 10000.;
std::vector<std::vector<double> > distance_table_;
void get_obj_center(double& obj_cx, double obj_cy, const msgs::DetectedObject& obj);
void init_distance_table(std::vector<msgs::DetectedObject>& objs1, std::vector<msgs::DetectedObject>& objs2);
void associate_data(std::vector<msgs::DetectedObject>& objs1, std::vector<msgs::DetectedObject>& objs2);
/************************************************************************/
void MySigintHandler(int sig);
int main(int argc, char** argv);

#endif  // __FUSION_H__