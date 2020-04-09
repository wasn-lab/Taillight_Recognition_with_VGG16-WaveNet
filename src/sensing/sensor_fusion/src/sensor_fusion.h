#ifndef __FUSION_H__
#define __FUSION_H__

#include <iostream>
#include <csignal>
#include "ros/ros.h"
#include <msgs/DetectedObjectArray.h>
#include <msgs/DetectedObject.h>

#include "hungarian.h"

#define DEBUG_HUNGARIAN 0
#define DEBUG_OBJCLASS 0
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
const int fusion_objclass_source = 0;  // 0: camera; 1: lidar
void fuseObjClass(uint16_t obj1_class, const size_t obj1_id, const std::string obj1_source, const uint16_t obj2_class,
                  const size_t obj2_id, const std::string obj2_source);
/************************************************************************/
void MySigintHandler(int sig);
int main(int argc, char** argv);

#endif  // __FUSION_H__