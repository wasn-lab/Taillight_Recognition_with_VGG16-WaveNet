#ifndef SAVE_ROUTE_H
#define SAVE_ROUTE_H

// ROS
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <std_msgs/Float64.h>
#include <tf/tf.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/package.h>

// others
#include <string>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

class SaveRoute
{
public:
  SaveRoute();
  ros::Subscriber initial_pose_subscriber;
  ros::Subscriber goal_subscriber;
  ros::Subscriber checkpoint_subscriber;
  ~SaveRoute();

  string fpname;
  string fpname_s;
  fstream file;


protected: 

  geometry_msgs::PoseStamped initial_pose_;
  geometry_msgs::PoseStamped goal_pose_;
  geometry_msgs::PoseStamped start_pose_;
  std::vector<geometry_msgs::PoseStamped> checkpoints_;
  
  void SaveInitialPose(const geometry_msgs::PoseWithCovarianceStampedConstPtr & initialpose_msg_ptr);
  void SaveGoalPose(const geometry_msgs::PoseStampedConstPtr & goal_msg_ptr);
  void SaveCheckPoint(const geometry_msgs::PoseStampedConstPtr & checkpoint_msg_ptr);
  ros::NodeHandle node;

private:
  
};

#endif  // SAVE_ROUTE_H
