#include <ros/ros.h>
#include <autoware_msgs/Lane.h>
#include <autoware_msgs/Waypoint.h> 
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <ros/package.h>
#include <fstream>
#include "libwaypoint_follower/libwaypoint_follower.h"
#include <std_msgs/Int32.h>
#include <std_msgs/Float64.h>
#include <nav_msgs/Path.h>
#include <math.h>

ros::Publisher basepath_pub;
ros::Publisher closetwaypoint_pub;
ros::Publisher obstacletwaypoint_pub;
ros::Publisher NavPath_Pub;
ros::Publisher rearcurrentpose_pub;

#define RT_PI 3.14159265358979323846

autoware_msgs::Lane waypoints_init;
autoware_msgs::Lane local_waypoints_;

int closest_local_index_ = -1;
int search_size_ = 30;
int closet_local_start_i = -10;
double wheel_dis = 3.8;

double seg_id[2000] = {};
double seg_x[2000] = {};
double seg_y[2000] = {};
double seg_z[2000] = {};
double seg_h[2000] = {};
double seg_l[2000] = {};
int read_index = 0;

int pre_obswaypoints_data = -1;
int obs_index = 0;

template <int size_readtmp>
void read_txt(std::string fpname, double (&SEG_ID)[size_readtmp],double (&SEG_X)[size_readtmp],double (&SEG_Y)[size_readtmp],double (&SEG_Z)[size_readtmp],double (&SEG_H)[size_readtmp],double (&SEG_L)[size_readtmp])
{
  // int read_index = 0;
  std::string fname = fpname;

    std::ifstream fin;
    char line[300];
    memset( line, 0, sizeof(line));

    fin.open(fname.c_str(),std::ios::in);
    if(!fin) 
    {
        std::cout << "Fail to import txt" <<std::endl;
        exit(1);
    }

    while(fin.getline(line,sizeof(line),'\n')) 
    {
      std::string nmea_str(line);
      std::stringstream ss(nmea_str);
      std::string token;

      getline(ss,token, ',');
      SEG_ID[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      SEG_X[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      SEG_Y[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      SEG_Z[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      SEG_H[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      SEG_L[read_index] = atof(token.c_str());
      // std::cout << read_index << ":" << SEG_L[read_index] <<std::endl;
      read_index += 1;
    }
}

void Ini_obs_bytxt()
{
  std::string fpname = ros::package::getPath("astar_initial");
  std::string fpname_s = fpname + "/data/20191127_waypoints_round.txt";
  read_txt(fpname_s, seg_id, seg_x, seg_y, seg_z, seg_h, seg_l);
}

void globalpathinit()
{
  autoware_msgs::Lane msg_;
  autoware_msgs::Waypoint pointmsg_;
  msg_.header.frame_id = "map";
  msg_.header.stamp = ros::Time::now();
  pointmsg_.pose.header.frame_id = "map";
  pointmsg_.pose.header.stamp = ros::Time::now();
  for (int i=0;i<read_index;i++)
  {
    pointmsg_.pose.pose.position.x = seg_x[i];
    pointmsg_.pose.pose.position.y = seg_y[i];
    pointmsg_.pose.pose.position.z = seg_z[i];
    double yaw = seg_h[i]*RT_PI/180.0;
    double yaw_ = yaw;
    if (yaw > RT_PI)
      yaw_ = yaw - 2*RT_PI;
    tf::Quaternion lane_q;
    lane_q.setRPY( 0, 0, yaw_);
    pointmsg_.pose.pose.orientation.x = lane_q.x();
    pointmsg_.pose.pose.orientation.y = lane_q.y();
    pointmsg_.pose.pose.orientation.z = lane_q.z();
    pointmsg_.pose.pose.orientation.w = lane_q.w();
    pointmsg_.twist.twist.linear.x = 1;
    pointmsg_.twist.twist.linear.y = 0;
    pointmsg_.twist.twist.linear.z = 0;

    msg_.waypoints.push_back(pointmsg_);
  }
  waypoints_init = msg_;
}

int getLocalClosestWaypoint(const autoware_msgs::Lane& waypoints, const geometry_msgs::Pose& pose, const int& search_size)
{
  static autoware_msgs::Lane local_waypoints;  // around self-vehicle
  const int prev_index = closest_local_index_;

  // search in all waypoints if lane_select judges you're not on waypoints
  if (closest_local_index_ == -1)
  {
    closest_local_index_ = getClosestWaypoint(waypoints, pose);
    // std::cout << "--------------" << closest_local_index_ << std::endl;
  }
  // search in limited area based on prev_index
  else
  {
    // std::cout << "+++++++++++++++++" << closest_local_index_ << std::endl;
    // get neighborhood waypoints around prev_index

    // int start_index = std::max(0, prev_index - search_size / 2);
    // int end_index = std::min(prev_index + search_size / 2, (int)waypoints.waypoints.size());
    // auto start_itr = waypoints.waypoints.begin() + start_index;
    // auto end_itr = waypoints.waypoints.begin() + end_index;
    // local_waypoints.waypoints = std::vector<autoware_msgs::Waypoint>(start_itr, end_itr);
    // // get closest waypoint in neighborhood waypoints
    // closest_local_index_ = start_index + getClosestWaypoint(local_waypoints, pose);

    int start_index = prev_index - search_size / 2;
    if (start_index < 0)
      start_index = start_index + read_index;

    // get closest waypoint in neighborhood waypoints
    closest_local_index_ = start_index + getClosestWaypoint(local_waypoints_, pose);
    // std::cout << "closest_local_index0_ : " << closest_local_index_ << std::endl;
    if (closest_local_index_ >= read_index)
      closest_local_index_ = closest_local_index_ - read_index;
    if (closest_local_index_ < 0)
      closest_local_index_ = closest_local_index_ + read_index;
  }
  return closest_local_index_;
}

void basepathgen_pub(int closet_i)
{
  autoware_msgs::Lane msg;
  autoware_msgs::Waypoint pointmsg;
  msg.header.frame_id = "map";
  msg.header.stamp = ros::Time::now();
  pointmsg.pose.header.frame_id = "map";
  pointmsg.pose.header.stamp = ros::Time::now();
  // rviz path
  nav_msgs::Path Dpath;
  geometry_msgs::PoseStamped Dpose;
  Dpath.header.frame_id = "map";
  Dpose.header.frame_id = "map";

  for (int i = closet_local_start_i; i < 1000; i++)
  {
    int j = i + closet_i;
    if (j >= read_index)
      j = j - read_index;
    if (j < 0)
      j = j + read_index;
    // pointmsg.pose.pose.position.x = seg_x[j];
    // pointmsg.pose.pose.position.y = seg_y[j];
    // pointmsg.pose.pose.position.z = seg_z[j];
    // double yaw = seg_h[j]*RT_PI/180.0;
    // double yaw_ = yaw;
    // if (yaw > RT_PI)
    //   yaw_ = yaw - 2*RT_PI;
    // tf::Quaternion lane_q;
    // lane_q.setRPY( 0, 0, yaw_);
    // pointmsg.pose.pose.orientation.x = lane_q.x();
    // pointmsg.pose.pose.orientation.y = lane_q.y();
    // pointmsg.pose.pose.orientation.z = lane_q.z();
    // pointmsg.pose.pose.orientation.w = lane_q.w();
    // pointmsg.twist.twist.linear.x = 1;//-5.39247;
    // pointmsg.twist.twist.linear.y = 0;//-32.24074;
    // pointmsg.twist.twist.linear.z = 0;//-2.855071;

    // std::cout << "lane_q.x() : " << lane_q.x() << std::endl;
    // std::cout << "lane_q.y() : " << lane_q.y() << std::endl;
    // std::cout << "lane_q.z() : " << lane_q.z() << std::endl;
    // std::cout << "lane_q.w() : " << lane_q.w() << std::endl;

    pointmsg = waypoints_init.waypoints[j];
    msg.waypoints.push_back(pointmsg);

    Dpose.pose = pointmsg.pose.pose;
    Dpath.poses.push_back(Dpose);
  }
  basepath_pub.publish(msg);
  NavPath_Pub.publish(Dpath);
}

void localpathgen(int closet_i)
{
  autoware_msgs::Lane msg__;
  autoware_msgs::Waypoint pointmsg__;
  msg__.header.frame_id = "map";
  msg__.header.stamp = ros::Time::now();
  pointmsg__.pose.header.frame_id = "map";
  pointmsg__.pose.header.stamp = ros::Time::now();
  for (int i = -search_size_/2; i < search_size_/2; i++)
  {
    int j = i + closet_i;
    if (j >= read_index)
      j = j - read_index;
    if (j < 0)
      j = j + read_index;
    // pointmsg.pose.pose.position.x = seg_x[j];
    // pointmsg.pose.pose.position.y = seg_y[j];
    // pointmsg.pose.pose.position.z = seg_z[j];
    // double yaw = seg_h[j]*RT_PI/180.0;
    // double yaw_ = yaw;
    // if (yaw > RT_PI)
    //   yaw_ = yaw - 2*RT_PI;
    // tf::Quaternion lane_q;
    // lane_q.setRPY( 0, 0, yaw_);
    // pointmsg.pose.pose.orientation.x = lane_q.x();
    // pointmsg.pose.pose.orientation.y = lane_q.y();
    // pointmsg.pose.pose.orientation.z = lane_q.z();
    // pointmsg.pose.pose.orientation.w = lane_q.w();
    // pointmsg.twist.twist.linear.x = 1;//-5.39247;
    // pointmsg.twist.twist.linear.y = 0;//-32.24074;
    // pointmsg.twist.twist.linear.z = 0;//-2.855071;

    // std::cout << "lane_q.x() : " << lane_q.x() << std::endl;
    // std::cout << "lane_q.y() : " << lane_q.y() << std::endl;
    // std::cout << "lane_q.z() : " << lane_q.z() << std::endl;
    // std::cout << "lane_q.w() : " << lane_q.w() << std::endl;

    pointmsg__ = waypoints_init.waypoints[j];

    msg__.waypoints.push_back(pointmsg__);
  }
  local_waypoints_ = msg__;
}

void CurrentPoseCallback(const geometry_msgs::PoseStamped& CPmsg)
{
  geometry_msgs::PoseStamped pose = CPmsg;
  geometry_msgs::PoseStamped rear_pose = pose;

  double roll, pitch, yaw;
  tf::Quaternion lidar_q(CPmsg.pose.orientation.x, CPmsg.pose.orientation.y, CPmsg.pose.orientation.z,CPmsg.pose.orientation.w);
  tf::Matrix3x3 lidar_m(lidar_q);
  lidar_m.getRPY(roll, pitch, yaw);

  rear_pose.pose.position.x = pose.pose.position.x - wheel_dis*std::cos(yaw);
  rear_pose.pose.position.y = pose.pose.position.y - wheel_dis*std::sin(yaw);
  rearcurrentpose_pub.publish(rear_pose);

  int closet_index = -1;
  while (closet_index < 0)
  {
    closet_index = getLocalClosestWaypoint(waypoints_init,rear_pose.pose,search_size_); // rear or front
  }
  std::cout << "closet_index = " << closet_index << std::endl;
  localpathgen(closet_index);
  basepathgen_pub(closet_index);

  std_msgs::Int32 closetwaypoint;
  closetwaypoint.data = -closet_local_start_i;//closet_index;
  closetwaypoint_pub.publish(closetwaypoint);

  // std::cout << "CPmsg.pose.orientation.x : " << CPmsg.pose.orientation.x << std::endl;
  // std::cout << "CPmsg.pose.orientation.y : " << CPmsg.pose.orientation.y << std::endl;
  // std::cout << "CPmsg.pose.orientation.z : " << CPmsg.pose.orientation.z << std::endl;
  // std::cout << "CPmsg.pose.orientation.w : " << CPmsg.pose.orientation.w << std::endl;
}

void obsdisCallback(const std_msgs::Float64::ConstPtr& obsdismsg)
{
  int obswaypoints_data = std::ceil(obsdismsg->data + wheel_dis);
  if (obswaypoints_data > 30)
    obswaypoints_data = -1;
  int obswaypoints_data_ = obswaypoints_data;

  ///////////////////////////////////////////////////////////////////
  if (obswaypoints_data <= pre_obswaypoints_data + 1 && obswaypoints_data >= pre_obswaypoints_data)
    obs_index += 1;
  else
    obs_index = 0;

  if (obs_index < 10) // detect time < 3s
    obswaypoints_data_ = -1;
  ///////////////////////////////////////////////////////////////////

  std_msgs::Int32 obswaypoints;
  obswaypoints.data = obswaypoints_data_;
  std::cout << "obswaypoints.data : " << obswaypoints.data << std::endl;
  obstacletwaypoint_pub.publish(obswaypoints);
  pre_obswaypoints_data = obswaypoints_data;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "astar_initial");
  ros::NodeHandle node;
  Ini_obs_bytxt();
  globalpathinit();
  ros::Subscriber current_pose_sub = node.subscribe("current_pose", 1, CurrentPoseCallback);
  ros::Subscriber obstacle_dis_sub = node.subscribe("Geofence_PC", 1, obsdisCallback);
  basepath_pub = node.advertise<autoware_msgs::Lane>("base_waypoints", 10, true);
  closetwaypoint_pub = node.advertise<std_msgs::Int32>("closest_waypoint", 10, true);
  obstacletwaypoint_pub = node.advertise<std_msgs::Int32>("obstacle_waypoint", 10, true);
  NavPath_Pub = node.advertise<nav_msgs::Path>("nav_path_astar_base", 10, true);
  rearcurrentpose_pub = node.advertise<geometry_msgs::PoseStamped>("rear_current_pose", 1, true);
  // ros::Rate loop_rate(0.0001);
  // while (ros::ok())
  // { 
  
  //   ros::spinOnce();
  //   loop_rate.sleep();   
  // }

  ros::spin();
  return 0;
};
