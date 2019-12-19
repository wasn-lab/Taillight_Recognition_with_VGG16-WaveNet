#include <ros/ros.h>
#include <autoware_msgs/Lane.h>
#include <autoware_msgs/Waypoint.h> 
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <fstream>
#include <std_msgs/Int32.h>
#include <nav_msgs/Path.h>
#include <target_planner/MM_TP_msg.h>
#include <target_planner/UKF_MM_msg.h>
#include <msgs/VehInfo.h>
#include <cstdlib>
#include <cmath>

#define RT_PI 3.14159265358979323846

ros::Publisher rear_target_pub;
ros::Publisher rear_vehicle_target_pub;
ros::Publisher front_target_pub;
ros::Publisher front_vehicle_target_pub;

struct pose
{
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
    double speed;
};

pose current_pose;
pose current_pose_ukf;
pose rear_current_pose;

struct targetpoint
{
    double x;
    double y;
    double z;
};

targetpoint rear_targetpoint;
targetpoint rear_vehicle_targetpoint;
targetpoint front_targetpoint;
targetpoint front_vehicle_targetpoint;

nav_msgs::Path final_waypoints;

double Look_ahead_time = 1.6;
double Look_ahead_S0 = 3;
double current_velocity_ = 0;

bool current_pose_ini = false;
bool ukfmm_ini = false;
bool final_path_ini = false;
bool current_velocity_initialized_ = false;

bool checkInitialized()
{
  bool initialized = false;
  // check for relay mode
  initialized = (current_pose_ini && ukfmm_ini && final_path_ini);
  return initialized;
}

void targetplanner(pose pose, targetpoint& target, targetpoint& vehicle_target, double wheel_dis, ros::Publisher target_pub, ros::Publisher vehicle_target_pub)
{ 
  double dis = 0.0;
  double ahead_dis = current_velocity_ * Look_ahead_time + Look_ahead_S0;
  // std::cout << "ahead_dis : " << ahead_dis << std::endl;
  double prev_pose_x_ = final_waypoints.poses[0].pose.position.x;
  double prev_pose_y_ = final_waypoints.poses[0].pose.position.y;
  double next_pose_x_ = final_waypoints.poses[1].pose.position.x;
  double next_pose_y_ = final_waypoints.poses[1].pose.position.y;
  double dis_waypoints_ = std::sqrt((next_pose_x_-prev_pose_x_)*(next_pose_x_-prev_pose_x_)+(next_pose_y_-prev_pose_y_)*(next_pose_y_-prev_pose_y_));
  double a_x = pose.x - prev_pose_x_;
  double a_y = pose.y - prev_pose_y_;
  double b_x = next_pose_x_ - prev_pose_x_;
  double b_y = next_pose_y_ - prev_pose_y_;
  double o_x = ((a_x*b_x + a_y*b_y)*b_x)/(dis_waypoints_);
  double ahead_dis_ = 0.0;
  // std::cout << "std::fabs(a_x*b_x + a_y*b_y) : " << std::fabs(a_x*b_x + a_y*b_y) << std::endl;
  // std::cout << "abs(a_x*b_x + a_y*b_y) : " << abs(a_x*b_x + a_y*b_y) << std::endl;
  // std::cout << next_pose_x_ << "/" << prev_pose_x_ << "/" << o_x << std::endl;
  // if ((a_x*b_x + a_y*b_y) > 0)
    // ahead_dis_ = ahead_dis + std::fabs(a_x*b_x + a_y*b_y)/std::sqrt(dis_waypoints_) + wheel_dis;
  // else
    // ahead_dis_ = ahead_dis - std::fabs(a_x*b_x + a_y*b_y)/std::sqrt(dis_waypoints_) + wheel_dis;
  // if ((a_x*b_x + a_y*b_y) > 0)
    ahead_dis_ = ahead_dis + (a_x*b_x + a_y*b_y)/std::sqrt(dis_waypoints_) + wheel_dis;
  // else
  //   ahead_dis_ = ahead_dis - (a_x*b_x + a_y*b_y)/std::sqrt(dis_waypoints_) + wheel_dis;
  // std::cout << "ahead_dis_ : " << ahead_dis_ << std::endl;
  int waypoints_size = final_waypoints.poses.size();
  int index = 0;
  double dis_waypoints = 0.0;
  // std::cout << "waypoints_size : " << waypoints_size << std::endl;
  for (int i = 1; i < waypoints_size; i++)
  {
    index = i;
    double prev_pose_x = final_waypoints.poses[i-1].pose.position.x;
    double prev_pose_y = final_waypoints.poses[i-1].pose.position.y;
    double next_pose_x = final_waypoints.poses[i].pose.position.x;
    double next_pose_y = final_waypoints.poses[i].pose.position.y;
    dis_waypoints = std::sqrt((next_pose_x - prev_pose_x)*(next_pose_x - prev_pose_x)+(next_pose_y - prev_pose_y)*(next_pose_y - prev_pose_y));
    dis += dis_waypoints;
    if (dis >= ahead_dis_)
      break;
  }
  // std::cout << "index : " << index << std::endl;
  double diff = dis - ahead_dis_;
  // std::cout << "diff : " << diff << std::endl;
  target.x = final_waypoints.poses[index].pose.position.x + (final_waypoints.poses[index-1].pose.position.x - final_waypoints.poses[index].pose.position.x)*diff/dis_waypoints;
  target.y = final_waypoints.poses[index].pose.position.y + (final_waypoints.poses[index-1].pose.position.y - final_waypoints.poses[index].pose.position.y)*diff/dis_waypoints;
  target.z = final_waypoints.poses[index].pose.position.z + (final_waypoints.poses[index-1].pose.position.z - final_waypoints.poses[index].pose.position.z)*diff/dis_waypoints;
  // std::cout << "target_x : " << target_x << std::endl;
  // std::cout << "target_y : " << target_y << std::endl;
  geometry_msgs::PoseStamped target_pose;
  target_pose.header.frame_id = "map";
  target_pose.header.stamp = ros::Time::now();
  target_pose.pose.position.x = target.x;
  target_pose.pose.position.y = target.y;
  target_pose.pose.position.z = target.z;
  target_pose.pose.orientation.w = 1.0;
  target_pub.publish(target_pose);

  double rot_ang = -pose.yaw;
  vehicle_target.x = (target.x-pose.x)*std::cos(rot_ang) - (target.y-pose.y)*std::sin(rot_ang);
  vehicle_target.y = (target.x-pose.x)*std::sin(rot_ang) + (target.y-pose.y)*std::cos(rot_ang);
  geometry_msgs::PoseStamped vehicle_target_pose;
  vehicle_target_pose.header.frame_id = "base_link";
  vehicle_target_pose.header.stamp = ros::Time::now();
  vehicle_target_pose.pose.position.x = vehicle_target.x;
  vehicle_target_pose.pose.position.y = vehicle_target.y;
  vehicle_target_pose.pose.position.z = target.z;
  vehicle_target_pose.pose.orientation.w = 1.0;
  vehicle_target_pub.publish(vehicle_target_pose);
}

void run()
{
  targetplanner(rear_current_pose, rear_targetpoint, rear_vehicle_targetpoint, 0.0, rear_target_pub, rear_vehicle_target_pub);
  std::cout << "rear_vehicle_targetpoint.x = " << rear_vehicle_targetpoint.x << std::endl;

  targetplanner(rear_current_pose, front_targetpoint, front_vehicle_targetpoint, 3.8, front_target_pub, front_vehicle_target_pub);
  std::cout << "front_vehicle_targetpoint.x = " << front_vehicle_targetpoint.x << std::endl;
}

void currentposeCallback(const geometry_msgs::PoseStamped::ConstPtr& PSmsg)
{ 
  tf::Quaternion lidar_q(PSmsg->pose.orientation.x, PSmsg->pose.orientation.y, PSmsg->pose.orientation.z,PSmsg->pose.orientation.w);
  tf::Matrix3x3 lidar_m(lidar_q);

  current_pose.x = PSmsg->pose.position.x;
  current_pose.y = PSmsg->pose.position.y;
  current_pose.z = PSmsg->pose.position.z;
  lidar_m.getRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);

  if (current_pose.yaw < 0)
    current_pose.yaw = current_pose.yaw + 2*RT_PI;
  if (current_pose.yaw >= 2*RT_PI)
    current_pose.yaw = current_pose.yaw - 2*RT_PI;
  current_pose_ini = true;
}

void rear_currentposeCallback(const geometry_msgs::PoseStamped::ConstPtr& PSmsg)
{ 
  tf::Quaternion lidar_q(PSmsg->pose.orientation.x, PSmsg->pose.orientation.y, PSmsg->pose.orientation.z,PSmsg->pose.orientation.w);
  tf::Matrix3x3 lidar_m(lidar_q);

  rear_current_pose.x = PSmsg->pose.position.x;
  rear_current_pose.y = PSmsg->pose.position.y;
  rear_current_pose.z = PSmsg->pose.position.z;
  lidar_m.getRPY(rear_current_pose.roll, rear_current_pose.pitch, rear_current_pose.yaw);

  if (rear_current_pose.yaw < 0)
    rear_current_pose.yaw = rear_current_pose.yaw + 2*RT_PI;
  if (rear_current_pose.yaw >= 2*RT_PI)
    rear_current_pose.yaw = rear_current_pose.yaw - 2*RT_PI;
  current_pose_ini = true;
}

void ukfmm_callback(const target_planner::UKF_MM_msg::ConstPtr& UKFMMmsg)
{
  current_pose_ukf.x = UKFMMmsg->X_UKF_SLAM[0];
  current_pose_ukf.y = UKFMMmsg->X_UKF_SLAM[1];
  current_pose_ukf.z = current_pose.z;
  current_pose_ukf.roll = current_pose.roll;
  current_pose_ukf.pitch = current_pose.pitch;
  current_pose_ukf.yaw = UKFMMmsg->X_UKF_SLAM[2];
  current_pose_ukf.speed = UKFMMmsg->X_UKF_SLAM[4];
  Look_ahead_time = UKFMMmsg->Look_ahead_time;
  ukfmm_ini = true; 
}

void final_waypoints_callback(const nav_msgs::Path& SWmsg)
{
  final_waypoints = SWmsg;
  final_path_ini = true;
}

void currentVelocityCallback(const msgs::VehInfo::ConstPtr& msg)
{
  current_velocity_ = msg->ego_speed;
  current_velocity_initialized_ = true;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "target_planner");
  ros::NodeHandle node;
  ros::Subscriber current_pose_sub = node.subscribe("current_pose", 1, currentposeCallback);
  ros::Subscriber rear_current_pose_sub = node.subscribe("rear_current_pose", 1, rear_currentposeCallback);
  ros::Subscriber ukfmm_sub = node.subscribe("ukf_mm_topic", 1, ukfmm_callback);
  ros::Subscriber safety_waypoints_sub = node.subscribe("nav_path_astar_final", 1, final_waypoints_callback);
  ros::Subscriber velocity_sub = node.subscribe("veh_info",1,currentVelocityCallback);
  rear_target_pub = node.advertise<geometry_msgs::PoseStamped>("rear_target_point",1);
  rear_vehicle_target_pub = node.advertise<geometry_msgs::PoseStamped>("rear_vehicle_target_point",1);
  front_target_pub = node.advertise<geometry_msgs::PoseStamped>("front_target_point",1);
  front_vehicle_target_pub = node.advertise<geometry_msgs::PoseStamped>("front_vehicle_target_point",1);

  ros::Rate loop_rate(100);
  while (ros::ok())
  {
    ros::spinOnce();
    if (checkInitialized())
    {
      break;
    }
    ROS_WARN("Waiting for subscribing topics...");
    ros::Duration(1.0).sleep();
  }
  while (ros::ok())
  { 
    run();
    ROS_INFO("Publish target point");
    ros::spinOnce();
    loop_rate.sleep();   
  }

  // ros::spin();
  return 0;
};
