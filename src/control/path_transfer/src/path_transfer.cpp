#include <ros/ros.h>
#include <autoware_planning_msgs/Trajectory.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/Pose.h>
#include <tf/tf.h>
#include <msgs/CurrentTrajInfo.h>

#define RT_PI 3.14159265358979323846

double angle_diff_setting = RT_PI/4;
double z_diff_setting = 2.0;
int LRturn = -1;
bool up_hill = false;

ros::Publisher nav_path_pub;
ros::Publisher currenttrajinfo_pub;

void calculate_LRturn(autoware_planning_msgs::Trajectory traj)
{
  double size = traj.points.size();
  double first_yaw = tf::getYaw(traj.points[0].pose.orientation);
  double last_yaw = tf::getYaw(traj.points[size-1].pose.orientation);
  LRturn = -1;
  double angle_diff = last_yaw - first_yaw;
  if (angle_diff < -RT_PI)
  {
    angle_diff = angle_diff + 2*RT_PI;
  }
  else if (angle_diff > RT_PI)
  {
    angle_diff = angle_diff - 2*RT_PI;
  }

  if (angle_diff > angle_diff_setting) //Left turn
  {
    LRturn = 1;
  }
  else if (angle_diff < -angle_diff_setting) //Right turn
  {
    LRturn = 2;
  }
  else //Straight
  {
    LRturn = 0;
  }
  // std::cout << "first_yaw : " << first_yaw << std::endl;
  // std::cout << "last_yaw : " << last_yaw << std::endl;
  // std::cout << "angle_diff_setting : " << angle_diff_setting << std::endl;
  // std::cout << "angle_diff : " << angle_diff << std::endl;
  // std::cout << "LRturn : " << LRturn << std::endl;
}

void calculate_slope(autoware_planning_msgs::Trajectory traj)
{
  double size = traj.points.size();
  double first_z = traj.points[0].pose.position.z;
  double last_z = traj.points[size-1].pose.position.z;
  up_hill = false;
  double z_diff = last_z - first_z;
  if (z_diff > z_diff_setting)
  {
    up_hill = true;
  }
  // std::cout << "first_z : " << first_z << std::endl;
  // std::cout << "last_z : " << last_z << std::endl;
  // std::cout << "z_diff_setting : " << z_diff_setting << std::endl;
  // std::cout << "z_diff : " << z_diff << std::endl;
  // std::cout << "up_hill : " << up_hill << std::endl;
}

void publish_CurrentTrajInfo()
{
  msgs::CurrentTrajInfo info;
  info.header.frame_id = "rear_wheel";
  info.header.stamp = ros::Time::now();
  info.LRturn = LRturn;
  info.Uphill = up_hill;
  currenttrajinfo_pub.publish(info);
}

void transfer_callback(const autoware_planning_msgs::Trajectory& traj)
{
  calculate_LRturn(traj);
  calculate_slope(traj);
  publish_CurrentTrajInfo();
  
  nav_msgs::Path current_path;
  geometry_msgs::PoseStamped current_posestamped;

  current_posestamped.header.frame_id = traj.header.frame_id;
  current_posestamped.header.stamp = ros::Time::now();
  current_path.header.frame_id = traj.header.frame_id;
  current_path.header.stamp = ros::Time::now();
 
  int traj_size = traj.points.size();
  for (int i = 0; i < traj_size; i++)
  {
    current_posestamped.pose.position.x = traj.points[i].pose.position.x;
    current_posestamped.pose.position.y = traj.points[i].pose.position.y;
    current_posestamped.pose.position.z = traj.points[i].pose.position.z; 
    current_path.poses.push_back(current_posestamped);
  }
  nav_path_pub.publish(current_path);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "path_transfer");
  ros::NodeHandle node;

  ros::param::get(ros::this_node::getName()+"/angle_diff_setting", angle_diff_setting);
  ros::param::get(ros::this_node::getName()+"/z_diff_setting", z_diff_setting);

  ros::Subscriber safety_waypoints_sub = node.subscribe("/planning/scenario_planning/trajectory", 1, transfer_callback);
  nav_path_pub = node.advertise<nav_msgs::Path>("nav_path_astar_final",1);
  currenttrajinfo_pub = node.advertise<msgs::CurrentTrajInfo>("current_trajectory_info",1);

  ros::spin();
  return 0;
};