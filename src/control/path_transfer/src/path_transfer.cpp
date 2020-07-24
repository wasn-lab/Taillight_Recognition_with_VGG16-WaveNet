#include <ros/ros.h>
#include <autoware_planning_msgs/Trajectory.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/Pose.h>



ros::Publisher nav_path_pub;



void transfer_callback(const autoware_planning_msgs::Trajectory & traj)
{
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
  ros::init(argc, argv, "target_planner");
  ros::NodeHandle node;
  ros::Subscriber safety_waypoints_sub = node.subscribe("/planning/scenario_planning/trajectory", 1, transfer_callback);
  nav_path_pub = node.advertise<nav_msgs::Path>("nav_path_astar_final",1);

  ros::spin();
  return 0;
};


