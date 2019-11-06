#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h> 

void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;
  
  transformStamped.header.stamp = ros::Time::now();
  transformStamped.header.frame_id = "map";
  transformStamped.child_frame_id = "base_link";
  transformStamped.transform.translation.x = msg->pose.position.x;
  transformStamped.transform.translation.y = msg->pose.position.y;
  transformStamped.transform.translation.z = msg->pose.position.z;
  transformStamped.transform.rotation.x = msg->pose.orientation.x;
  transformStamped.transform.rotation.y = msg->pose.orientation.y;
  transformStamped.transform.rotation.z = msg->pose.orientation.z;
  transformStamped.transform.rotation.w = msg->pose.orientation.w;

  br.sendTransform(transformStamped);
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "my_tf2_broadcaster");
    
  tf2_ros::StaticTransformBroadcaster br1;
  geometry_msgs::TransformStamped transformStamped1;

  transformStamped1.header.stamp = ros::Time::now();
  transformStamped1.header.frame_id = "base_link";
  transformStamped1.child_frame_id = "lidar";
  transformStamped1.transform.translation.x = 0;
  transformStamped1.transform.translation.y = 0;
  transformStamped1.transform.translation.z = 0;
  transformStamped1.transform.rotation.x = 0;
  transformStamped1.transform.rotation.y = 0;
  transformStamped1.transform.rotation.z = 0;
  transformStamped1.transform.rotation.w = 1;

  br1.sendTransform(transformStamped1);

  ros::NodeHandle node;
  ros::Subscriber sub = node.subscribe("current_pose", 10, &poseCallback);

  ros::spin();
  return 0;
};