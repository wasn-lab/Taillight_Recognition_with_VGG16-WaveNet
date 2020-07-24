#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <sensor_msgs/Imu.h>
#include <tf/tf.h>
#include <fstream>
#include <std_msgs/Int32.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>
#include <cmath>
#include <autoware_perception_msgs/DynamicObject.h>
#include <autoware_perception_msgs/DynamicObjectArray.h>
#include <msgs/VehInfo.h>

//For PCL
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

ros::Publisher rearcurrentpose_pub;
ros::Publisher enable_avoid_pub;
ros::Publisher objects_pub;
ros::Publisher nogroundpoints_pub;
ros::Publisher twist_pub;

static sensor_msgs::Imu imu_data_rad;

#define RT_PI 3.14159265358979323846

double wheel_dis = 3.8;
bool avoid_flag = 0;

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
}

void avoidingflagCallback(const std_msgs::Int32::ConstPtr& avoidflagmsg)
{
  avoid_flag = avoidflagmsg->data;
}

void objectsCallback(const autoware_perception_msgs::DynamicObjectArray& objectsmsg)
{
  objects_pub.publish(objectsmsg);
}

void LidnogroundpointCallback(const sensor_msgs::PointCloud2& Lngpmsg)
{
  nogroundpoints_pub.publish(Lngpmsg);
}

void currentVelocityCallback(const msgs::VehInfo::ConstPtr& msg)
{
  geometry_msgs::TwistStamped veh_twist;
  veh_twist.header.frame_id = "rear_wheel";
  veh_twist.header.stamp = ros::Time::now();
  veh_twist.twist.linear.x = msg->ego_speed;
  veh_twist.twist.angular.x = imu_data_rad.angular_velocity.x;
  veh_twist.twist.angular.y = imu_data_rad.angular_velocity.y;
  veh_twist.twist.angular.z = imu_data_rad.angular_velocity.z;
  twist_pub.publish(veh_twist);
}

void imudataCallback(const sensor_msgs::Imu& msg)
{
  imu_data_rad = msg;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "planning_initial");
  ros::NodeHandle node;
  ros::Subscriber current_pose_sub = node.subscribe("current_pose", 1, CurrentPoseCallback);
  ros::Subscriber avoiding_flag_sub = node.subscribe("avoiding_path", 1, avoidingflagCallback);
  ros::Subscriber objects_sub = node.subscribe("objects", 1, objectsCallback);
  ros::Subscriber Lidnogroundpoint_sub = node.subscribe("ring_edge_point_cloud", 1, LidnogroundpointCallback); // /LidarAll/NonGround2
  ros::Subscriber velocity_sub = node.subscribe("veh_info",1,currentVelocityCallback);
  ros::Subscriber imu_data_sub = node.subscribe("imu_data_rad",1,imudataCallback);
  rearcurrentpose_pub = node.advertise<geometry_msgs::PoseStamped>("rear_current_pose", 1, true);
  objects_pub = node.advertise<autoware_perception_msgs::DynamicObjectArray>("output/pp_objects", 1, true);
  nogroundpoints_pub = node.advertise<sensor_msgs::PointCloud2>("output/lidar_no_ground", 1, true);
  twist_pub = node.advertise<geometry_msgs::TwistStamped>("/localization/twist", 1, true);
  // enable_avoid_pub = node.advertise<std_msgs::Bool>("enable_avoid", 10, true);
  // ros::Rate loop_rate(0.0001);
  // while (ros::ok())
  // { 
  
  //   ros::spinOnce();
  //   loop_rate.sleep();   
  // }

  ros::spin();
  return 0;
};
