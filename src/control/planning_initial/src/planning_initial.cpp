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
#include <msgs/Spat.h>
#include <autoware_perception_msgs/LampState.h>
#include <autoware_perception_msgs/TrafficLightState.h>
#include <autoware_perception_msgs/TrafficLightStateArray.h>
#include "msgs/Flag_Info.h"

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
ros::Publisher trafficlight_pub;

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

void trafficCallback(const msgs::Spat::ConstPtr& msg)
{
  int light_status = (int)(msg->signal_state);
  double confidence = 1.0;
  autoware_perception_msgs::LampState lampstate;
  autoware_perception_msgs::TrafficLightState trafficlightstate;
  autoware_perception_msgs::TrafficLightStateArray trafficlightstatearray;
  trafficlightstatearray.header.frame_id = "map";
  trafficlightstatearray.header.stamp = ros::Time::now();
  trafficlightstate.id = 402079;
  if (light_status == 129) // red
  {
    lampstate.type = autoware_perception_msgs::LampState::RED;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  else if(light_status == 130) // yellow
  {
    lampstate.type = autoware_perception_msgs::LampState::YELLOW;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  else if(light_status == 48) // green straight + green right
  {
    lampstate.type = autoware_perception_msgs::LampState::UP;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
    lampstate.type = autoware_perception_msgs::LampState::RIGHT;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  else if(light_status == 9) // red + green left
  {
    lampstate.type = autoware_perception_msgs::LampState::RED;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
    lampstate.type = autoware_perception_msgs::LampState::LEFT;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  else // unknown
  {
    lampstate.type = autoware_perception_msgs::LampState::UNKNOWN;
    lampstate.confidence = 0.0;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  trafficlightstatearray.states.push_back(trafficlightstate);
  // trafficlight_pub.publish(trafficlightstatearray);
}

void trafficDspaceCallback(const msgs::Flag_Info::ConstPtr& msg)
{
  int light_status = int(msg->Dspace_Flag02);
  int countDown = int(msg->Dspace_Flag03);
  double confidence = 1.0;
  autoware_perception_msgs::LampState lampstate;
  autoware_perception_msgs::TrafficLightState trafficlightstate;
  autoware_perception_msgs::TrafficLightStateArray trafficlightstatearray;
  trafficlightstatearray.header.frame_id = "map";
  trafficlightstatearray.header.stamp = ros::Time::now();
  trafficlightstate.id = 402079;
  if (light_status == 129) // red
  {
    lampstate.type = autoware_perception_msgs::LampState::RED;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  else if(light_status == 130) // yellow
  {
    lampstate.type = autoware_perception_msgs::LampState::YELLOW;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  else if(light_status == 48) // green straight + green right
  {
    lampstate.type = autoware_perception_msgs::LampState::UP;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
    lampstate.type = autoware_perception_msgs::LampState::RIGHT;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  else if(light_status == 9) // red + green left
  {
    lampstate.type = autoware_perception_msgs::LampState::RED;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
    lampstate.type = autoware_perception_msgs::LampState::LEFT;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  else // unknown
  {
    lampstate.type = autoware_perception_msgs::LampState::UNKNOWN;
    lampstate.confidence = 0.0;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  trafficlightstatearray.states.push_back(trafficlightstate);
  trafficlight_pub.publish(trafficlightstatearray);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "planning_initial");
  ros::NodeHandle node;
  ros::Subscriber current_pose_sub = node.subscribe("current_pose", 1, CurrentPoseCallback);
  ros::Subscriber avoiding_flag_sub = node.subscribe("avoiding_path", 1, avoidingflagCallback);
  ros::Subscriber objects_sub = node.subscribe("input/objects", 1, objectsCallback);
  ros::Subscriber Lidnogroundpoint_sub = node.subscribe("input/lidar_no_ground", 1, LidnogroundpointCallback); // /LidarAll/NonGround2
  ros::Subscriber velocity_sub = node.subscribe("veh_info",1,currentVelocityCallback);
  ros::Subscriber imu_data_sub = node.subscribe("imu_data_rad",1,imudataCallback);
  ros::Subscriber traffic_sub = node.subscribe("/traffic", 1, trafficCallback);
  ros::Subscriber traffic_Dspace_sub = node.subscribe("/Flag_Info02", 1, trafficDspaceCallback);
  rearcurrentpose_pub = node.advertise<geometry_msgs::PoseStamped>("rear_current_pose", 1, true);
  objects_pub = node.advertise<autoware_perception_msgs::DynamicObjectArray>("output/objects", 1, true);
  nogroundpoints_pub = node.advertise<sensor_msgs::PointCloud2>("output/lidar_no_ground", 1, true);
  twist_pub = node.advertise<geometry_msgs::TwistStamped>("/localization/twist", 1, true);
  trafficlight_pub = node.advertise<autoware_perception_msgs::TrafficLightStateArray>("output/traffic_light", 1, true);
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
