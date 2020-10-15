#include <ros/ros.h>
#include <msgs/BehaviorSceneRegister.h>
#include <autoware_planning_msgs/StopReason.h>
#include <autoware_planning_msgs/StopReasonArray.h>
#include <geometry_msgs/PoseStamped.h>

ros::Publisher bus_stop_register_pub;

msgs::BehaviorSceneRegister bus_stop_register_last;

void bus_stop_register(const msgs::BehaviorSceneRegister::ConstPtr& msg)
{
  msgs::BehaviorSceneRegister bus_stop_register;
  bus_stop_register.Module =  msg->Module;
  bus_stop_register.ModuleId =  msg->ModuleId;
  bus_stop_register.RegisterFlag = msg->RegisterFlag;
  bus_stop_register.StopZone = msg->RegisterFlag;
  bus_stop_register.Distance = 0;

  bus_stop_register_last = bus_stop_register;

  bus_stop_register_pub.publish(bus_stop_register);
}

void register_callback(const msgs::BehaviorSceneRegister::ConstPtr& msg)
{
  if (msg->Module == "bus_stop")
  {
    bus_stop_register(msg);
  }
}

void stop_reasons_callback(const autoware_planning_msgs::StopReasonArray::ConstPtr& msg)
{
  std::cout << "msg->stop_reasons[0].reason : " << msg->stop_reasons[0].reason << std::endl;
  if (msg->stop_reasons[0].reason == "\"BusStop\"")
  {
    std::cout << "--------------------" << std::endl;
  }
}

void current_pose_callback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "scene_register_checker");
  ros::NodeHandle node;

  ros::Subscriber behavior_scene_register_sub = node.subscribe("/planning/scenario_planning/status/behavior_scene_register", 1, register_callback);
  ros::Subscriber stop_reasons_sub = node.subscribe("/planning/scenario_planning/status/stop_reasons", 1, stop_reasons_callback);
  ros::Subscriber current_pose_sub = node.subscribe("/current_pose", 1, current_pose_callback);
  bus_stop_register_pub = node.advertise<msgs::BehaviorSceneRegister>("bus_stop_register_info",1);

  ros::spin();
  return 0;
};