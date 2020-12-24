#include <ros/ros.h>
#include <tf/tf.h>
#include <fstream>
#include <std_msgs/Int32.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>
#include <cmath>
#include <msgs/Flag_Info.h>

ros::Publisher enable_avoid_pub;
ros::Publisher avoiding_path_pub;

#define RT_PI 3.14159265358979323846

bool enable_avoidance_ = false;
bool force_disable_avoidance_ = false;
int obs_index_base = 0;
double project_dis = 100;

void obsdisbaseCallback(const std_msgs::Float64::ConstPtr& obsdismsg_base)
{
  double obswaypoints_data_base = obsdismsg_base->data;
  if (obswaypoints_data_base > 40) ///////////////////////
  {
    obswaypoints_data_base = -1;
  }

  // if (obswaypoints_data_base == -1)
  // {
  //   obs_index_base += 1;
  // }
  // else
  // {
  //   obs_index_base = 0;
  // }

  std_msgs::Int32 avoid_path;
  // if (obs_index_base > 20)
  // {
  //   avoid_path.data = 1;
  // }
  // else
  // {
  //   avoid_path.data = 0;
  // }

  if (enable_avoidance_ == 1 && obswaypoints_data_base == -1 && project_dis < 0.5)
  {
    avoid_path.data = 1;
  }
  else
  {
    avoid_path.data = 0;
  }
  
  avoiding_path_pub.publish(avoid_path);
}

void avoidstatesubCallback(const msgs::Flag_Info& msg)
{
  double avoid_state_index_ = msg.Dspace_Flag03;
  // std::cout << "avoid_state_index_ : " << avoid_state_index_ << std::endl;
  std_msgs::Bool enable_avoidance;// = false;
  std::cout << "force_disable_avoidance_ : " << force_disable_avoidance_ << std::endl;
  if (avoid_state_index_ == 1 && !force_disable_avoidance_)
  {
    enable_avoidance.data = true;
  }
  else
  {
    enable_avoidance.data = false;
  }
  enable_avoidance_ = enable_avoidance.data;
  enable_avoid_pub.publish(enable_avoidance);
}

void overshootorigdisCallback(const std_msgs::Float64& msg)
{
  project_dis = msg.data;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "decision_maker");
  ros::NodeHandle node;

  ros::param::get(ros::this_node::getName()+"/force_disable_avoidance", force_disable_avoidance_);

  ros::Subscriber avoid_state_sub = node.subscribe("Flag_Info01", 1, avoidstatesubCallback);
  ros::Subscriber obstacle_dis_1_sub = node.subscribe("Geofence_original", 1, obsdisbaseCallback);
  ros::Subscriber veh_overshoot_orig_dis_sub = node.subscribe("veh_overshoot_orig_dis", 1, overshootorigdisCallback);
  enable_avoid_pub = node.advertise<std_msgs::Bool>("/planning/scenario_planning/lane_driving/motion_planning/obstacle_avoidance_planner/enable_avoidance", 10, true);
  avoiding_path_pub = node.advertise<std_msgs::Int32>("avoidpath_reach_goal", 10, true);

  // ros::Rate loop_rate(10);
  // while (ros::ok())
  // { 
  //   ros::spinOnce();
  //   loop_rate.sleep();   
  // }

  ros::spin();
  return 0;
};
