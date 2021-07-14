#include <ros/ros.h>
#include <tf/tf.h>
#include <fstream>
#include <std_msgs/Int32.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>
#include <cmath>
#include <msgs/Flag_Info.h>
#include <msgs/LaneEvent.h>

ros::Publisher enable_avoid_pub;
ros::Publisher overtake_over_pub;
ros::Publisher enable_lane_change_by_obstacle_pub;

#define RT_PI 3.14159265358979323846

bool enable_avoidance_ = false;
bool force_disable_avoidance_ = false;
int obs_index_base = 0;
double project_dis = 100;

bool lane_event_enable_overtake = true;
bool disable_lane_event_ = false;

bool lane_change_ready_ = false;

void obsdisbaseCallback(const std_msgs::Float64::ConstPtr& obsdismsg_base)
{
  double obswaypoints_data_base = obsdismsg_base->data;
  if (obswaypoints_data_base > 30) ///////////////////////
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

  std_msgs::Int32 overtake_over;
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
    overtake_over.data = 1;
  }
  else
  {
    overtake_over.data = 0;
  }

  // if (!lane_event_enable_overtake && project_dis < 0.5)
  // {
  //   overtake_over.data = 1;
  // }
  
  overtake_over_pub.publish(overtake_over);
}

void lanechangereadyCallback(const std_msgs::Bool& msg)
{
  lane_change_ready_ = msg.data;
}

// void avoidstatesubCallback(const msgs::Flag_Info& msg)
// {
//   double avoid_state_index_ = msg.Dspace_Flag03;
//   // std::cout << "avoid_state_index_ : " << avoid_state_index_ << std::endl;
//   std_msgs::Bool enable_avoidance;// = false;
//   // std::cout << "force_disable_avoidance_ : " << force_disable_avoidance_ << std::endl;
//   if (avoid_state_index_ == 1 && !force_disable_avoidance_)
//   {
//     enable_avoidance.data = true;
//   }
//   else
//   {
//     enable_avoidance.data = false;
//   }
//   enable_avoidance_ = enable_avoidance.data;
//   enable_avoid_pub.publish(enable_avoidance);
// }

void avoidstatesubCallback(const msgs::Flag_Info& msg)
{
  double avoid_state_index_ = msg.Dspace_Flag03;
  // std::cout << "avoid_state_index_ : " << avoid_state_index_ << std::endl;
  std_msgs::Bool enable_avoidance;// = false;
  // std::cout << "force_disable_avoidance_ : " << force_disable_avoidance_ << std::endl;
  if (avoid_state_index_ == 1 && !force_disable_avoidance_)
  {
    enable_avoidance.data = true;
  }
  else
  {
    enable_avoidance.data = false;
  }
  enable_avoidance_ = enable_avoidance.data;
  enable_lane_change_by_obstacle_pub.publish(enable_avoidance);
}

void overshootorigdisCallback(const std_msgs::Float64& msg)
{
  project_dis = msg.data;
}

void laneeventCallback(const msgs::LaneEvent::ConstPtr& msg)
{
  lane_event_enable_overtake = true;
  if (msg->is_in_n10_0 || msg->is_in_0_70_incoming || msg->is_in_n40_n10_incoming)
  {
    lane_event_enable_overtake = false;
  }
  std::cout << "lane_event_enable_overtake : " << lane_event_enable_overtake << std::endl;
  if (disable_lane_event_)
  {
    lane_event_enable_overtake = true;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "decision_maker");
  ros::NodeHandle node;

  ros::param::get(ros::this_node::getName()+"/force_disable_avoidance", force_disable_avoidance_);
  ros::param::get(ros::this_node::getName()+"/disable_lane_event", disable_lane_event_);

  ros::Subscriber avoid_state_sub = node.subscribe("Flag_Info01", 1, avoidstatesubCallback);
  ros::Subscriber obstacle_dis_1_sub = node.subscribe("Geofence_original", 1, obsdisbaseCallback);
  ros::Subscriber veh_overshoot_orig_dis_sub = node.subscribe("veh_overshoot_orig_dis", 1, overshootorigdisCallback);
  ros::Subscriber lane_event_sub = node.subscribe("lane_event", 1, laneeventCallback);
  ros::Subscriber lanechangeready_sub = node.subscribe("/planning/scenario_planning/lane_driving/lane_change_ready", 1, lanechangereadyCallback);
  enable_lane_change_by_obstacle_pub = node.advertise<std_msgs::Bool>("/planning/scenario_planning/lane_driving/obstacle_lane_change_approval", 10, true);
  enable_avoid_pub = node.advertise<std_msgs::Bool>("/planning/scenario_planning/lane_driving/motion_planning/obstacle_avoidance_planner/enable_avoidance", 10, true);
  overtake_over_pub = node.advertise<std_msgs::Int32>("avoidpath_reach_goal", 10, true);

  // ros::Rate loop_rate(10);
  // while (ros::ok())
  // { 
  //   ros::spinOnce();
  //   loop_rate.sleep();   
  // }

  ros::spin();
  return 0;
};
