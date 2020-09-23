#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "save_route.h"

//fstream file;
geometry_msgs::TransformStamped transformStamped;
std::string map_frame_ = "map";

bool transformPose(
  const geometry_msgs::PoseStamped & input_pose, geometry_msgs::PoseStamped * output_pose,
  const std::string target_frame)
{
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tfListener(tf_buffer_);
  sleep(1);
  geometry_msgs::TransformStamped transform;
  try {
    transform = tf_buffer_.lookupTransform(target_frame, input_pose.header.frame_id, ros::Time(0));
    tf2::doTransform(input_pose, *output_pose, transform);
    return true;
  } catch (tf2::TransformException & ex) {
    ROS_WARN("%s", ex.what());
    return false;
  }
}

SaveRoute::SaveRoute():node("~")
{
  goal_subscriber = node.subscribe("/move_base_simple/goal", 1, &SaveRoute::SaveGoalPose, this);
  checkpoint_subscriber = node.subscribe("/checkpoint", 1, &SaveRoute::SaveCheckPoint, this);
}

SaveRoute::~SaveRoute()
{
  file.close();
}

void SaveRoute::SaveGoalPose(const geometry_msgs::PoseStampedConstPtr& goal_msg_ptr)
{
  geometry_msgs::PoseStamped goal_msg;

  if (!transformPose(*goal_msg_ptr, &goal_msg, map_frame_)) {
    ROS_ERROR("Failed to get goal pose in map frame. Aborting mission planning");
    return;
  }
  
  fpname = ros::package::getPath("save_route");
  fpname_s = fpname + "/data/test.txt";
  
  file.open(fpname_s.c_str(),ios::out);
  if(file.fail())
  {
    cout << "could not open the file--" << endl;
  }
  else
  {
    file.clear();

    file << goal_msg.pose.position.x << "," << goal_msg.pose.position.y << "," << goal_msg.pose.position.z << "," 
    << goal_msg.pose.orientation.x << ","<< goal_msg.pose.orientation.y << "," 
    << goal_msg.pose.orientation.z << "," << goal_msg.pose.orientation.w ;
    
    cout << goal_msg.pose.position.x << "," << goal_msg.pose.position.y << "," << goal_msg.pose.position.z << "," 
    << goal_msg.pose.orientation.x << ","<< goal_msg.pose.orientation.y << "," 
    << goal_msg.pose.orientation.z << "," << goal_msg.pose.orientation.w ;
  }
}

void SaveRoute::SaveCheckPoint(const geometry_msgs::PoseStampedConstPtr & checkpoint_msg_ptr)
{
  geometry_msgs::PoseStamped checkpoint_msg;
  if (!transformPose(*checkpoint_msg_ptr, &checkpoint_msg, map_frame_)) {
    ROS_ERROR("Failed to get goal pose in map frame. Aborting mission planning");
    return;
  }

  if(file.fail())
  {
    cout << "could not open the file" << endl;
  }
  else
  {
    file << "\n" << checkpoint_msg.pose.position.x << "," << checkpoint_msg.pose.position.y << "," << checkpoint_msg.pose.position.z << "," 
    << checkpoint_msg.pose.orientation.x << ","<< checkpoint_msg.pose.orientation.y << "," 
    << checkpoint_msg.pose.orientation.z << "," << checkpoint_msg.pose.orientation.w;

    cout << "\n" << checkpoint_msg.pose.position.x << "," << checkpoint_msg.pose.position.y << "," << checkpoint_msg.pose.position.z << "," 
    << checkpoint_msg.pose.orientation.x << ","<< checkpoint_msg.pose.orientation.y << "," 
    << checkpoint_msg.pose.orientation.z << "," << checkpoint_msg.pose.orientation.w ;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "save_route");
  
  SaveRoute route;
  ros::spin();
  return 0;
};