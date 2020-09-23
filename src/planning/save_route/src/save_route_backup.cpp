#include <ros/ros.h>
#include "save_route.h"

//fstream file;
geometry_msgs::TransformStamped transformStamped;

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
  
  if (goal_msg_ptr->header.frame_id == "map")
  {
    goal_msg = *goal_msg_ptr;
  }
  else
  {
    goal_msg.header.frame_id = "map";
    goal_msg.header.stamp = ros::Time::now();
    goal_msg.pose.position.x = goal_msg_ptr->pose.position.x + transformStamped.transform.translation.x;
    goal_msg.pose.position.y = goal_msg_ptr->pose.position.y + transformStamped.transform.translation.y;
    goal_msg.pose.position.z = goal_msg_ptr->pose.position.z + transformStamped.transform.translation.z;
    goal_msg.pose.orientation.x = goal_msg_ptr->pose.orientation.x;// + transformStamped.transform.rotation.x;
    goal_msg.pose.orientation.y = goal_msg_ptr->pose.orientation.y;// + transformStamped.transform.rotation.y;
    goal_msg.pose.orientation.z = goal_msg_ptr->pose.orientation.z;// + transformStamped.transform.rotation.z;
    goal_msg.pose.orientation.w = goal_msg_ptr->pose.orientation.w;// + transformStamped.transform.rotation.w;
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

  if (checkpoint_msg_ptr->header.frame_id == "map")
  {
    checkpoint_msg = *checkpoint_msg_ptr;
  }
  else
  {
    checkpoint_msg.header.frame_id = "map";
    checkpoint_msg.header.stamp = ros::Time::now();
    checkpoint_msg.pose.position.x = checkpoint_msg_ptr->pose.position.x + transformStamped.transform.translation.x;
    checkpoint_msg.pose.position.y = checkpoint_msg_ptr->pose.position.y + transformStamped.transform.translation.y;
    checkpoint_msg.pose.position.z = checkpoint_msg_ptr->pose.position.z + transformStamped.transform.translation.z;
    checkpoint_msg.pose.orientation.x = checkpoint_msg_ptr->pose.orientation.x;// + transformStamped.transform.rotation.x;
    checkpoint_msg.pose.orientation.y = checkpoint_msg_ptr->pose.orientation.y;// + transformStamped.transform.rotation.y;
    checkpoint_msg.pose.orientation.z = checkpoint_msg_ptr->pose.orientation.z;// + transformStamped.transform.rotation.z;
    checkpoint_msg.pose.orientation.w = checkpoint_msg_ptr->pose.orientation.w;// + transformStamped.transform.rotation.w;
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

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);
  while (ros::ok())
  {
    // geometry_msgs::TransformStamped transformStamped;
    try{
      transformStamped = tfBuffer.lookupTransform("map", "viewer", ros::Time(0));
    }
    catch (tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
      ros::Duration(1.0).sleep();
      continue;
    }
    cout << "transform initial ok !" << endl;
    cout << "transform x : " << transformStamped.transform.translation.x << ", " << "transform y : " << transformStamped.transform.translation.y << endl;
    break;
  }
  
  SaveRoute route;
  ros::spin();
  return 0;
};