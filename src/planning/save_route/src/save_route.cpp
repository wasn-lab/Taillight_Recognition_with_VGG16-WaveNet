#include <ros/ros.h>
#include "save_route.h"

//fstream file;

SaveRoute::SaveRoute():node("~")
{
  goal_subscriber = node.subscribe("/move_base_simple/goal", 1, &SaveRoute::SaveGoalPose, this);
  checkpoint_subscriber =node.subscribe("/checkpoint", 1, &SaveRoute::SaveCheckPoint, this);
}

SaveRoute::~SaveRoute()
{
    file.close();
}

void SaveRoute::SaveGoalPose(const geometry_msgs::PoseStampedConstPtr & goal_msg_ptr)
{
  geometry_msgs::PoseStamped goal_msg;
  goal_msg = *goal_msg_ptr;
  
  fpname = ros::package::getPath("save_route");
  fpname_s = fpname + "/data/test.txt";
  

  file.open(fpname_s.c_str(),ios::out);
  if(file.fail())
  {
    cout << "could not open the file" << endl;
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
  checkpoint_msg = *checkpoint_msg_ptr;

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

