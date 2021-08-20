#include <ros/ros.h> 
#include <serial/serial.h>  //ROS serial package 
#include <std_msgs/String.h> 
#include <std_msgs/Empty.h> 

int main(int argc, char **argv)
{
  ros::init(argc, argv, "talker_control");  

  ros::NodeHandle n;     

  ros::Publisher chatter_pub = n.advertise<std_msgs::String>("/control/plc_control", 1000); 

 
  ros::Rate loop_rate(10); 
  int count = 0;

  while (ros::ok())
  {
    std_msgs::String msg;
    char x,y;
    std::string ss = " ";

    printf("\nInput:");
    std::cin >> x;
    std::cin >> y;

    msg.data = x + ss + y;

    //ROS_INFO("You will publish %s", msg.data.c_str());   

    chatter_pub.publish(msg);

    ros::spinOnce();

    loop_rate.sleep();

  }
  return 0;
}