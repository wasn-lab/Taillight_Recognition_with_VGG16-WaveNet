/// standard
#include <iostream>
#include <string>
#include <boost/asio/ip/host_name.hpp>
/// ros
#include "ros/ros.h"

int main(int argc, char** argv)
{
  std::cout << "===== ros_time_test. =====" << std::endl;
  std::string host_name = boost::asio::ip::host_name();
  int current = 0;
  std::size_t pos = host_name.find("-", current);
  host_name = host_name.substr(current, pos-current);
  std::cout << "host_name: " <<  host_name << std::endl;
  
  ros::init(argc, argv, "ros_time_test_" + host_name);
  ros::NodeHandle nh;

  ros::Rate loop_rate(30)
  while(ros::ok())
  {
    ros::Time t1 = ros::Time::now();
    std::cout << host_name << " pc - time now: " << t1.sec << "." << t1.nsec << std::endl;
    loop_rate.sleep();
  }
  std::cout << "===== ros_time_test shutdown. =====" << std::endl;
  return 0;
}
