#ifndef __ROS_PARAMS_PARSER_H__
#define __ROS_PARAMS_PARSER_H__

#include "tpp.h"

namespace tpp
{
class ROSParamsParser
{
public:
  ROSParamsParser(const ros::NodeHandle& nh) : nh_(nh)
  {
  }
  ~ROSParamsParser()
  {
  }

  void get_ros_param_bool(const std::string& param_name, bool& output);
  void get_ros_param_int(const std::string& param_name, int& output);
  void get_ros_param_uint(const std::string& param_name, unsigned int& output);
  void get_ros_param_double(const std::string& param_name, double& output);
  void get_ros_param_color(const std::string& param_name, std_msgs::ColorRGBA& output);

private:
  DISALLOW_COPY_AND_ASSIGN(ROSParamsParser);

  ros::NodeHandle nh_;
};
}  // namespace tpp

#endif  //__ROS_PARAMS_PARSER_H__