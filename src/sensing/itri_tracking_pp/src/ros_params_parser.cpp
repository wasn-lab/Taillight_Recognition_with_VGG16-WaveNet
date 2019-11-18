#include "ros_params_parser.h"

namespace tpp
{
void ROSParamsParser::get_ros_param_bool(const std::string& param_name, bool& output)
{
  int x_int;
  get_ros_param_int(param_name, x_int);
  output = (bool)x_int;
}

void ROSParamsParser::get_ros_param_int(const std::string& param_name, int& output)
{
  if (ros::param::has(param_name))
  {
    nh_.param(param_name, output, (int)0);

    ROS_INFO("Got param '%s': %d", param_name.c_str(), output);
  }
  else
  {
    ROS_ERROR("Failed to get param '%s'", param_name.c_str());
  }
}

void ROSParamsParser::get_ros_param_uint(const std::string& param_name, unsigned int& output)
{
  if (ros::param::has(param_name))
  {
    int output_int = 0;
    nh_.param(param_name, output_int, (int)0);

    output = (output_int >= 0) ? (unsigned int)output_int : 0;

    ROS_INFO("Got param '%s': %u", param_name.c_str(), output);
  }
  else
  {
    ROS_ERROR("Failed to get param '%s'", param_name.c_str());
  }
}

void ROSParamsParser::get_ros_param_double(const std::string& param_name, double& output)
{
  if (ros::param::has(param_name))
  {
    nh_.param(param_name, output, (double)0.);

    ROS_INFO("Got param '%s': %f", param_name.c_str(), output);
  }
  else
  {
    ROS_ERROR("Failed to get param '%s'", param_name.c_str());
  }
}

void ROSParamsParser::get_ros_param_color(const std::string& param_name, std_msgs::ColorRGBA& output)
{
  std::vector<float> v;
  if (ros::param::has(param_name))
  {
    nh_.param(param_name, v, std::vector<float>());
    output.r = v[0];
    output.g = v[1];
    output.b = v[2];
    output.a = v[3];

    ROS_INFO("Got param '%s': R=%.2f G=%.2f B=%.2f A=%.2f", param_name.c_str(), output.r, output.g, output.b, output.a);
  }
  else
  {
    ROS_ERROR("Failed to get param '%s'", param_name.c_str());
  }
}
}  // namespace tpp