#include <cv_bridge/cv_bridge.h>
#include "parknet_visualizer.h"

ParknetVisualizer::ParknetVisualizer(const std::string& topic_name) : topic_name_(topic_name)
{
}

ParknetVisualizer::~ParknetVisualizer()
{
  publisher_.shutdown();
}

int ParknetVisualizer::config_publisher(ros::NodeHandle& node_handle)
{
  //  image_transport::ImageTransport it(node_handle);
  publisher_ = node_handle.advertise<sensor_msgs::Image>(topic_name_, 1);
  return 0;
}

int ParknetVisualizer::publish_cvmat(const cv::Mat& in_img)
{
  publisher_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", in_img).toImageMsg());
  return 0;
}

std::string ParknetVisualizer::get_topic_name() const
{
  return topic_name_;
}
