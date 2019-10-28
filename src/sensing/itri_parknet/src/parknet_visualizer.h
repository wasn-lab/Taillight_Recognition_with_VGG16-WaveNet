#ifndef __PARKNET_VISUALIZER_H__
#define __PARKNET_VISUALIZER_H__

#include <string>
#include <image_transport/image_transport.h>
#include "opencv2/core/mat.hpp"

class ParknetVisualizer
{
private:
  const std::string topic_name_;
  //  image_transport::Publisher publisher_;
  ros::Publisher publisher_;

public:
  // Methods
  ParknetVisualizer(const std::string& topic_name);
  ~ParknetVisualizer();
  int config_publisher(ros::NodeHandle& nh);
  int publish_cvmat(const cv::Mat& img);
  std::string get_topic_name() const;
};
#endif  // __PARKNET_VISUALIZER_H__
