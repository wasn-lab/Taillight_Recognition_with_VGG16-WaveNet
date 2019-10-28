#ifndef __TEGRA_A_GRABBER__
#define __TEGRA_A_GRABBER__

#include "CameraGrabber.h"

namespace SensingSubSystem
{
class TegraAGrabber
{
public:
  TegraAGrabber();
  ~TegraAGrabber();
  void initializeModules();
  bool runPerception();

protected:
  void InitParameters();
  std::vector<size_t> image_num;
  std::vector<cv::Mat> canvas;

  BufferConfig camera_buffer_;
  DisplayConfig display_;

private:
  // Sensing Modules
  MultiGMSLCameraGrabber* grabber;
  std::vector<Npp8u*> npp8u_ptrs_;
  NPPResizer resizer_;

  // ROS publisher
  ros::NodeHandle n;
  RosImagePubSub ros_image;
};
}

#endif
