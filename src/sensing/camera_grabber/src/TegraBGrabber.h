#ifndef __TEGRA_B_GRABBER__
#define __TEGRA_B_GRABBER__

#include "CameraGrabber.h"

namespace SensingSubSystem
{
class TegraBGrabber
{
public:
  TegraBGrabber();
  ~TegraBGrabber();
  void initializeModules();
  bool runPerception();

protected:
  void InitParameters();
  std::vector<int> cam_ids_;
  std::vector<cv::Mat> canvas;

  BufferConfig camera_buffer_;

private:
  // Sensing Modules
  MultiGMSLCameraGrabber* grabber;
  std::vector<Npp8u*> npp8u_ptrs_;
  std::vector<Npp8u*> npp8u_ptrs_distorted_;
  NPPResizer resizer_;
  NPPRemapper remapper_;

  // ROS publisher
  ros::NodeHandle n;
  RosImagePubSub ros_image;
};
}

#endif
