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

#if CAR_MODEL_IS_B1
  // TODO: fill in the correct camera id.
  const std::vector<int> cam_ids_{ camera::id::left_60,  camera::id::front_60,  camera::id::right_60,
                                   camera::id::left_120, camera::id::front_120, camera::id::right_120 };
#elif CAR_MODEL_IS_HINO
  const std::vector<int> cam_ids_{camera::id::left_120, camera::id::front_120, camera::id::right_120};
#else
#error "car model is not well defined"
#endif

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
