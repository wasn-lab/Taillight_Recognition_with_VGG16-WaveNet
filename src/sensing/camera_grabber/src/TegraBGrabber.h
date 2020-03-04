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
  void initializeModules(bool do_resize);
  bool runPerception();

protected:
  void InitParameters();

#if CAR_MODEL_IS_B1
  // TODO: fill in the correct camera id.
  const std::vector<int> cam_ids_{ 
    camera::id::top_front_120,
    camera::id::top_right_front_120,
    camera::id::top_right_rear_120,
    camera::id::top_left_front_120,
    camera::id::top_left_rear_120,
    camera::id::top_rear_120
  };
#elif CAR_MODEL_IS_B1_V2
  const std::vector<int> cam_ids_{ 
    camera::id::front_top_close_120,
    camera::id::right_front_60,
    camera::id::right_back_60,
    camera::id::left_front_60,
    camera::id::left_back_60,
    camera::id::back_top_120
  };
#elif CAR_MODEL_IS_HINO
  const std::vector<int> cam_ids_{ camera::id::left_120, camera::id::front_120, camera::id::right_120 };
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
  NPPTransformer transformer_;
  bool _resize;

  // ROS publisher
  ros::NodeHandle n;
  RosImagePubSub ros_image;
};
} // namespace SensingSubSystem

#endif
