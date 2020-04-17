#ifndef __TEGRA_A_GRABBER__
#define __TEGRA_A_GRABBER__

#include "CameraGrabber.h"
#include "camera_params.h"

namespace SensingSubSystem
{
class TegraAGrabber
{
public:
  TegraAGrabber();
  ~TegraAGrabber();
  void initializeModules(const bool do_resize, const bool do_crop);
  bool runPerception();

protected:
  void InitParameters();
#if CAR_MODEL_IS_B1
  // TODO: fill in the correct camera id.
  const std::vector<int> cam_ids_{ camera::id::right_60, camera::id::front_60, camera::id::left_60 };
#elif CAR_MODEL_IS_B1_V2 || CAR_MODEL_IS_OMNIBUS
  const std::vector<int> cam_ids_{ camera::id::front_bottom_60, camera::id::front_top_far_30, camera::id::front_bottom_60_crop };
#elif CAR_MODEL_IS_HINO
  const std::vector<int> cam_ids_{ camera::id::left_60, camera::id::front_60, camera::id::right_60,
                                   camera::id::left_30, camera::id::front_30, camera::id::right_30 };
#else
#error "car model is not well defined"
#endif
  std::vector<cv::Mat> canvas;

  BufferConfig camera_buffer_;
  DisplayConfig display_;

private:
  // Sensing Modules
  MultiGMSLCameraGrabber* grabber;
  std::vector<Npp8u*> npp8u_ptrs_;
  NPPResizer resizer_;
  int num_src_bytes_;
  bool resize_;
  bool crop_;

  // ROS publisher
  ros::NodeHandle n;
  RosImagePubSub ros_image;
};
}  // namespace SensingSubSystem

#endif
