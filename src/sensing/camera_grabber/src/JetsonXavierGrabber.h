#ifndef __TEGRA_C_GRABBER__
#define __TEGRA_C_GRABBER__

#include "CameraGrabber.h"
#include "camera_params.h"

namespace SensingSubSystem
{
class JetsonXavierGrabber
{
public:
  JetsonXavierGrabber();
  ~JetsonXavierGrabber();
  bool initializeModulesGst(const bool do_resize);
  bool runPerceptionGst();

  // Gstreamer
  bool gst_pipeline_init(int video_index);

protected:
  void InitParameters();
#if CAR_MODEL_IS_B1
  // TODO: fill in the correct camera id.
  const std::vector<int> cam_ids_{ camera::id::right_60, camera::id::front_60, camera::id::left_60 };
#elif CAR_MODEL_IS_B1_V2
  const std::vector<int> cam_ids_{ camera::id::front_bottom_60, camera::id::front_top_far_30,
                                   camera::id::front_bottom_60_crop };
#elif CAR_MODEL_IS_B1_V3  // Camera use Gstreamer

  const std::vector<int> cam_ids_{ camera::id::front_bottom_60,     camera::id::front_top_far_30,

                                   camera::id::front_top_close_120, camera::id::right_front_60,
                                   camera::id::right_back_60,       camera::id::left_front_60,
                                   camera::id::left_back_60,        camera::id::back_top_120 };
#else
#error "car model is not well defined"
#endif
  std::vector<cv::Mat> canvas;

  BufferConfig camera_buffer_;
  DisplayConfig display_;

private:
  // Sensing Modules
  std::vector<Npp8u*> npp8u_ptrs_;
  NPPResizer resizer_;
  int num_src_bytes_;
  bool resize_;

  // ROS publisher
  ros::NodeHandle n;
  RosImagePubSub ros_image;

  // Gstream
  std::vector<cv::VideoCapture> video_capture_list;
};
}  // namespace SensingSubSystem

#endif
