#ifndef __TEGRA_C_GRABBER__
#define __TEGRA_C_GRABBER__

#include <msgs/MotionVector.h>
#include <msgs/MotionVectorArray.h>
#include "mvextractor_standalone.h"
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
#if CAR_MODEL_IS_B1_V3 || CAR_MODEL_IS_C1 || CAR_MODEL_IS_C2 || CAR_MODEL_IS_C3 // Camera use Gstreamer

  const std::vector<int> cam_ids_{ camera::id::front_bottom_60,     camera::id::front_top_far_30,

                                   camera::id::front_top_close_120, camera::id::right_front_60,
                                   camera::id::right_back_60,       camera::id::left_front_60,
                                   camera::id::left_back_60,        camera::id::back_top_120 };
#else
#error "car model is not well defined"
#endif
  std::vector<cv::Mat> canvas;
  std::vector<cv::Mat> canvas_tmp;
  std::vector<context_t> ctx;
  std::vector<cv::Mat> cvBGR;
  std::vector<cv::Mat> cvYUV;
  std::vector<SafeQueue<cv::Mat>> cvBGR_queue;
  std::vector<std::future<int>> fuRes_enc;
  std::vector<uint32_t> debug_counter;
  std::vector<cv::Mat> cvMV;
  std::vector<msgs::MotionVectorArray> mv_msgs_array;

  BufferConfig camera_buffer_;
  DisplayConfig display_;

private:
  // Sensing Modules
  std::vector<Npp8u*> npp8u_ptrs_;
  NPPResizer resizer_;
  NPPRemapper remapper_;    
  bool resize_;
  bool motion_vector_;

  // ROS publisher
  ros::NodeHandle n;
  RosImagePubSub ros_image;
  ros::Time ros_time_;  

  // Gstream
  std::vector<cv::VideoCapture> video_capture_list;
};
}  // namespace SensingSubSystem

#endif
