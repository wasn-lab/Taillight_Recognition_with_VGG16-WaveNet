#include <openpose_ros_io.h>

using namespace openpose_ros;

OpenPoseROSIO::OpenPoseROSIO(OpenPose& openPose) : nh_("/openpose_ros_node"), it_(nh_)
{
  // Subscribe to input video feed and publish human lists as output
  std::string image_topic;
  std::string output_topic;
  std::string input_image_transport_type;

  nh_.param("image_topic", image_topic, std::string("/camera/image_raw"));
  nh_.param("input_image_transport_type", input_image_transport_type, std::string("raw"));
  nh_.param("output_topic", output_topic, std::string("/openpose_ros/human_list"));
  nh_.param("display_output", display_output_flag_, true);
  nh_.param("print_keypoints", print_keypoints_flag_, false);
  nh_.param("save_original_video", save_original_video_flag_, false);
  nh_.param("save_openpose_video", save_openpose_video_flag_, false);
  nh_.param("original_video_file_name", original_video_file_name_, std::string(""));
  nh_.param("openpose_video_file_name", openpose_video_file_name_, std::string(""));
  nh_.param("video_fps", video_fps_, 10);
}

void OpenPoseROSIO::stop()
{
  if (save_original_video_flag_)
  {
    original_video_writer_.release();
  }
  if (save_openpose_video_flag_)
  {
    openpose_video_writer_.release();
  }
}
