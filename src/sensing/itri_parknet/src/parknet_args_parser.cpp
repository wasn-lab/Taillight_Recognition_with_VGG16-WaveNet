#include <cstdlib>
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include "parknet_args_parser.h"
#include "parknet.h"

namespace parknet
{
DEFINE_string(camera_port, "b", "Receive images from the specified camera port");
DEFINE_bool(fuse_side_cameras, false, "Fuse the side camera detection results to form a parking slot.");
DEFINE_bool(barrier_synchronization, true, "Publish messages only when all the cameras finish detection");
DEFINE_bool(profiling, false, "Gracefully exit after running for 1 minute.");
DEFINE_double(score_thresh, 0.25, "Probability threshold for detected objects");
DEFINE_double(parknet_nms_thresh, 0.35, "IOU threshold for bounding box candidates");
DEFINE_double(front_120_sx_compensation, 0, "Spatial x-axis compensation for distance estimation in front_120 camera");
DEFINE_double(right_120_sx_compensation, 0, "Spatial x-axis compensation for distance estimation in right_120 camera");
DEFINE_double(left_120_sx_compensation, 0, "Spatial x-axis compensation for distance estimation in left_120 camera");
DEFINE_double(front_120_sy_compensation, 0, "Spatial y-axis compensation for distance estimation in front_120 camera");
DEFINE_double(right_120_sy_compensation, 0, "Spatial y-axis compensation for distance estimation in right_120 camera");
DEFINE_double(left_120_sy_compensation, 0, "Spatial y-axis compensation for distance estimation in left_120 camera");
DEFINE_bool(save_raw_image, false, "Save each raw image into disk.");
DEFINE_bool(save_detected_image, false, "Save each detected image into disk.");
DEFINE_bool(save_voc, false, "Save each detection results in VOC format.");
DEFINE_bool(save_preprocessed_image, false, "Save each preprocessed image into disk.");
DEFINE_bool(display_gui, false, "Use GUI to display detection results");
DEFINE_bool(publish_detection_results, true, "Publish ros messages about the detection result in cv::Mat.");
DEFINE_bool(use_tiny_yolov3, false, "Use Tiny yolov3 (faster but lower accuracy)");

std::string get_camera_port()
{
  return std::string(FLAGS_camera_port);
}

double get_yolo_score_threshold()
{
  return FLAGS_score_thresh;
}

double get_yolo_nms_threshold()
{
  return FLAGS_parknet_nms_thresh;
}

double get_front_120_sx_compensation()
{
  return FLAGS_front_120_sx_compensation;
}

double get_front_120_sy_compensation()
{
  return FLAGS_front_120_sy_compensation;
}

double get_right_120_sx_compensation()
{
  return FLAGS_right_120_sx_compensation;
}

double get_right_120_sy_compensation()
{
  return FLAGS_right_120_sy_compensation;
}

double get_left_120_sx_compensation()
{
  return FLAGS_left_120_sx_compensation;
}

double get_left_120_sy_compensation()
{
  return FLAGS_left_120_sy_compensation;
}

bool should_save_raw_image()
{
  return FLAGS_save_raw_image;
}

bool should_save_detected_image()
{
  return FLAGS_save_detected_image;
}

bool should_save_voc()
{
  return FLAGS_save_voc;
}

bool should_save_preprocessed_image()
{
  return FLAGS_save_preprocessed_image;
}

bool should_display_gui()
{
  const char* display_guienv = std::getenv("FLAGS_display_gui");
  if (display_guienv)
  {
    std::string s = display_guienv;
    return ((!s.compare("true")) || (!s.compare("1")));
  }
  else
  {
    return FLAGS_display_gui;
  }
}

bool in_barrier_synchroniztion_mode()
{
  return FLAGS_barrier_synchronization;
}

bool in_profiling_mode()
{
  return FLAGS_profiling;
}

bool should_publish_detection_results()
{
  return FLAGS_publish_detection_results;
}

bool should_fuse_side_cameras()
{
  return FLAGS_fuse_side_cameras;
}

bool use_tiny_yolov3()
{
  return FLAGS_use_tiny_yolov3;
}

};  // namespace parknet
