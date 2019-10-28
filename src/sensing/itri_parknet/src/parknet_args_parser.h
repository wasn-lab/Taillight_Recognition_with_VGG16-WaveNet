#ifndef __PARKNET_ARGS_PARSER_H__
#define __PARKNET_ARGS_PARSER_H__

#include <string>

namespace parknet
{
// Getters
std::string get_camera_port();
double get_yolo_score_threshold();
double get_yolo_nms_threshold();
double get_front_120_sx_compensation();
double get_front_120_sy_compensation();
double get_right_120_sx_compensation();
double get_right_120_sy_compensation();
double get_left_120_sx_compensation();
double get_left_120_sy_compensation();
bool should_save_detected_image();
bool should_save_voc();
bool should_save_preprocessed_image();
bool should_save_raw_image();
bool should_display_gui();
bool in_barrier_synchroniztion_mode();
bool in_profiling_mode();
bool should_publish_detection_results();
bool should_fuse_side_cameras();
bool use_tiny_yolov3();
};  // namespace parknet

#endif  //__PARKNET_ARGS_PARSER_H__
