/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */

#ifndef __PARKNET_NODE_IMPL_H__
#define __PARKNET_NODE_IMPL_H__

#include "parknet.h"
#include "parknet_camera.h"
#include <atomic>
#include <fstream>
#include <thread>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <boost/shared_ptr.hpp>

#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

#include <cv_bridge/cv_bridge.h>

#include "msgs/ParkingSlotResult.h"
#include "msgs/ParkingSlot.h"
#include "msgs/MarkingPoint.h"
#include "nppdefs.h"

#include <rect_class_score.h>

#if USE(TENSORRT)
#include "trt_yolo3_detector.h"
#endif

#include "yolo3_detector.h"
extern "C" {
#undef __cplusplus
#include "box.h"
#include "image.h"
#include "network.h"
#include "detection_layer.h"
#include "parser.h"
#include "region_layer.h"
#include "utils.h"
#include "image.h"
#define __cplusplus
}

class Alignment;
class ParknetImageManager;
class ParknetVisualizer;
class CameraDistanceMapper;

class ParknetNodeImpl
{
  // ROS
  ros::Publisher publisher_results_;
  image_transport::Subscriber im_subscribers_[parknet::camera::num_cams_e];
  ros::NodeHandle node_handle_;
  std::unique_ptr<ParknetVisualizer> parknet_visualizer_[parknet::camera::num_cams_e];

// Detector backend
#if USE(TENSORRT)
  parknet::TRTYolo3Detector trt_yolo_detector_;
#elif USE(DARKNET)
  darknet::Yolo3Detector yolo_detector_;
#else
#error "Use either TensorRT or darknet as detection backend."
#endif

  // Images (Input)
  std::unique_ptr<ParknetImageManager> parknet_image_manager_ptr_;
  const bool should_display_gui_;
  const bool should_publish_detection_results_;
  const bool should_save_detected_image_;
  const bool should_save_voc_;
  const bool should_save_preprocessed_image_;
  const bool should_save_raw_image_;
  uint32_t prev_detected_image_checksum_[parknet::camera::num_cams_e];
  uint32_t prev_raw_image_checksum_[parknet::camera::num_cams_e];
  uint32_t prev_preprocessed_image_checksum_[parknet::camera::num_cams_e];

  // vars for threading
  std::condition_variable barrier_cond_var_;
  std::mutex barrier_mutex_;
  std::thread barrier_checking_thread_;
  bool unittest_mode_;

  // std::mutex dst_mutex_[parknet::num_cams];
  std::atomic_uint has_detection_results_;
  std::atomic_uint under_detection_;
  bool stop_threads_;

  // Integrate with main app
  bool main_app_mode_;

  // Distance mapping
  std::unique_ptr<CameraDistanceMapper> dist_mapper_ptr_[parknet::camera::num_cams_e];

  // private functions
  void __init_internal_vars();
  void save_detected_image_if_necessary(const int cam_id);
  void save_preprocessed_image_if_necessary(const int cam_id);
  void save_raw_image_if_necessary(const int cam_id);

  void image_callback_left_120(const sensor_msgs::ImageConstPtr& in_image_message);
  void image_callback_front_120(const sensor_msgs::ImageConstPtr& in_image_message);
  void image_callback_right_120(const sensor_msgs::ImageConstPtr& in_image_message);
  void display_detection_results_if_necessary(const int camd_id);
  void publish_detection_results_if_necessary(const int cam_id);
  void detect_parking_slots_by_cam_id(const sensor_msgs::ImageConstPtr& in_image_message, const int cam_id);
  void detect_parking_slots_by_cam_id_internal(const int cam_id);
  void barrier_checking();
  void init_darknet_detector();
  int gen_parking_slot_result(msgs::ParkingSlotResult&);
  void publish_parking_slot_result();
  void gracefully_stop_threads();
  void visualize_if_necessary();
  void on_new_detection(const int cam_id);
  void on_new_image_arrival(const int cam_id);
  bool has_new_detection(const int cam_id) const;
  bool image_is_under_detection(const int cam_id) const;
  void reset_detection_result_by_cam_id(const int cam_id);
  void reset_all_detection_results();
  bool can_goto_next_round_of_detection() const;
  bool in_main_app_mode() const;
  void copy_and_preprocess_raw_image(const sensor_msgs::ImageConstPtr& in_image_message, const int);
  void copy_and_preprocess_raw_image(const cv::Mat& in_image, const int);
  void copy_and_preprocess_raw_image(const Npp8u* npp8u_ptr, const int cam_id, const int num_bytes);
  bool convert_side_corners_to_parking_slot_result(msgs::ParkingSlotResult* pslot_result);

public:
  ParknetNodeImpl();
  ~ParknetNodeImpl();
  void subscribe_and_advertise_topics(ros::NodeHandle& node_handle);
  void subscribe_and_advertise_topics();
  void set_main_app_mode(bool mode);
  void set_unittest_mode(bool mode);
  const std::vector<RectClassScore<float> > get_detection(const int cam_id) const;
  int on_init();
  int on_inference(std::vector<cv::Mat>& frames);
  int on_inference(std::vector<Npp8u*>& npp8u_ptrs_cuda, const int num_bytes);
  int on_release();
  void run(int argc, char* argv[]);
};

#endif  // __PARKNET_NODE_IMPL_H__
