/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */

#include <thread>
#include <condition_variable>
#include <fstream>
#include "glog/logging.h"
#include "pcl_ros/point_cloud.h"

#include "parknet.h"
#include "parknet_node.h"
#include "parknet_node_impl.h"
#include "parknet_args_parser.h"
#include "parknet_camera.h"
#include "parknet_cv_colors.h"
#include "parknet_subscription.h"
#include "parknet_image_utils.h"
#include "parknet_image_manager.h"
#include "parknet_logging.h"
#include "parknet_advertise_utils.h"
#include "parknet_visualizer.h"
#include "npp_wrapper.h"
#include "camera_params.h"
#include "camera_utils.h"
#include "camera_distance_mapper.h"

static bool done_with_profiling()
{
  static int num_loop = 0;
  if (!parknet::in_profiling_mode())
  {
    return false;
  }
  num_loop++;
  return bool(num_loop > 60 * parknet::expected_fps);
}

static std::string get_timestamp_sn()
{
  auto now = ros::Time::now();
  return std::to_string(now.sec) + "_" + std::to_string(now.nsec);
}

ParknetNodeImpl::ParknetNodeImpl()
  : should_display_gui_(parknet::should_display_gui())
  , should_publish_detection_results_(parknet::should_publish_detection_results())
  , should_save_detected_image_(parknet::should_save_detected_image())
  , should_save_voc_(parknet::should_save_voc())
  , should_save_preprocessed_image_(parknet::should_save_preprocessed_image())
  , should_save_raw_image_(parknet::should_save_raw_image())
{
}

void ParknetNodeImpl::__init_internal_vars()
{
  init_darknet_detector();
  stop_threads_ = false;
  has_detection_results_ = 0;
  under_detection_ = 0;
  for (int i = 0; i < parknet::camera::num_cams_e; i++)
  {
    parknet_visualizer_[i].reset(new ParknetVisualizer("/PSOutput/" + parknet::camera_names[i]));
    dist_mapper_ptr_[i].reset(new CameraDistanceMapper(parknet::camera_id_mapping[i]));
  }
  parknet_image_manager_ptr_.reset(new ParknetImageManager());
  set_main_app_mode(false);
}

ParknetNodeImpl::~ParknetNodeImpl()
{
  gracefully_stop_threads();
  barrier_checking_thread_.join();
}

void ParknetNodeImpl::gracefully_stop_threads()
{
  stop_threads_ = true;
  barrier_cond_var_.notify_all();
}

void ParknetNodeImpl::display_detection_results_if_necessary(const int cam_id)
{
  if (!should_display_gui_)
  {
    return;
  }
  cv::Mat annonated_image;
  parknet_image_manager_ptr_->get_annotated_image(cam_id, dist_mapper_ptr_[cam_id].get(), annonated_image);
  if ((annonated_image.rows <= 0) || (annonated_image.cols <= 0))
  {
    return;
  }

  VLOG(2) << parknet::camera_names[cam_id] << ": display image " << &annonated_image;
  cv::namedWindow(parknet::camera_names[cam_id], cv::WINDOW_AUTOSIZE);
  cv::imshow(parknet::camera_names[cam_id], annonated_image);
  cv::waitKey(3);
  annonated_image.release();
}

void ParknetNodeImpl::publish_detection_results_if_necessary(const int cam_id)
{
  if (!should_publish_detection_results_)
  {
    return;
  }
  cv::Mat annonated_image;
  parknet_image_manager_ptr_->get_annotated_image(cam_id, dist_mapper_ptr_[cam_id].get(), annonated_image);
  if ((annonated_image.rows <= 0) || (annonated_image.cols <= 0))
  {
    return;
  }
  parknet_visualizer_[cam_id]->publish_cvmat(annonated_image);
  annonated_image.release();
}

void ParknetNodeImpl::save_detected_image_if_necessary(const int cam_id)
{
  if (!should_save_detected_image_)
  {
    return;
  }

  cv::Mat image;
  parknet_image_manager_ptr_->get_annotated_image(cam_id, dist_mapper_ptr_[cam_id].get(), image);
  if (image.empty() || camera::is_black_image(image))
  {
    return;
  }

  auto checksum = camera::calc_cvmat_checksum(image);
  if (checksum == prev_detected_image_checksum_[cam_id])
  {
    return;
  }
  std::string filename = parknet::camera_names[cam_id] + "_detected_" + get_timestamp_sn() + ".jpg";
  cv::imwrite(filename, image);
  LOG(INFO) << "Save " << filename;
  prev_detected_image_checksum_[cam_id] = checksum;
}

void ParknetNodeImpl::save_raw_image_if_necessary(const int cam_id)
{
  if (!should_save_raw_image_)
  {
    return;
  }
  cv::Mat image;
  parknet_image_manager_ptr_->get_raw_image(cam_id, image);
  if (image.empty() || camera::is_black_image(image))
  {
    LOG(INFO) << "empty or black raw images -- No save it to disk.";
    return;
  }

  auto checksum = camera::calc_cvmat_checksum(image);
  if (checksum == prev_raw_image_checksum_[cam_id])
  {
    return;
  }

  std::string filename = parknet::camera_names[cam_id] + "_raw_" + get_timestamp_sn() + ".jpg";
  cv::imwrite(filename, image);
  LOG(INFO) << "Save to " << filename;
  prev_raw_image_checksum_[cam_id] = checksum;
}

void ParknetNodeImpl::save_preprocessed_image_if_necessary(const int cam_id)
{
  if ((!should_save_preprocessed_image_) && (!should_save_voc_))
  {
    return;
  }

  cv::Mat image;
  parknet_image_manager_ptr_->get_preprocessed_image(cam_id, image);
  if (image.empty() || camera::is_black_image(image))
  {
    return;
  }

  auto checksum = camera::calc_cvmat_checksum(image);
  if (checksum == prev_preprocessed_image_checksum_[cam_id])
  {
    return;
  }

  std::string filename = parknet::camera_names[cam_id] + "_preprocessed_" + get_timestamp_sn() + ".jpg";

  const auto detected_objs = parknet_image_manager_ptr_->get_detection(cam_id);
  if (should_save_voc_ && (detected_objs.size() > 0))
  {
    const auto image_ratio = camera::image_ratio_on_yolov3;
    std::string voc_filename = parknet::camera_names[cam_id] + "_preprocessed_" + get_timestamp_sn() + ".txt";
    std::ofstream fp(voc_filename);
    assert(fp);
    for (const auto& obj : detected_objs)
    {
      // x, y are center of the detection box
      double x = (obj.x + (obj.w / 2) - camera::left_border) / image_ratio;
      x = x / camera::raw_image_width;
      double y = (obj.y + (obj.h / 2) - camera::top_border) / image_ratio;
      y = y / camera::raw_image_height;
      const double w = (obj.w / image_ratio) / camera::raw_image_width;
      const double h = (obj.h / image_ratio) / camera::raw_image_height;
      fp << obj.class_type - 1 << " " << x << " " << y << " " << w << " " << h << "\n";
    }
    LOG(INFO) << "Save " << voc_filename;
    cv::imwrite(filename, image);
    LOG(INFO) << "Save " << filename;
  }
  else if (should_save_preprocessed_image_)
  {
    cv::imwrite(filename, image);
    LOG(INFO) << "Save " << filename;
  }

  prev_preprocessed_image_checksum_[cam_id] = checksum;
}

void ParknetNodeImpl::image_callback_left_120(const sensor_msgs::ImageConstPtr& in_image_message)
{
  const int cam_id = parknet::camera::left_120_e;
  VLOG_IF(1, cam_id == 0) << parknet::camera_names[cam_id] << ": start image_callback";
  if (image_is_under_detection(cam_id))
  {
    LOG_EVERY_N(INFO, 60) << parknet::camera_names[cam_id] << ": skip frame before consuming detection results";
    barrier_cond_var_.notify_one();
    return;
  }
  on_new_image_arrival(cam_id);
  std::thread t(&ParknetNodeImpl::detect_parking_slots_by_cam_id, this, in_image_message, cam_id);
  t.detach();
}

void ParknetNodeImpl::image_callback_front_120(const sensor_msgs::ImageConstPtr& in_image_message)
{
  const int cam_id = parknet::camera::front_120_e;
  if (image_is_under_detection(cam_id))
  {
    LOG_EVERY_N(INFO, 60) << parknet::camera_names[cam_id] << ": skip frame before consuming detection results";
    barrier_cond_var_.notify_one();
    return;
  }
  on_new_image_arrival(cam_id);
  std::thread t(&ParknetNodeImpl::detect_parking_slots_by_cam_id, this, in_image_message, cam_id);
  t.detach();
}

void ParknetNodeImpl::image_callback_right_120(const sensor_msgs::ImageConstPtr& in_image_message)
{
  const int cam_id = parknet::camera::right_120_e;
  if (image_is_under_detection(cam_id))
  {
    LOG_EVERY_N(INFO, 60) << parknet::camera_names[cam_id] << ": skip frame before consuming detection results";
    barrier_cond_var_.notify_one();
    return;
  }
  on_new_image_arrival(cam_id);

  std::thread t(&ParknetNodeImpl::detect_parking_slots_by_cam_id, this, in_image_message, cam_id);
  t.detach();
}

void ParknetNodeImpl::copy_and_preprocess_raw_image(const sensor_msgs::ImageConstPtr& in_image_message,
                                                    const int cam_id)
{
  TIME_IT_BEGIN(preprocess);
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(in_image_message, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    LOG(ERROR) << "cv_bridge exception: " << e.what();
    return;
  }
  copy_and_preprocess_raw_image(cv_ptr->image, cam_id);
  if (parknet::in_profiling_mode())
  {
    TIME_IT_END(preprocess);
  }
}

void ParknetNodeImpl::copy_and_preprocess_raw_image(const cv::Mat& in_image, const int cam_id)
{
  assert(in_image.channels() == 3);
  parknet_image_manager_ptr_->set_raw_image(in_image, cam_id);
}

void ParknetNodeImpl::copy_and_preprocess_raw_image(const Npp8u* npp8u_ptr, const int cam_id, const int num_bytes)
{
  parknet_image_manager_ptr_->set_raw_image(npp8u_ptr, cam_id, num_bytes);
}

void ParknetNodeImpl::visualize_if_necessary()
{
  for (int i = 0; i < parknet::camera::num_cams_e; i++)
  {
    const int cam_id = i;
    save_preprocessed_image_if_necessary(cam_id);
    display_detection_results_if_necessary(cam_id);
    publish_detection_results_if_necessary(cam_id);
    save_detected_image_if_necessary(cam_id);
    save_raw_image_if_necessary(cam_id);
  }
}

void ParknetNodeImpl::detect_parking_slots_by_cam_id(const sensor_msgs::ImageConstPtr& in_image_message,
                                                     const int cam_id)
{
  VLOG(2) << parknet::camera_names[cam_id] << ": detect parking lot in an image.";
  copy_and_preprocess_raw_image(in_image_message, cam_id);
  detect_parking_slots_by_cam_id_internal(cam_id);
}

void ParknetNodeImpl::detect_parking_slots_by_cam_id_internal(const int cam_id)
{
  auto cur_begin = std::chrono::high_resolution_clock::now();
  std::vector<RectClassScore<float> > detection;

#if USE(TENSORRT)
  trt_yolo_detector_.detect(detection, parknet_image_manager_ptr_->get_blob(cam_id), cam_id);
#elif USE(DARKNET)
  cv::Mat preprocessed_image;
  parknet_image_manager_ptr_->get_preprocessed_image(cam_id, preprocessed_image);
  auto darknet_image = parknet::convert_to_darknet_image(preprocessed_image);
  yolo_detector_.detect(detection, darknet_image);
  free(darknet_image.data);
#else
#error "Must use one detector."
#endif
  parknet_image_manager_ptr_->set_detection(detection, cam_id);

  for (size_t i = 0; i < detection.size(); i++)
  {
    VLOG(2) << parknet::camera_names[cam_id] << ": x=" << detection[i].x << " y=" << detection[i].y
            << " w=" << detection[i].w << " h=" << detection[i].h << " score=" << detection[i].score;
  }

  auto cur_end = std::chrono::high_resolution_clock::now();
  auto detection_time_in_ms = parknet::calc_duration_in_millisecond(cur_begin, cur_end);

  VLOG(2) << parknet::camera_names[cam_id] << ": detection done in " << detection_time_in_ms << "ms";

  on_new_detection(cam_id);
  assert(has_new_detection(cam_id));
  barrier_cond_var_.notify_one();
}

void ParknetNodeImpl::barrier_checking()
{
  std::unique_lock<std::mutex> ulock(barrier_mutex_);
  while (ros::ok() && (!stop_threads_) && (!unittest_mode_))
  {
    if (can_goto_next_round_of_detection())
    {
      publish_parking_slot_result();
    }
    barrier_cond_var_.wait(ulock);
  }
}

void ParknetNodeImpl::subscribe_and_advertise_topics()
{
  subscribe_and_advertise_topics(node_handle_);
}

void ParknetNodeImpl::subscribe_and_advertise_topics(ros::NodeHandle& node_handle)
{
  image_transport::ImageTransport it(node_handle);

  publisher_results_ = node_handle.advertise<msgs::ParkingSlotResult>("/parking_slot_result", 1);

  if (should_publish_detection_results_)
  {
    for (int i = 0; i < parknet::camera::num_cams_e; i++)
    {
      parknet_visualizer_[i]->config_publisher(node_handle);
      LOG(INFO) << "publish detection result in " << parknet_visualizer_[i]->get_topic_name();
    }
  }

  if (!in_main_app_mode())
  {
    auto port = parknet::get_camera_port();
    if (port.length() == 0)
    {
      // grabber publish images in 608x608
      im_subscribers_[parknet::camera::left_120_e] =
          it.subscribe(parknet::get_subscribed_image_topic(parknet::camera::left_120_e), 2,
                       &ParknetNodeImpl::image_callback_left_120, this);
      im_subscribers_[parknet::camera::front_120_e] =
          it.subscribe(parknet::get_subscribed_image_topic(parknet::camera::front_120_e), 2,
                       &ParknetNodeImpl::image_callback_front_120, this);
      im_subscribers_[parknet::camera::right_120_e] =
          it.subscribe(parknet::get_subscribed_image_topic(parknet::camera::right_120_e), 2,
                       &ParknetNodeImpl::image_callback_right_120, this);
    }
    else
    {
      // grabber publish images in 1920x1208, to be deprecated.
      im_subscribers_[parknet::camera::left_120_e] =
          it.subscribe(parknet::get_subscribed_image_topic(parknet::camera::left_120_e), 2,
                       &ParknetNodeImpl::image_callback_left_120, this, image_transport::TransportHints("compressed"));
      im_subscribers_[parknet::camera::front_120_e] =
          it.subscribe(parknet::get_subscribed_image_topic(parknet::camera::front_120_e), 2,
                       &ParknetNodeImpl::image_callback_front_120, this, image_transport::TransportHints("compressed"));
      im_subscribers_[parknet::camera::right_120_e] =
          it.subscribe(parknet::get_subscribed_image_topic(parknet::camera::right_120_e), 2,
                       &ParknetNodeImpl::image_callback_right_120, this, image_transport::TransportHints("compressed"));
    }
    for (int i = 0; i < parknet::camera::num_cams_e; i++)
    {
      const int cam_id = i;
      LOG(INFO) << "Subscribing to " << parknet::get_subscribed_image_topic(cam_id);
    }
  }
}

void ParknetNodeImpl::init_darknet_detector()
{
#if USE(DARKNET)
  // load cfg
  std::string network_definition_file;
  std::string pretrained_model_file, names_file;

  network_definition_file = parknet::get_yolo_cfg_file();
  pretrained_model_file = parknet::get_yolo_weights_file();
  double score_thresh = parknet::get_yolo_score_threshold();
  double nms_thresh = parknet::get_yolo_nms_threshold();

  LOG(INFO) << "Initializing Yolo on Darknet...";
  yolo_detector_.load(network_definition_file, pretrained_model_file, score_thresh, nms_thresh);
#endif
}

/**
 * Try to form a parking slot from side corners.
 **/
bool ParknetNodeImpl::convert_side_corners_to_parking_slot_result(msgs::ParkingSlotResult* pslot_result)
{
  assert(pslot_result->parking_slots.size() == 0);
  if (!parknet::should_fuse_side_cameras())
  {
    return false;
  }
  auto left_corners = parknet_image_manager_ptr_->get_detection(parknet::camera::left_120_e);
  auto right_corners = parknet_image_manager_ptr_->get_detection(parknet::camera::right_120_e);
  if ((left_corners.size() != 1) || (right_corners.size() != 1))
  {
    return false;
  }
  VLOG(2) << __FUNCTION__;
  msgs::MarkingPoint mps[2];
  msgs::ParkingSlot pslot;
  mps[0] =
      parknet::convert_corner_to_marking_point(left_corners[0], *dist_mapper_ptr_[parknet::camera::left_120_e].get());
  if (parknet::is_marking_point_near_image_border(parknet::camera::left_120_e, mps[0]))
  {
    return false;
  }

  mps[1] =
      parknet::convert_corner_to_marking_point(right_corners[0], *dist_mapper_ptr_[parknet::camera::right_120_e].get());
  if (parknet::is_marking_point_near_image_border(parknet::camera::right_120_e, mps[1]))
  {
    return false;
  }
  VLOG(1) << "left corner: " << left_corners[0].toString();
  VLOG(1) << "right corner: " << right_corners[0].toString();
  VLOG(1) << "left/right marking points : (" << mps[0].x << ", " << mps[0].y << "), (" << mps[1].x << ", " << mps[1].y
          << ")";

  auto succ = parknet::convert_2_marking_points_to_parking_slot(mps, &pslot);
  if (!succ)
  {
    return false;
  }
  parknet::print_parking_slot(pslot, "side corners form a parking slot: ");
  pslot_result->parking_slots.emplace_back(pslot);
  return true;
}

int ParknetNodeImpl::gen_parking_slot_result(msgs::ParkingSlotResult& out)
{
  msgs::ParkingSlotResult idv[parknet::camera::num_cams_e];
  size_t num_parking_slots = 0;
  for (int i = 0; i < parknet::camera::num_cams_e; i++)
  {
    const int cam_id = i;
    auto corners = parknet_image_manager_ptr_->get_detection(cam_id);
    if (corners.size() >= 5)
    {
      idv[i] =
          parknet::convert_5_or_more_corners_to_parking_slot_result(cam_id, corners, *dist_mapper_ptr_[cam_id].get());
    }
    else if (corners.size() == 4)
    {
      idv[i] = parknet::convert_4_corners_to_parking_slot_result(cam_id, corners, *dist_mapper_ptr_[cam_id].get());
    }
    else if (corners.size() == 3)
    {
      idv[i] = parknet::convert_3_corners_to_parking_slot_result(cam_id, corners, *dist_mapper_ptr_[cam_id].get());
    }
    else if ((corners.size() == 2) && (cam_id == parknet::camera::front_120_e))
    {
      idv[i] = parknet::convert_2_corners_to_parking_slot_result_in_front_120(cam_id, corners,
                                                                              *dist_mapper_ptr_[cam_id].get());
    }
    else if ((corners.size() == 2) && (cam_id == parknet::camera::right_120_e))
    {
      idv[i] = parknet::convert_2_corners_to_parking_slot_result_in_right_120(cam_id, corners,
                                                                              *dist_mapper_ptr_[cam_id].get());
    }

    const double sx_compensation = parknet::get_sx_compensation(cam_id);
    const double sy_compensation = parknet::get_sy_compensation(cam_id);
    for (auto& pslot : idv[i].parking_slots)
    {
      for (auto& marking_point : pslot.marking_points)
      {
        marking_point.x += sx_compensation;
        marking_point.y += sy_compensation;
      }
    }
    num_parking_slots += idv[i].parking_slots.size();
    reset_detection_result_by_cam_id(cam_id);
  }
  out.parking_slots.reserve(num_parking_slots);
  for (int i = 0; i < parknet::camera::num_cams_e; i++)
  {
    out.parking_slots.insert(out.parking_slots.end(), idv[i].parking_slots.begin(), idv[i].parking_slots.end());
  }
  if (out.parking_slots.size() == 0)
  {
    convert_side_corners_to_parking_slot_result(&out);
  }
  // Distance estimation uses LiDarTop coordinate system. Revise it based on /base_link.
  for (auto& pslot : out.parking_slots)
  {
    for (auto& marking_point : pslot.marking_points)
    {
      // offset x by 0.4m and z by 3.46m.
      marking_point.x += 0.4;
      marking_point.z += 3.46;
    }
  }
  auto now = ros::Time::now();
  out.header.stamp.sec = now.sec;
  out.header.stamp.nsec = now.nsec;
  return 0;
}

void ParknetNodeImpl::on_new_detection(const int cam_id)
{
  has_detection_results_ |= (1 << cam_id);
}

void ParknetNodeImpl::on_new_image_arrival(const int cam_id)
{
  under_detection_ |= (1 << cam_id);
}

bool ParknetNodeImpl::has_new_detection(const int cam_id) const
{
  return bool(has_detection_results_ & (1 << cam_id));
}

bool ParknetNodeImpl::image_is_under_detection(const int cam_id) const
{
  return bool(under_detection_ & (1 << cam_id));
}

void ParknetNodeImpl::reset_all_detection_results()
{
  for (int i = 0; i < parknet::camera::num_cams_e; i++)
  {
    const int cam_id = i;
    reset_detection_result_by_cam_id(cam_id);
  }
  assert(has_detection_results_ == 0);
  under_detection_ = 0;
}

void ParknetNodeImpl::reset_detection_result_by_cam_id(const int cam_id)
{
  unsigned int mask = (~0) ^ (1 << cam_id);
  has_detection_results_ &= mask;
  assert(!has_new_detection(cam_id));
}

bool ParknetNodeImpl::can_goto_next_round_of_detection() const
{
  return bool(has_detection_results_ == parknet::all_cameras_done_detection);
}

void ParknetNodeImpl::set_main_app_mode(bool mode)
{
  LOG(INFO) << "set main_app mode: " << mode;
  main_app_mode_ = mode;
}

void ParknetNodeImpl::set_unittest_mode(bool mode)
{
  LOG(INFO) << "set unittest mode: " << mode;
  unittest_mode_ = mode;
}

bool ParknetNodeImpl::in_main_app_mode() const
{
  return main_app_mode_;
}

const std::vector<RectClassScore<float> > ParknetNodeImpl::get_detection(const int cam_id) const
{
  // Each detection result.toString() is in the form
  // 1(x:360.626, y:190.639, w:17.0945, h:17.9133) =0.940128 in 608x608
  return parknet_image_manager_ptr_->get_detection(cam_id);
}

void ParknetNodeImpl::publish_parking_slot_result()
{
  LOG_EVERY_N(INFO, 120) << __FUNCTION__;
  if (parknet::in_barrier_synchroniztion_mode() && !can_goto_next_round_of_detection())
  {
    const int val = has_detection_results_.load();
    LOG_EVERY_N(INFO, 60) << "skip publishing result in this time slot: " << val;
  }
  else
  {
    msgs::ParkingSlotResult result;

    TIME_IT_BEGIN(post_processing);
    gen_parking_slot_result(result);
    if (parknet::in_profiling_mode())
    {
      TIME_IT_END(post_processing);
    }

    TIME_IT_BEGIN(output);
    // if (!in_main_app_mode())
    // {
    publisher_results_.publish(result);
    // }
    if (parknet::in_profiling_mode())
    {
      TIME_IT_END(output);
    }

    reset_all_detection_results();
    CHECK(has_detection_results_ == 0);
  }
}

int ParknetNodeImpl::on_init()
{
  __init_internal_vars();
  barrier_checking_thread_ = std::thread(&ParknetNodeImpl::barrier_checking, this);
  set_main_app_mode(true);
  set_unittest_mode(false);
  return 0;
}

int ParknetNodeImpl::on_inference(std::vector<Npp8u*>& npp8u_ptrs_cuda, const int num_bytes)
{
  // Called in main app mode
  LOG_EVERY_N(INFO, 60) << __FUNCTION__ << " npp version. num_bytes: " << num_bytes;
  std::thread detection_threads[parknet::camera::num_cams_e];
  bool running_threads[parknet::camera::num_cams_e];
  int num_detections = 0;
  for (int i = 0; i < parknet::camera::num_cams_e; i++)
  {
    const int cam_id = i;
    running_threads[cam_id] = false;
    if (image_is_under_detection(cam_id))
    {
      LOG_EVERY_N(INFO, 60) << parknet::camera_names[cam_id] << ": skip frame before consuming detection results";
      barrier_cond_var_.notify_one();
    }
    else
    {
      on_new_image_arrival(cam_id);
      copy_and_preprocess_raw_image(npp8u_ptrs_cuda[cam_id], cam_id, num_bytes);
      detection_threads[cam_id] = std::thread(&ParknetNodeImpl::detect_parking_slots_by_cam_id_internal, this, cam_id);
      running_threads[cam_id] = true;
    }
    visualize_if_necessary();
  }
  for (int i = 0; i < parknet::camera::num_cams_e; i++)
  {
    if (running_threads[i])
    {
      detection_threads[i].join();
    }
  }
  for (int i = 0; i < parknet::camera::num_cams_e; i++)
  {
    if (running_threads[i])
    {
      num_detections += parknet_image_manager_ptr_->get_num_detections(i);
    }
  }

  return num_detections;
}

int ParknetNodeImpl::on_inference(std::vector<cv::Mat>& my_frames)
{
  // Called in main app mode
  LOG_EVERY_N(INFO, 60) << __FUNCTION__ << " cv::Mat version";
  std::thread detection_threads[parknet::camera::num_cams_e];
  bool running_threads[parknet::camera::num_cams_e];
  int num_detections = 0;
  for (int i = 0; i < parknet::camera::num_cams_e; i++)
  {
    const int cam_id = i;
    running_threads[cam_id] = false;
    if (image_is_under_detection(cam_id))
    {
      LOG_EVERY_N(INFO, 60) << parknet::camera_names[cam_id] << ": skip frame before consuming detection results";
      barrier_cond_var_.notify_one();
    }
    else
    {
      on_new_image_arrival(cam_id);
      copy_and_preprocess_raw_image(my_frames[i], cam_id);
      detection_threads[cam_id] = std::thread(&ParknetNodeImpl::detect_parking_slots_by_cam_id_internal, this, cam_id);
      running_threads[cam_id] = true;
    }
    visualize_if_necessary();
  }
  for (int i = 0; i < parknet::camera::num_cams_e; i++)
  {
    if (running_threads[i])
    {
      detection_threads[i].join();
    }
  }
  for (int i = 0; i < parknet::camera::num_cams_e; i++)
  {
    if (running_threads[i])
    {
      num_detections += parknet_image_manager_ptr_->get_num_detections(i);
    }
  }

  return num_detections;
}

int ParknetNodeImpl::on_release()
{
  LOG(INFO) << __FUNCTION__;
  gracefully_stop_threads();
  return 0;
}

void ParknetNodeImpl::run(int argc, char* argv[])
{
  on_init();
  set_main_app_mode(false);
  subscribe_and_advertise_topics();
  // ros::AsyncSpinner spinner(parknet::camera::num_cams_e);
  ros::AsyncSpinner spinner(1);
  spinner.start();
  ros::Rate r(parknet::expected_fps);
  while (ros::ok() && !done_with_profiling())
  {
    VLOG(2) << __FUNCTION__;
    visualize_if_necessary();
    r.sleep();
  }
  on_release();
  spinner.stop();
  LOG(INFO) << "END detection";
}
