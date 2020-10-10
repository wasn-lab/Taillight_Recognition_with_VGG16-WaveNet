#ifndef __PEDESTRIAN_EVENT_H__
#define __PEDESTRIAN_EVENT_H__

#include "ros/ros.h"
#include <ros/spinner.h>
#include <ros/callback_queue.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/Image.h>
#include <tf/tf.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/Point32.h>
#include <nav_msgs/Path.h>
#include "msgs/VehInfo.h"
#include "msgs/BoxPoint.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/PedObject.h"
#include "msgs/PedObjectArray.h"
#include "autoware_planning_msgs/Trajectory.h"
#include "msgs/PredictSkeleton.h"
#include "msgs/Keypoints.h"
#include "msgs/Keypoint.h"

#include <opencv2/opencv.hpp>  // opencv general include file
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>  // opencv machine learning include file
#include <cv_bridge/cv_bridge.h>
#include <ped_def.h>
#include <map>
#include <boost/circular_buffer.hpp>
#include <buffer.h>
#include <skeleton_buffer.h>
#include <openpose.h>
#include <openpose_ros_io.h>
#include <openpose_flags.h>
// C++ std library dependencies
#include <chrono>  // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <thread>  // std::this_thread
#include <fstream>
#include <sys/ioctl.h>
#include <unistd.h>

#include <tensorflow/c/c_api.h>  // TensorFlow C API header.
//#include <scope_guard.hpp>
//#include <tf_utils.hpp>

#define USE_2D_FOR_ALARM 0
#define DUMP_LOG 0
#define PRINT_MESSAGE 0
#define USE_GLOG 1
#if USE_GLOG
#include "glog/logging.h"
#define LOG_INFO LOG(INFO)
#define LOG_WARNING LOG(WARNING)
#define LOG_ERROR LOG(ERROR)
#define LOG_FATAL LOG(FATAL)
#else
#define LOG_INFO std::cout
#define LOG_WARNING std::cout
#define LOG_ERROR std::cout
#define LOG_FATAL std::cout
#endif

#define M_PIl 3.141592653589793238462643383279502884L /* pi */

namespace ped
{
class PedestrianEvent
{
public:
  PedestrianEvent()
  {
  }

  ~PedestrianEvent()
  {
  }

  // Functions
  void run();
  void veh_info_callback(const msgs::VehInfo::ConstPtr& msg);
  void nav_path_callback(const nav_msgs::Path::ConstPtr& msg);
  void lanelet2_trajectory_callback(const autoware_planning_msgs::Trajectory::ConstPtr& msg);
  void lanelet2_route_callback(const visualization_msgs::MarkerArray::ConstPtr& msg);
  void cache_front_image_callback(const sensor_msgs::Image::ConstPtr& msg);
  void cache_left_image_callback(const sensor_msgs::Image::ConstPtr& msg);
  void cache_right_image_callback(const sensor_msgs::Image::ConstPtr& msg);
  void cache_fov30_image_callback(const sensor_msgs::Image::ConstPtr& msg);
  void cache_image_callback(const sensor_msgs::Image::ConstPtr& msg,
                            boost::circular_buffer<std::pair<ros::Time, cv::Mat>>& image_cache);
  void front_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void left_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void right_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void fov30_callback(const msgs::DetectedObjectArray::ConstPtr& msg);
  void main_callback(const msgs::DetectedObjectArray::ConstPtr& msg,
                     boost::circular_buffer<std::pair<ros::Time, cv::Mat>>& image_cache, int from_camera,
                     std::vector<SkeletonBuffer>& skeleton_buffer);
  void draw_ped_front_callback(const msgs::PedObjectArray::ConstPtr& msg);
  void draw_ped_left_callback(const msgs::PedObjectArray::ConstPtr& msg);
  void draw_ped_right_callback(const msgs::PedObjectArray::ConstPtr& msg);
  void draw_ped_fov30_callback(const msgs::PedObjectArray::ConstPtr& msg);
  void draw_pedestrians_callback(const msgs::PedObjectArray::ConstPtr& msg,
                                 boost::circular_buffer<std::pair<ros::Time, cv::Mat>>& image_cache, int from_camera);
  void pedestrian_event();
  float crossing_predict(std::vector<std::vector<float>>& bbox_array, std::vector<std::vector<cv::Point2f>>& keypoint_array, int id,
                         ros::Time time);
  float* get_triangle_angle(float x1, float y1, float x2, float y2, float x3, float y3);
  float get_distance2(float x1, float y1, float x2, float y2);
  float get_angle2(float x1, float y1, float x2, float y2);
  float predict_rf_pose(const cv::Mat& input_data);
  bool filter(const msgs::BoxPoint box_point, ros::Time time_stamp);
  bool check_in_polygon(cv::Point2f position, std::vector<cv::Point2f>& polygon);
  void clean_old_skeleton_buffer(std::vector<SkeletonBuffer>& skeleton_buffer);

  // void draw_pedestrians(cv::Mat matrix);
  bool keypoint_is_detected(cv::Point2f keypoint);
  float adjust_probability(msgs::PedObject obj);
  int get_facing_direction(const std::vector<cv::Point2f>& keypoints);
  int get_body_direction(const std::vector<cv::Point2f>& keypoints);
  void display_on_terminal();

  // OpenPose components
  int openPoseROS();
  std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> createDatum(cv::Mat& mat);
  bool display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datums_ptr);
  std::vector<cv::Point2f> get_openpose_keypoint(cv::Mat& input_image);
  openpose_ros::OpenPose openPose;
  cv::dnn::Net net_openpose;

  // All buffer components
  msgs::VehInfo veh_info;
  std::vector<cv::Point3f> lanelet2_route_left;
  std::vector<cv::Point3f> lanelet2_route_right;
  std::vector<cv::Point2f> lanelet2_trajectory;
  std::string time_nav_path = "NA";
  boost::circular_buffer<std::pair<ros::Time, cv::Mat>> front_image_cache;
  boost::circular_buffer<std::pair<ros::Time, cv::Mat>> fov30_image_cache;
  boost::circular_buffer<std::pair<ros::Time, cv::Mat>> left_image_cache;
  boost::circular_buffer<std::pair<ros::Time, cv::Mat>> right_image_cache;
  std::vector<std::string> ped_info;
  std::string delay_from_camera = "NA";
  std::string chatter_callback_info = "Not running";
  Buffer buffer_front;
  Buffer buffer_left;
  Buffer buffer_right;
  Buffer buffer_fov30;
  std::vector<SkeletonBuffer> skeleton_buffer_front;
  std::vector<SkeletonBuffer> skeleton_buffer_left;
  std::vector<SkeletonBuffer> skeleton_buffer_right;
  std::vector<SkeletonBuffer> skeleton_buffer_fov30;
  int buffer_size = 60;

  // mutex for each bufffer component
  std::mutex mu_veh_info;
  std::mutex mu_lanelet2_route;
  std::mutex mu_lanelet2_trajectory;
  std::mutex mu_image_cache;
  std::mutex mu_ped_info;
  std::mutex mu_buffer;
  std::mutex mu_skeleton_buffer;

  // ROS components
  ros::ServiceClient skip_frame_client;
  ros::Publisher chatter_pub_front;
  ros::Publisher chatter_pub_left;
  ros::Publisher chatter_pub_right;
  ros::Publisher chatter_pub_fov30;
  ros::Publisher box_pub_front;
  ros::Publisher box_pub_left;
  ros::Publisher box_pub_right;
  ros::Publisher box_pub_fov30;
  ros::Publisher alert_pub_front;
  ros::Publisher alert_pub_left;
  ros::Publisher alert_pub_right;
  ros::Publisher alert_pub_fov30;
  ros::Publisher warning_zone_pub;
  ros::Time total_time;
  tf2_ros::Buffer tfBuffer;

  // Variables
  cv::Ptr<cv::ml::RTrees> rf;
  cv::Ptr<cv::ml::RTrees> rf_pose;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_1;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_2;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_3;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_4;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_5;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_6;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_7;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_8;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_9;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_10;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_11;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_12;
  boost::shared_ptr<ros::AsyncSpinner> g_spinner_13;
  bool g_enable = false;
  bool g_trigger = false;
  int count;
  std::ofstream file;
  struct winsize terminal_size;
  double average_inference_time = 0;

  // Setup variables
  int cross_threshold = 55;  // percentage
  const double scaling_ratio_width = 0.3167;
  const double scaling_ratio_height = 0.3179;
  const int number_keypoints = 25;
  const int feature_num = 1174;
  const int frame_num = 10;

  // ROS param
  bool show_probability = true;
  int input_source = 3;
  double max_distance = 50;
  double danger_zone_distance = 2;
  bool use_2d_for_alarm = false;
};
}  // namespace ped

#endif  // __PEDESTRIAN_EVENT_H__
