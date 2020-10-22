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
#define DUMP_LOG 1
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
  float crossing_predict(std::vector<std::vector<float>>& bbox_array,
                         std::vector<std::vector<cv::Point2f>>& keypoint_array, int id, ros::Time time);
  float* get_triangle_angle(float x1, float y1, float x2, float y2, float x3, float y3);
  float get_distance2(float x1, float y1, float x2, float y2);
  float get_angle2(float x1, float y1, float x2, float y2);
  float predict_rf_pose(const cv::Mat& input_data);
  bool filter(const msgs::BoxPoint box_point, ros::Time time_stamp);
  bool check_in_polygon(cv::Point2f position, std::vector<cv::Point2f>& polygon);
  void clean_old_skeleton_buffer(std::vector<SkeletonBuffer>& skeleton_buffer, ros::Time msg_timestamp);

  // void draw_pedestrians(cv::Mat matrix);
  bool keypoint_is_detected(cv::Point2f keypoint);
  float adjust_probability(msgs::PedObject obj);
  int get_facing_direction(const std::vector<cv::Point2f>& keypoints);
  int get_body_direction(const std::vector<cv::Point2f>& keypoints);
  void display_on_terminal();

  // OpenPose components
  int openposeROS();
  std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> create_datum(cv::Mat& mat);
  bool display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datums_ptr);
  std::vector<cv::Point2f> get_openpose_keypoint(cv::Mat& input_image);
  openpose_ros::OpenPose openpose_;

  // All buffer components
  msgs::VehInfo veh_info_;
  std::vector<cv::Point3f> lanelet2_route_left_;
  std::vector<cv::Point3f> lanelet2_route_right_;
  std::vector<cv::Point2f> lanelet2_trajectory_;
  std::string time_nav_path_ = "NA";
  boost::circular_buffer<std::pair<ros::Time, cv::Mat>> front_image_cache_;
  boost::circular_buffer<std::pair<ros::Time, cv::Mat>> fov30_image_cache_;
  boost::circular_buffer<std::pair<ros::Time, cv::Mat>> left_image_cache_;
  boost::circular_buffer<std::pair<ros::Time, cv::Mat>> right_image_cache_;
  std::vector<std::string> ped_info_;
  std::string delay_from_camera_ = "NA";
  std::string chatter_callback_info_ = "Not running";
  std::vector<SkeletonBuffer> skeleton_buffer_front_;
  std::vector<SkeletonBuffer> skeleton_buffer_left_;
  std::vector<SkeletonBuffer> skeleton_buffer_right_;
  std::vector<SkeletonBuffer> skeleton_buffer_fov30_;
  int buffer_size_ = 60;

  // mutex for each bufffer component
  std::mutex mu_veh_info_;
  std::mutex mu_lanelet2_route_;
  std::mutex mu_lanelet2_trajectory_;
  std::mutex mu_time_nav_path_;
  std::mutex mu_front_image_cache_;
  std::mutex mu_fov30_image_cache_;
  std::mutex mu_left_image_cache_;
  std::mutex mu_right_image_cache_;
  std::mutex mu_ped_info_;
  std::mutex mu_delay_from_camera_;
  std::mutex mu_chatter_callback_info_;
  std::mutex mu_skeleton_buffer_;

  // ROS components
  ros::ServiceClient skip_frame_client_;
  ros::Publisher chatter_pub_front_;
  ros::Publisher chatter_pub_left_;
  ros::Publisher chatter_pub_right_;
  ros::Publisher chatter_pub_fov30_;
  ros::Publisher box_pub_front_;
  ros::Publisher box_pub_left_;
  ros::Publisher box_pub_right_;
  ros::Publisher box_pub_fov30_;
  ros::Publisher alert_pub_front_;
  ros::Publisher alert_pub_left_;
  ros::Publisher alert_pub_right_;
  ros::Publisher alert_pub_fov30_;
  ros::Publisher warning_zone_pub_;
  ros::Time total_time_;
  tf2_ros::Buffer tf_buffer_;

  // Variables
  cv::Ptr<cv::ml::RTrees> rf_pose_;
  boost::shared_ptr<ros::AsyncSpinner> async_spinner_1_;
  boost::shared_ptr<ros::AsyncSpinner> async_spinner_2_;
  boost::shared_ptr<ros::AsyncSpinner> async_spinner_3_;
  boost::shared_ptr<ros::AsyncSpinner> async_spinner_4_;
  boost::shared_ptr<ros::AsyncSpinner> async_spinner_5_;
  boost::shared_ptr<ros::AsyncSpinner> async_spinner_6_;
  boost::shared_ptr<ros::AsyncSpinner> async_spinner_7_;
  boost::shared_ptr<ros::AsyncSpinner> async_spinner_8_;
  boost::shared_ptr<ros::AsyncSpinner> async_spinner_9_;
  boost::shared_ptr<ros::AsyncSpinner> async_spinner_10_;
  boost::shared_ptr<ros::AsyncSpinner> async_spinner_11_;
  boost::shared_ptr<ros::AsyncSpinner> async_spinner_12_;
  boost::shared_ptr<ros::AsyncSpinner> async_spinner_13_;
  bool spinner_trigger_ = false;
  int count_;
  std::ofstream file_;
  double average_inference_time_ = 0;

  // Setup variables
  const double scaling_ratio_width_ = 0.3167;
  const double scaling_ratio_height_ = 0.3179;
  const unsigned int number_keypoints_ = 25;
  const unsigned int feature_num_ = 1174;
  const unsigned int frame_num_ = 10;

  // ROS param
  int cross_threshold_ = 55;  // percentage
  bool show_probability_ = true;
  int input_source_ = 3;
  double max_distance_ = 50;
  double danger_zone_distance_ = 2;
  bool use_2d_for_alarm_ = false;
  int skip_frame_number_ = 1;

  int direction_table_[16][5] = {
    {0,0,0,0,4},
    {1,0,0,0,1},
    {0,1,0,0,1},
    {1,1,0,0,1},
    {0,0,1,0,0},
    {1,0,1,0,4},
    {0,1,1,0,2},
    {1,1,1,0,1},
    {0,0,0,1,0},
    {1,0,0,1,3},
    {0,1,0,1,4},
    {1,1,0,1,2},
    {0,0,1,1,0},
    {1,0,1,1,2},
    {0,1,1,1,0},
    {1,1,1,1,2},
  };
};
}  // namespace ped

#endif  // __PEDESTRIAN_EVENT_H__
