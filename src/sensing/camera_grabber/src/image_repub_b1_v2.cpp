/// standard
#include <iostream>

/// ros
#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <std_msgs/Empty.h>

/// package
#include "camera_params.h"

/// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/// camera layout
#if CAR_MODEL_IS_B1_V2
const std::vector<camera::id> g_cam_ids{ camera::id::front_bottom_60,     camera::id::front_top_far_30,
                                         camera::id::front_top_close_120, camera::id::right_front_60,
                                         camera::id::right_back_60,       camera::id::left_front_60,
                                         camera::id::left_back_60,        camera::id::back_top_120 };
#else
#error "car model is not well defined"
#endif

/// params
bool g_input_resize = true;
bool g_img_result_publish = true;
bool g_display = false;

/// ros
std::vector<std::string> g_cam_topic_names(g_cam_ids.size());
std::vector<image_transport::Publisher> g_cam_pubs(g_cam_ids.size());
std::vector<ros::Publisher> g_heartbeat_pubs(g_cam_ids.size());

/// image
int g_image_w = camera::image_width;
int g_image_h = camera::image_height;
int g_raw_image_w = camera::raw_image_width;
int g_raw_image_h = camera::raw_image_height;
std::vector<cv::Mat> g_mats(g_cam_ids.size());

/// publish image message
void image_publisher(image_transport::Publisher& img_pub, ros::Publisher& heartbeat_pub, const sensor_msgs::Image::ConstPtr& img_msg)
{
  img_pub.publish(img_msg);
  std_msgs::Empty empty_msg;
  heartbeat_pub.publish(empty_msg);
}
//////////////////// for camera image
void callback_cam_front_bottom_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_bottom_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);

  image_publisher(g_cam_pubs[cam_order], g_heartbeat_pubs[cam_order], msg);

  // std::cout << camera::topics[g_cam_ids[cam_order]] << " time: " << msg->header.stamp.sec << "."
  // << msg->header.stamp.nsec << std::endl;
}

void callback_cam_front_top_far_30(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_top_far_30);
  int cam_order = std::distance(g_cam_ids.begin(), it);

  image_publisher(g_cam_pubs[cam_order], g_heartbeat_pubs[cam_order], msg);

  // std::cout << camera::topics[g_cam_ids[cam_order]] << " time: " << msg->header.stamp.sec << "."
  // << msg->header.stamp.nsec << std::endl;
}

void callback_cam_front_top_close_120(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_top_close_120);
  int cam_order = std::distance(g_cam_ids.begin(), it);

  image_publisher(g_cam_pubs[cam_order], g_heartbeat_pubs[cam_order], msg);
  // std::cout << camera::topics[g_cam_ids[cam_order]] << " time: " << msg->header.stamp.sec << "."
  // << msg->header.stamp.nsec << std::endl;
}

void callback_cam_right_front_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_front_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);

  image_publisher(g_cam_pubs[cam_order], g_heartbeat_pubs[cam_order], msg);
  // std::cout << camera::topics[g_cam_ids[cam_order]] << " time: " << msg->header.stamp.sec << "."
  // << msg->header.stamp.nsec << std::endl;
}

void callback_cam_right_back_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_back_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);

  image_publisher(g_cam_pubs[cam_order], g_heartbeat_pubs[cam_order], msg);
  // std::cout << camera::topics[g_cam_ids[cam_order]] << " time: " << msg->header.stamp.sec << "."
  // << msg->header.stamp.nsec << std::endl;
}

void callback_cam_left_front_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_front_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);

  image_publisher(g_cam_pubs[cam_order], g_heartbeat_pubs[cam_order], msg);
  // std::cout << camera::topics[g_cam_ids[cam_order]] << " time: " << msg->header.stamp.sec << "."
  // << msg->header.stamp.nsec << std::endl;
}

void callback_cam_left_back_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_back_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);

  image_publisher(g_cam_pubs[cam_order], g_heartbeat_pubs[cam_order], msg);
  // std::cout << camera::topics[g_cam_ids[cam_order]] << " time: " << msg->header.stamp.sec << "."
  // << msg->header.stamp.nsec << std::endl;
}

void callback_cam_back_top_120(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::back_top_120);
  int cam_order = std::distance(g_cam_ids.begin(), it);

  image_publisher(g_cam_pubs[cam_order], g_heartbeat_pubs[cam_order], msg);
  // std::cout << camera::topics[g_cam_ids[cam_order]] << " time: " << msg->header.stamp.sec << "."
  // << msg->header.stamp.nsec << std::endl;
}
int main(int argc, char** argv)
{
  std::cout << "===== image_repub_b1_v2 startup. =====" << std::endl;
  ros::init(argc, argv, "image_repub_b1_v2");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  /// ros Subscriber
  std::vector<ros::Subscriber> cam_subs(g_cam_ids.size());

  /// get callback function
  static void (*f_callbacks_cam[])(const sensor_msgs::Image::ConstPtr&) = {
    callback_cam_front_bottom_60, callback_cam_front_top_far_30, callback_cam_front_top_close_120,
    callback_cam_right_front_60,  callback_cam_right_back_60,    callback_cam_left_front_60,
    callback_cam_left_back_60,    callback_cam_back_top_120
  };

  /// set topic name
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    g_cam_topic_names[cam_order] = camera::topics[g_cam_ids[cam_order]];
  }

  /// ros Subscriber
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    cam_subs[cam_order] = nh.subscribe(g_cam_topic_names[cam_order] + std::string("/raw"), 1, f_callbacks_cam[cam_order]);
    g_cam_pubs[cam_order] = it.advertise(g_cam_topic_names[cam_order], 1);
    g_heartbeat_pubs[cam_order] = nh.advertise<std_msgs::Empty>(g_cam_topic_names[cam_order] + std::string("/heartbeat"), 1);
  }

  /// main loop start
  int thread_count = int(g_cam_ids.size());  /// camera raw
  ros::MultiThreadedSpinner spinner(thread_count);
  spinner.spin();

  std::cout << "===== image_repub_b1_v2 shutdown. =====" << std::endl;
  return 0;
}
