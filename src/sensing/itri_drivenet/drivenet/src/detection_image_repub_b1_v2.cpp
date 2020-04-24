/// std
#include <thread>
#include <mutex>
#include <iostream>

/// ros
#include "ros/ros.h"
#include <ros/package.h>
#include "std_msgs/Header.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

/// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/// util
#include "camera_params.h"  // include camera topic name

/// camera layout
#if CAR_MODEL_IS_B1_V2
const std::vector<camera::id> g_cam_ids{ camera::id::left_back_60, camera::id::front_top_close_120,
                                         camera::id::right_back_60, camera::id::left_front_60,
                                         camera::id::right_front_60 };
const std::vector<bool> g_do_flip{ false, false, false, true, true };
#else
#error "car model is not well defined"
#endif

// ROS Publisher: 1, represent all cams
image_transport::Publisher g_img_pub;

/// param
bool g_input_resize = true;
bool g_img_result_publish = true;
bool g_display = false;

/// thread
std::vector<std::mutex> g_cam_mutex(g_cam_ids.size());
std::mutex g_header_mutex;

/// images
int g_img_w = camera::raw_image_width;
int g_img_h = camera::raw_image_height;
std::vector<cv::Mat> g_mats(g_cam_ids.size());
std_msgs::Header g_msg_header;

/// callback
void callback_cam_left_back_60(const sensor_msgs::Image::ConstPtr& msg)
{
  int cam_order = 0;
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_cam_mutex[cam_order].lock();
  g_mats[cam_order] = cv_ptr->image;
  g_cam_mutex[cam_order].unlock();
}
void callback_cam_front_top_close_120(const sensor_msgs::Image::ConstPtr& msg)
{
  int cam_order = 1;
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_cam_mutex[cam_order].lock();
  g_mats[cam_order] = cv_ptr->image;
  g_cam_mutex[cam_order].unlock();
  g_header_mutex.lock();
  g_msg_header = msg->header;
  g_header_mutex.unlock();
}
void callback_cam_right_back_60(const sensor_msgs::Image::ConstPtr& msg)
{
  int cam_order = 2;
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_cam_mutex[cam_order].lock();
  g_mats[cam_order] = cv_ptr->image;
  g_cam_mutex[cam_order].unlock();
}
void callback_cam_left_front_60(const sensor_msgs::Image::ConstPtr& msg)
{
  int cam_order = 3;
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_cam_mutex[cam_order].lock();
  g_mats[cam_order] = cv_ptr->image;
  g_cam_mutex[cam_order].unlock();
}

void callback_cam_right_front_60(const sensor_msgs::Image::ConstPtr& msg)
{
  int cam_order = 4;
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_cam_mutex[cam_order].lock();
  g_mats[cam_order] = cv_ptr->image;
  g_cam_mutex[cam_order].unlock();
}

/// Prepare cv::Mat
void image_init()
{
  if (g_input_resize)
  {
    g_img_w = camera::image_width;
    g_img_h = camera::image_height;
  }
  // g_img_size = g_img_w * g_img_h;

  for (size_t ndx = 0; ndx < g_cam_ids.size(); ndx++)
  {
    g_mats[ndx] = cv::Mat(g_img_h, g_img_w, CV_8UC3, cv::Scalar(255, 255, 255));
  }
}
/// publish image message
void image_publisher(const cv::Mat& image, const std_msgs::Header& header, image_transport::Publisher& img_pub)
{
  sensor_msgs::ImagePtr img_msg;
  img_msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
  img_pub.publish(img_msg);
}

/// publish image message
void collect_repub()
{
  std::string window_name = "stitch_img";
  if (g_display)
  {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 480, 360);
    cv::moveWindow(window_name, 545, 30);
  }
  std::vector<cv::Mat> mats(g_cam_ids.size());
  cv::Mat stitch_img = cv::Mat(g_img_h * 2, g_img_w * 3, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::Mat stitch_img_resize = cv::Mat((g_img_h * 2) / 2, (g_img_w * 3) / 2, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::Mat stitch_img_row_1 = cv::Mat(g_img_h, g_img_w * 3, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::Mat stitch_img_row_2 = cv::Mat(g_img_h, g_img_w * 3, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::Mat white_img = cv::Mat(g_img_h, g_img_w, CV_8UC3, cv::Scalar(255, 255, 255));
  std_msgs::Header msg_header;

  ros::Rate r(20);
  while (ros::ok())
  {
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      g_cam_mutex[cam_order].lock();
      mats[cam_order] = g_mats[cam_order].clone();
      g_cam_mutex[cam_order].unlock();

      g_header_mutex.lock();
      msg_header = g_msg_header;
      g_header_mutex.unlock();

      if (g_do_flip[cam_order])
      {
        cv::flip(mats[cam_order], mats[cam_order], 1);
      }
    }
    cv::hconcat(mats[0], mats[1], stitch_img_row_1);
    cv::hconcat(stitch_img_row_1, mats[2], stitch_img_row_1);
    cv::hconcat(mats[3], white_img, stitch_img_row_2);
    cv::hconcat(stitch_img_row_2, mats[4], stitch_img_row_2);
    cv::vconcat(stitch_img_row_1, stitch_img_row_2, stitch_img);
    cv::resize(stitch_img, stitch_img_resize, cv::Size((g_img_w * 3) / 2, (g_img_h * 2) / 2));

    if (g_display)
    {
      cv::imshow(window_name, stitch_img_resize);
      cv::waitKey(1);
    }
    if (g_img_result_publish)
    {
      image_publisher(stitch_img_resize, msg_header, g_img_pub);
    }
    r.sleep();
  }
}

int main(int argc, char** argv)
{
  std::cout << "===== detection_image_repub_b1_v2 startup. =====" << std::endl;
  ros::init(argc, argv, "detection_image_repub_b1_v2");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  std::vector<std::string> cam_topic_names(g_cam_ids.size());
  std::vector<ros::Subscriber> cam_subs(g_cam_ids.size());
  std::string cam_pub_topic_name = "/cam/detect_image/stitch_mid";

  static void (*f_cam_callbacks[])(const sensor_msgs::Image::ConstPtr&) = {
    callback_cam_left_back_60, callback_cam_front_top_close_120, callback_cam_right_back_60, callback_cam_left_front_60,
    callback_cam_right_front_60
  };
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    cam_topic_names[cam_order] = camera::topics[g_cam_ids[cam_order]] + std::string("/detect_image");
    cam_subs[cam_order] = nh.subscribe(cam_topic_names[cam_order], 1, f_cam_callbacks[cam_order]);
  }

  if (g_img_result_publish)
  {
    g_img_pub = it.advertise(cam_pub_topic_name, 1);
  }

  /// init
  image_init();

  /// main loop start
  std::thread main_thread(collect_repub);
  std::cout << "===== detection_image_repub_b1_v2 running... =====" << std::endl;
  int thread_count = int(g_cam_ids.size()) + 1;  /// camera raw + object + lidar raw
  ros::MultiThreadedSpinner spinner(thread_count);
  spinner.spin();

  main_thread.join();
  std::cout << "===== detection_image_repub_b1_v2 shutdown. =====" << std::endl;
  return 0;
}
