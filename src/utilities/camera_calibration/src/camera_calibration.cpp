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
#if CAR_MODEL_IS_B1_V3 || CAR_MODEL_IS_C1
const std::vector<camera::id> g_cam_ids{ camera::id::front_bottom_60,     camera::id::front_top_far_30,
                                   camera::id::front_top_close_120, camera::id::right_front_60,
                                   camera::id::right_back_60,       camera::id::left_front_60,
                                   camera::id::left_back_60,        camera::id::back_top_120 };
const std::vector<bool> g_do_flip{ false, false, false, true, false, true, false, false};
#else
#error "car model is not well defined"
#endif

/// param
bool g_input_resize = true;
bool g_display = true;

/// thread
std::vector<std::mutex> g_cam_mutex(g_cam_ids.size());

/// images
int g_img_w = camera::raw_image_width;
int g_img_h = camera::raw_image_height;
std::vector<cv::Mat> g_mats(g_cam_ids.size());

/// callback
void callback_cam_front_bottom_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_bottom_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_cam_mutex[cam_order].lock();
  g_mats[cam_order] = cv_ptr->image;
  g_cam_mutex[cam_order].unlock();
}
void callback_cam_front_top_far_30(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_top_far_30);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_cam_mutex[cam_order].lock();
  g_mats[cam_order] = cv_ptr->image;
  g_cam_mutex[cam_order].unlock();
}
void callback_cam_front_top_close_120(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_top_close_120);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_cam_mutex[cam_order].lock();
  g_mats[cam_order] = cv_ptr->image;
  g_cam_mutex[cam_order].unlock();
}
void callback_cam_right_front_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_front_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_cam_mutex[cam_order].lock();
  g_mats[cam_order] = cv_ptr->image;
  g_cam_mutex[cam_order].unlock();
}
void callback_cam_right_back_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_back_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_cam_mutex[cam_order].lock();
  g_mats[cam_order] = cv_ptr->image;
  g_cam_mutex[cam_order].unlock();
}
void callback_cam_left_front_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_front_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_cam_mutex[cam_order].lock();
  g_mats[cam_order] = cv_ptr->image;
  g_cam_mutex[cam_order].unlock();
}
void callback_cam_left_back_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_back_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_cam_mutex[cam_order].lock();
  g_mats[cam_order] = cv_ptr->image;
  g_cam_mutex[cam_order].unlock();
}
void callback_cam_back_top_120(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::back_top_120);
  int cam_order = std::distance(g_cam_ids.begin(), it);
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

void calibration_viewer()
{
  std::string window_name_front = "stitch_img_front";
  std::string window_name_side = "stitch_img_side";
  std::string window_name_back = "stitch_img_back";
  if (g_display)
  {
    cv::namedWindow(window_name_front, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_front, 480, 360);
    cv::moveWindow(window_name_front, 65, 30);

    cv::namedWindow(window_name_side, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_side, 480, 360);
    cv::moveWindow(window_name_side, 545, 30);

    cv::namedWindow(window_name_back, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_back, 480, 360);
    cv::moveWindow(window_name_back, 545 + 480, 30);
  }
  std::vector<cv::Mat> mats(g_cam_ids.size());
  cv::Mat stitch_img_front;
  cv::Mat stitch_img_side;
  cv::Mat stitch_img_side_row_1;
  cv::Mat stitch_img_side_row_2;
  std_msgs::Header msg_header;

  int pixel_loc_col = 50;
  int pixel_loc_row = 25;
  std::vector<cv::Point> line_col {cv::Point(-1, -1), cv::Point(-1, -1), cv::Point(-1, -1), cv::Point((g_img_w-1)-pixel_loc_col, 0), cv::Point(pixel_loc_col, 0), cv::Point(pixel_loc_col, 0), cv::Point((g_img_w-1)-pixel_loc_col, 0), cv::Point(-1, -1)};
  std::vector<cv::Point> line_row {cv::Point(-1, -1), cv::Point(-1, -1), cv::Point(-1, -1), cv::Point(-1, -1), cv::Point(-1, -1), cv::Point(-1, -1), cv::Point(-1, -1), cv::Point(0, (g_img_h-1)-pixel_loc_row)};
  
  ros::Rate r(20);
  while (ros::ok())
  {
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      g_cam_mutex[cam_order].lock();
      mats[cam_order] = g_mats[cam_order].clone();
      g_cam_mutex[cam_order].unlock();

      cv::line(mats[cam_order], line_col[cam_order],  line_col[cam_order] + cv::Point(0, g_img_h), CV_RGB(255, 0, 0), 1, 8, 0);
      cv::line(mats[cam_order], line_row[cam_order],  line_row[cam_order] + cv::Point(g_img_w, 0), CV_RGB(255, 0, 0), 1, 8, 0);

      if (g_do_flip[cam_order])
      {
        cv::flip(mats[cam_order], mats[cam_order], 1);
      }

    }
    cv::hconcat(mats[6], mats[4], stitch_img_side_row_1);
    cv::hconcat(mats[5], mats[3], stitch_img_side_row_2);
    cv::vconcat(stitch_img_side_row_1, stitch_img_side_row_2, stitch_img_side);

    cv::vconcat(mats[1], mats[0], stitch_img_front);
    cv::vconcat(stitch_img_front, mats[2], stitch_img_front);
    cv::line(stitch_img_front, cv::Point(g_img_w/2, 0), cv::Point(g_img_w/2, g_img_h*3), CV_RGB(255, 0, 0), 1, 8, 0);

    if (g_display)
    {
      cv::imshow(window_name_front, stitch_img_front);
      cv::imshow(window_name_side, stitch_img_side);
      cv::imshow(window_name_back, mats[7]);
      cv::waitKey(1);
    }
    r.sleep();
  }
}

int main(int argc, char** argv)
{
  std::cout << "===== camera_calibration startup. =====" << std::endl;
  ros::init(argc, argv, "camera_calibration");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  std::vector<std::string> cam_topic_names(g_cam_ids.size());
  std::vector<ros::Subscriber> cam_subs(g_cam_ids.size());

  static void (*f_cam_callbacks[])(const sensor_msgs::Image::ConstPtr&) = {
    callback_cam_front_bottom_60, callback_cam_front_top_far_30, callback_cam_front_top_close_120,
    callback_cam_right_front_60, callback_cam_right_back_60, callback_cam_left_front_60, callback_cam_left_back_60,
    callback_cam_back_top_120
  };

  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    cam_topic_names[cam_order] = camera::topics[g_cam_ids[cam_order]];
    cam_subs[cam_order] = nh.subscribe(cam_topic_names[cam_order], 1, f_cam_callbacks[cam_order]);
  }

  /// init
  image_init();

  /// main loop start
  std::thread main_thread(calibration_viewer);
  std::cout << "===== camera_calibration running... =====" << std::endl;
  int thread_count = int(g_cam_ids.size());  /// camera raw
  ros::MultiThreadedSpinner spinner(thread_count);
  spinner.spin();

  main_thread.join();
  std::cout << "===== camera_calibration shutdown. =====" << std::endl;
  return 0;
}
