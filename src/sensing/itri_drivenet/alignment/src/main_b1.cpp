/// standard
#include <iostream>

/// ros
#include "ros/ros.h"
#include <msgs/DetectedObjectArray.h>
#include <msgs/DetectedObject.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

/// package
#include "camera_params.h"
#include "drivenet/image_preprocessing.h"
#include "alignment.h"

/// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

/// thread
#include <mutex>

/// namespace
using namespace DriveNet;

/// camera layout
#if CAR_MODEL_IS_B1
const std::vector<int> g_cam_ids{ camera::id::front_60};
#else
#error "car model is not well defined"
#endif

/// class
Alignment g_alignment;

/// thread
mutex g_sync_lock_cam;
mutex g_sync_lock_object;
mutex g_sync_lock_lidar;

/// params
bool g_is_compressed = false;

/// image
int g_image_w_ = camera::image_width;
int g_image_h_ = camera::image_height;
cv::Mat g_mat_0;
cv::Mat g_mat_0_raw;

/// lidar
pcl::PointCloud<pcl::PointXYZI> g_lidarall_nonground;

/// object
std::vector<msgs::DetectedObject> g_object_0;

//////////////////// for camera image
void callback_cam_0(const sensor_msgs::Image::ConstPtr& msg)
{
  g_sync_lock_cam.lock();
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_mat_0 = cv_ptr->image;
  std_msgs::Header h = msg->header;
  g_sync_lock_cam.unlock();
}

//////////////////// for camera image
void callback_decode_cam_0(sensor_msgs::CompressedImage compressImg)
{
  g_sync_lock_cam.lock();
  cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(g_mat_0);
  g_sync_lock_cam.unlock();
}

//////////////////// for camera object
void callback_object_cam_0(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  g_sync_lock_object.lock();
  g_object_0 = msg->objects;
  g_sync_lock_object.unlock();
  // std::cout << camera::topics_obj[g_cam_ids[0]] << " size: " << g_object_0.size() << std::endl;
  // cout<< camera::topics_obj[g_cam_ids[0]] <<endl;
}

/// similar to above, this is just a backup and testing for printing lidar data ///
void lidarAllCallback(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  g_sync_lock_lidar.lock();
  pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_cur_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  *ptr_cur_cloud = *msg;
  g_lidarall_nonground = *ptr_cur_cloud;
  g_sync_lock_lidar.unlock();
  // std::cout << "Point cloud size: " << g_lidarall_nonground.size() << std::endl;
  // std::cout << "Lidar x: " << g_lidarall_nonground.points[0].x << ", y: " << g_lidarall_nonground.points[0].y << ",
  // z: " << g_lidarall_nonground.points[0].z << std::endl;
}

void drawPointCloudOnImage()
{
  for (size_t i = 0; i < g_lidarall_nonground.size(); i++)
  {
    if (g_lidarall_nonground.points[i].x > 0)
    {
      // std::cout << "Lidar x: " << g_lidarall_nonground.points[i].x << ", y: " << g_lidarall_nonground.points[i].y <<
      // ", z: " << g_lidarall_nonground.points[i].z << std::endl;
      PixelPosition pixel_position_;
      pixel_position_ = g_alignment.projectPointToPixel(g_lidarall_nonground.points[i]);
      if (pixel_position_.u >= 0 && pixel_position_.v >= 0)
      {
        cv::Point center_point_ = cv::Point(pixel_position_.u, pixel_position_.v);
        cv::circle(g_mat_0, center_point_, 1, Color::g_color_green, -1, LINE_8, 0);
        // std::cout << "Camera u: " << pixel_position_.u << ", v: " << pixel_position_.v << std::endl;
      }
    }
  }
}
int main(int argc, char** argv)
{
  std::cout << "===== Alignment startup. =====" << std::endl;
  ros::init(argc, argv, "Alignment");
  ros::NodeHandle nh;

  /// camera subscriber
  ros::Subscriber image_sub_cam_0;
  std::string camera_topicname_cam_0 = camera::topics[g_cam_ids[0]];
  if (g_is_compressed)
  {
    image_sub_cam_0 = nh.subscribe(camera_topicname_cam_0 + std::string("/compressed"), 1, callback_decode_cam_0);
  }
  else
  {
    image_sub_cam_0 = nh.subscribe(camera_topicname_cam_0, 1, callback_cam_0);
  }
  ros::Subscriber object_sub_cam_0;
  std::string object_topicName_cam_0 = camera::topics_obj[g_cam_ids[0]];
  object_sub_cam_0 = nh.subscribe(object_topicName_cam_0, 1, callback_object_cam_0);

  /// lidar subscriber
  ros::Subscriber lidarall;
  lidarall = nh.subscribe("/LidarAll", 1, lidarAllCallback);

  /// init
  g_alignment.projectMatrixInit(g_cam_ids[0]);

  std::string window_name_cam_0 = camera_topicname_cam_0;
  cv::namedWindow(window_name_cam_0, cv::WINDOW_NORMAL);
  cv::resizeWindow(window_name_cam_0, 480, 360);
  cv::moveWindow(window_name_cam_0, 1025, 30);

  ros::Rate loop_rate(30);
  std::cout << "===== Alignment running... =====" << std::endl;
  while (ros::ok())
  {
    ros::spinOnce();
    if (!g_mat_0.empty())
    {
      g_sync_lock_cam.lock();
      g_sync_lock_lidar.lock();
      drawPointCloudOnImage();
      g_sync_lock_lidar.unlock();
      cv::imshow(window_name_cam_0, g_mat_0);
      g_sync_lock_cam.unlock();
      cv::waitKey(1);
    }
    loop_rate.sleep();
  }
  std::cout << "===== Alignment shutdown. =====" << std::endl;
  return 0;
}
