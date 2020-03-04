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
#include "alignment.h"

/// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>

/// thread
#include <mutex>

/// camera layout
#if CAR_MODEL_IS_B1
const std::vector<camera::id> g_cam_ids{ camera::id::front_60 };
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
int g_image_w = camera::image_width;
int g_image_h = camera::image_height;
cv::Mat g_mat_0;
cv::Mat g_mat_0_raw;

/// lidar
pcl::PointCloud<pcl::PointXYZI>::Ptr g_lidarall_ptr(new pcl::PointCloud<pcl::PointXYZI>);
boost::shared_ptr<pcl::visualization::PCLVisualizer> g_viewer(new pcl::visualization::PCLVisualizer ("Cloud_Viewer"));
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_lidarall(g_lidarall_ptr, 255, 255, 255);
std::vector<pcl::visualization::Camera> g_cam; 

/// object
std::vector<msgs::DetectedObject> g_object_0;

//////////////////// for camera image
void callback_cam_Front_60(const sensor_msgs::Image::ConstPtr& msg)
{
  g_sync_lock_cam.lock();
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_mat_0 = cv_ptr->image;
  std_msgs::Header h = msg->header;
  g_sync_lock_cam.unlock();
}

//////////////////// for camera image
void callback_decode_cam_Front_60(sensor_msgs::CompressedImage compressImg)
{
  g_sync_lock_cam.lock();
  cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(g_mat_0);
  g_sync_lock_cam.unlock();
}

//////////////////// for camera object
void callback_object_cam_Front_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  g_sync_lock_object.lock();
  g_object_0 = msg->objects;
  g_sync_lock_object.unlock();
  // std::cout << camera::topics_obj[g_cam_ids[0]] << " size: " << g_object_0.size() << std::endl;
  // cout<< camera::topics_obj[g_cam_ids[0]] <<endl;
}

/// similar to above, this is just a backup and testing for printing lidar data ///
void callback_lidarall(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  g_sync_lock_lidar.lock();
  *g_lidarall_ptr = *msg;
  g_sync_lock_lidar.unlock();
  // std::cout << "Point cloud size: " << g_lidarall_ptr->size() << std::endl;
  // std::cout << "Lidar x: " << g_lidarall_ptr->points[0].x << ", y: " << g_lidarall_ptr->points[0].y << ", z: " << g_lidarall_ptr->points[0].z << std::endl;
}

void viewerInitializer()
{
  g_viewer->initCameraParameters ();    
  g_viewer->addCoordinateSystem (3.0 , 0, 0, 0);  // Origin(0, 0, 0)
  g_viewer->setCameraPosition(0, 0, 20, 0.2, 0, 0); // bird view
  g_viewer->setBackgroundColor (0, 0, 0);
  g_viewer->setShowFPS (false); 
}

void drawPointCloudOnImage()
{
  for (size_t i = 0; i < g_lidarall_ptr->size(); i++)
  {
    if (g_lidarall_ptr->points[i].x > 0)
    {
      // std::cout << "Lidar x: " << g_lidarall_ptr->points[i].x << ", y: " << g_lidarall_ptr->points[i].y <<
      // ", z: " << g_lidarall_ptr->points[i].z << std::endl;
      PixelPosition pixel_position{-1, -1};
      pixel_position = g_alignment.projectPointToPixel(g_lidarall_ptr->points[i]);
      if (pixel_position.u >= 0 && pixel_position.v >= 0)
      {
        cv::Point center_point_ = cv::Point(pixel_position.u, pixel_position.v);
        float distance_x = g_lidarall_ptr->points[i].x;
        cv::Scalar point_color = g_alignment.getDistColor(distance_x);
        cv::circle(g_mat_0, center_point_, 1, point_color, -1, cv::LINE_8, 0);
        // std::cout << "Camera u: " << pixel_position.u << ", v: " << pixel_position.v << std::endl;
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
    image_sub_cam_0 = nh.subscribe(camera_topicname_cam_0 + std::string("/compressed"), 1, callback_decode_cam_Front_60);
  }
  else
  {
    image_sub_cam_0 = nh.subscribe(camera_topicname_cam_0, 1, callback_cam_Front_60);
  }
  ros::Subscriber object_sub_cam_0;
  std::string object_topicName_cam_0 = camera::topics_obj[g_cam_ids[0]];
  object_sub_cam_0 = nh.subscribe(object_topicName_cam_0, 1, callback_object_cam_Front_60);

  /// lidar subscriber
  ros::Subscriber lidarall;
  lidarall = nh.subscribe("/LidarAll", 1, callback_lidarall);

  /// init
  g_alignment.projectMatrixInit(g_cam_ids[0]);

  std::string window_name_cam_0 = camera_topicname_cam_0;
  cv::namedWindow(window_name_cam_0, cv::WINDOW_NORMAL);
  cv::resizeWindow(window_name_cam_0, 480, 360);
  cv::moveWindow(window_name_cam_0, 1025, 30);

  /// viwerinit
  viewerInitializer();

  ros::Rate loop_rate(30);
  std::cout << "===== Alignment running... =====" << std::endl;
  while (ros::ok())
  {
    g_viewer->removePointCloud("Cloud viewer");
    if (!g_mat_0.empty())
    {
      g_sync_lock_cam.lock();
      g_sync_lock_lidar.lock();
      drawPointCloudOnImage();
      /// draw lidarall
      g_viewer->addPointCloud<pcl::PointXYZI> (g_lidarall_ptr, g_rgb_lidarall, "Cloud viewer");
      g_sync_lock_lidar.unlock();
      cv::imshow(window_name_cam_0, g_mat_0);
      g_sync_lock_cam.unlock();
      cv::waitKey(1);
    }
    ros::spinOnce();
    g_viewer->spinOnce();
    loop_rate.sleep();
  }
  std::cout << "===== Alignment shutdown. =====" << std::endl;
  return 0;
}
