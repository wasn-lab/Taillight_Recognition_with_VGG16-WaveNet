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
const std::vector<camera::id> g_cam_ids{ camera::id::front_60, camera::id::top_front_120, camera::id::left_60,
                                         camera::id::right_60 };
#else
#error "car model is not well defined"
#endif

/// class
std::vector<Alignment> g_alignments(g_cam_ids.size());

/// thread
std::vector<mutex> g_sync_lock_cams(g_cam_ids.size());
mutex g_sync_lock_object;
mutex g_sync_lock_lidar;

/// params
bool g_is_compressed = false;

/// image
int g_image_w = camera::image_width;
int g_image_h = camera::image_height;
std::vector<cv::Mat> g_mats(g_cam_ids.size());

/// lidar
pcl::PointCloud<pcl::PointXYZI>::Ptr g_lidarall_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cam_front_60_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cam_top_front_120_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cam_left_60_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cam_right_60_ptr(new pcl::PointCloud<pcl::PointXYZI>);
boost::shared_ptr<pcl::visualization::PCLVisualizer> g_viewer(new pcl::visualization::PCLVisualizer("Cloud_Viewer"));
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_lidarall(g_lidarall_ptr, 255, 255, 255);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_front_60(g_cam_front_60_ptr, 255, 255, 0);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_top_front_120(g_cam_top_front_120_ptr, 255,
                                                                                         0, 0);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_left_60(g_cam_left_60_ptr, 0, 0, 255);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_right_60(g_cam_right_60_ptr, 0, 0, 255);
std::vector<pcl::visualization::Camera> g_cam;

/// object
std::vector<std::vector<msgs::DetectedObject>> g_objects(g_cam_ids.size());

//////////////////// for camera image
void callback_cam_front_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_cams[cam_order].lock();
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_mats[cam_order] = cv_ptr->image;
  std_msgs::Header h = msg->header;
  g_sync_lock_cams[cam_order].unlock();
}
void callback_cam_top_front_120(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::top_front_120);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_cams[cam_order].lock();
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_mats[cam_order] = cv_ptr->image;
  std_msgs::Header h = msg->header;
  g_sync_lock_cams[cam_order].unlock();
}

void callback_cam_left_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_cams[cam_order].lock();
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_mats[cam_order] = cv_ptr->image;
  std_msgs::Header h = msg->header;
  g_sync_lock_cams[cam_order].unlock();
}
void callback_cam_right_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_cams[cam_order].lock();
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_mats[cam_order] = cv_ptr->image;
  std_msgs::Header h = msg->header;
  g_sync_lock_cams[cam_order].unlock();
}

//////////////////// for camera image
void callback_cam_decode_front_60(sensor_msgs::CompressedImage compressImg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_cams[cam_order].lock();
  cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(g_mats[cam_order]);
  g_sync_lock_cams[cam_order].unlock();
}
void callback_cam_decode_top_front_120(sensor_msgs::CompressedImage compressImg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::top_front_120);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_cams[cam_order].lock();
  cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(g_mats[cam_order]);
  g_sync_lock_cams[cam_order].unlock();
}
void callback_cam_decode_left_60(sensor_msgs::CompressedImage compressImg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_cams[cam_order].lock();
  cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(g_mats[cam_order]);
  g_sync_lock_cams[cam_order].unlock();
}
void callback_cam_decode_right_60(sensor_msgs::CompressedImage compressImg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_cams[cam_order].lock();
  cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(g_mats[cam_order]);
  g_sync_lock_cams[cam_order].unlock();
}

//////////////////// for camera object
void callback_object_cam_front_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_object.lock();
  g_objects[cam_order] = msg->objects;
  g_sync_lock_object.unlock();
}
void callback_object_cam_top_front_120(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::top_front_120);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_object.lock();
  g_objects[cam_order] = msg->objects;
  g_sync_lock_object.unlock();
}
void callback_object_cam_left_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_object.lock();
  g_objects[cam_order] = msg->objects;
  g_sync_lock_object.unlock();
}
void callback_object_cam_right_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_object.lock();
  g_objects[cam_order] = msg->objects;
  g_sync_lock_object.unlock();
}
void callback_lidarall(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  g_sync_lock_lidar.lock();
  *g_lidarall_ptr = *msg;
  g_sync_lock_lidar.unlock();
  // std::cout << "Point cloud size: " << g_lidarall_ptr->size() << std::endl;
  // std::cout << "Lidar x: " << g_lidarall_ptr->points[0].x << ", y: " << g_lidarall_ptr->points[0].y << ", z: " <<
  // g_lidarall_ptr->points[0].z << std::endl;
}

void pclViewerInitializer()
{
  g_viewer->initCameraParameters();
  g_viewer->addCoordinateSystem(3.0, 0, 0, 0);       // Origin(0, 0, 0)
  g_viewer->setCameraPosition(0, 0, 20, 0.2, 0, 0);  // bird view
  g_viewer->setBackgroundColor(0, 0, 0);
  g_viewer->setShowFPS(false);
}

void cvViewerInitializer(std::vector<std::string> cam_topic_names)
{
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    cv::namedWindow(cam_topic_names[cam_order], cv::WINDOW_NORMAL);
    cv::resizeWindow(cam_topic_names[cam_order], 480, 360);
    cv::moveWindow(cam_topic_names[cam_order], 1025, 30);
  }
}

void alignmentInitializer()
{
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    /// alignment init
    g_alignments[cam_order].projectMatrixInit(g_cam_ids[cam_order]);
  }
}

void drawPointCloudOnImage()
{
  pcl::copyPointCloud(*g_lidarall_ptr, *g_cam_front_60_ptr);
  pcl::copyPointCloud(*g_lidarall_ptr, *g_cam_top_front_120_ptr);
  pcl::copyPointCloud(*g_lidarall_ptr, *g_cam_left_60_ptr);
  pcl::copyPointCloud(*g_lidarall_ptr, *g_cam_right_60_ptr);
  std::vector<pcl::PointCloud<pcl::PointXYZI>> cam_points = { *g_cam_front_60_ptr, *g_cam_top_front_120_ptr,
                                                              *g_cam_left_60_ptr, *g_cam_right_60_ptr };
  std::vector<int> cloud_sizes(g_cam_ids.size(), 0);

  for (size_t i = 0; i < g_lidarall_ptr->size(); i++)
  {
    if (g_lidarall_ptr->points[i].x > 0)
    {
      for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
      {
        PixelPosition pixel_position{ -1, -1 };
        pixel_position = g_alignments[cam_order].projectPointToPixel(g_lidarall_ptr->points[i]);
        if (pixel_position.u >= 0 && pixel_position.v >= 0)
        {
          cv::Point center_point = cv::Point(pixel_position.u, pixel_position.v);
          float distance_x = g_lidarall_ptr->points[i].x;
          cv::Scalar point_color = g_alignments[cam_order].getDistColor(distance_x);
          cv::circle(g_mats[cam_order], center_point, 1, point_color, -1, cv::LINE_8, 0);
          cam_points[cam_order].points[cloud_sizes[cam_order]] = cam_points[cam_order].points[i];
          cloud_sizes[cam_order]++;
          // std::cout << "Camera u: " << pixel_position.u << ", v: " << pixel_position.v << std::endl;
        }
      }
      // std::cout << "Lidar x: " << g_lidarall_ptr->points[i].x << ", y: " << g_lidarall_ptr->points[i].y <<
      // ", z: " << g_lidarall_ptr->points[i].z << std::endl;
    }
  }
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    cam_points[cam_order].resize(cloud_sizes[cam_order]);
  }
  *g_cam_front_60_ptr = cam_points[0];
  *g_cam_top_front_120_ptr = cam_points[1];
  *g_cam_left_60_ptr = cam_points[2];
  *g_cam_right_60_ptr = cam_points[3];
}
int main(int argc, char** argv)
{
  std::cout << "===== Alignment startup. =====" << std::endl;
  ros::init(argc, argv, "Alignment");
  ros::NodeHandle nh;

  /// camera subscriber
  std::vector<std::string> cam_topic_names(g_cam_ids.size());
  std::vector<std::string> bbox_topic_names(g_cam_ids.size());
  std::vector<ros::Subscriber> cam_subs(g_cam_ids.size());
  std::vector<ros::Subscriber> object_subs(g_cam_ids.size());
  static void (*f_callbacks_cam[])(const sensor_msgs::Image::ConstPtr&) = {
    callback_cam_front_60, callback_cam_top_front_120, callback_cam_left_60, callback_cam_right_60
  };
  static void (*f_callbacks_cam_decode[])(
      sensor_msgs::CompressedImage) = { callback_cam_decode_front_60, callback_cam_decode_top_front_120,
                                        callback_cam_decode_left_60, callback_cam_decode_right_60 };
  static void (*f_callbacks_object[])(
      const msgs::DetectedObjectArray::ConstPtr&) = { callback_object_cam_front_60, callback_object_cam_top_front_120,
                                                      callback_object_cam_left_60, callback_object_cam_right_60 };

  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    /// new subscriber
    cam_topic_names[cam_order] = camera::topics[g_cam_ids[cam_order]];
    bbox_topic_names[cam_order] = camera::topics_obj[g_cam_ids[cam_order]];

    if (g_is_compressed)
    {
      cam_subs[cam_order] =
          nh.subscribe(cam_topic_names[cam_order] + std::string("/compressed"), 1, f_callbacks_cam_decode[cam_order]);
    }
    else
    {
      cam_subs[cam_order] = nh.subscribe(cam_topic_names[cam_order], 1, f_callbacks_cam[cam_order]);
    }
    object_subs[cam_order] = nh.subscribe(bbox_topic_names[cam_order], 1, f_callbacks_object[cam_order]);
  }

  /// lidar subscriber
  ros::Subscriber lidarall;
  lidarall = nh.subscribe("/LidarAll", 1, callback_lidarall);

  /// viwer init
  pclViewerInitializer();
  cvViewerInitializer(cam_topic_names);

  /// class init
  alignmentInitializer();

  /// main loop
  ros::Rate loop_rate(30);
  std::cout << "===== Alignment running... =====" << std::endl;
  while (ros::ok())
  {
    /// remove points on pcl viewer
    g_viewer->removePointCloud("Cloud viewer");
    g_viewer->removePointCloud("Front 60 Cloud viewer");
    g_viewer->removePointCloud("Top Front 120 Cloud viewer");
    g_viewer->removePointCloud("Left 60 Cloud viewer");
    g_viewer->removePointCloud("Right 60 Cloud viewer");

    g_sync_lock_lidar.lock();  // mutex lidar
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      g_sync_lock_cams[cam_order].lock();  // mutex camera
      if (!g_mats[cam_order].empty())
      {
        /// draw lidarall on cv viewer
        drawPointCloudOnImage();
        cv::imshow(cam_topic_names[cam_order], g_mats[cam_order]);
      }
      g_sync_lock_cams[cam_order].unlock();  // mutex camera
    }
    /// draw points on pcl viewer
    g_viewer->addPointCloud<pcl::PointXYZI>(g_lidarall_ptr, g_rgb_lidarall, "Cloud viewer");
    g_viewer->addPointCloud<pcl::PointXYZI>(g_cam_front_60_ptr, g_rgb_cam_front_60, "Front 60 Cloud viewer");
    g_viewer->addPointCloud<pcl::PointXYZI>(g_cam_top_front_120_ptr, g_rgb_cam_top_front_120, "Top Front 120 Cloud "
                                                                                              "viewer");
    g_viewer->addPointCloud<pcl::PointXYZI>(g_cam_left_60_ptr, g_rgb_cam_left_60, "Left 60 Cloud viewer");
    g_viewer->addPointCloud<pcl::PointXYZI>(g_cam_right_60_ptr, g_rgb_cam_right_60, "Right 60 Cloud viewer");

    g_sync_lock_lidar.unlock();  // mutex lidar
    cv::waitKey(1);

    ros::spinOnce();
    g_viewer->spinOnce();
    loop_rate.sleep();
  }
  std::cout << "===== Alignment shutdown. =====" << std::endl;
  return 0;
}
