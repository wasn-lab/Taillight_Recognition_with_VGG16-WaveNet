/// standard
#include <iostream>
#include <thread>

/// ros
#include "ros/ros.h"
#include <msgs/DetectedObjectArray.h>
#include <msgs/DetectedObject.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/cache.h>

/// package
#include "camera_params.h"
#include "fusion_source_id.h"
#include "alignment.h"
#include "visualization_util.h"
#include <drivenet/object_label_util.h>
#include "point_preprocessing.h"
#include "ssn_util.h"
#include "points_in_image_area.h"
#include "sync_message.h"
#include "object_generator.h"
#include "cloud_cluster.h"
#include "UserDefine.h" // CLUSTER_INFO struct

/// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/common.h>
#include <pcl/filters/conditional_removal.h>

/// thread
#include <mutex>

/// namespace
using namespace DriveNet;

/// camera layout
#if CAR_MODEL_IS_B1_V2
const std::vector<camera::id> g_cam_ids{ camera::id::front_bottom_60, camera::id::front_top_far_30,
                                         camera::id::right_back_60, camera::id::left_back_60,
                                         camera::id::right_front_60, camera::id::left_front_60 };
#else
#error "car model is not well defined"
#endif

/// class
std::vector<Alignment> g_alignments(g_cam_ids.size());
Visualization g_visualization;
ObjectGenerator g_object_generator;
std::vector<CloudCluster> g_cloud_clusters(g_cam_ids.size());

/// thread
std::vector<std::mutex> g_mutex_cams(g_cam_ids.size());
std::mutex g_mutex_cams_process;
std::mutex g_mutex_objects;
std::mutex g_mutex_objects_process;
std::mutex g_mutex_lidar_raw;
std::mutex g_mutex_lidar_nonground;
std::mutex g_mutex_lidar_process;
std::mutex g_mutex_lidar_ssn;
std::mutex g_mutex_cams_points;
std::mutex g_mutex_objects_points;
std::recursive_mutex g_mutex_cam_times;
std::recursive_mutex g_mutex_lidar_time;
std::recursive_mutex g_mutex_lidar_nonground_time;
std::recursive_mutex g_mutex_lidar_ssn_time;
std::vector<std::mutex> g_mutex_cam_time(g_cam_ids.size());
std::mutex g_mutex_cube;
std::mutex g_mutex_polygon;

/// params
bool g_is_enable_default_3d_bbox = true;
bool g_do_clustering = false;
bool g_data_sync = true;  // trun on or trun off data sync function
bool g_is_display = false;

/// inference params
bool g_is_data_sync = false;
std::vector<bool> g_is_object_update(g_cam_ids.size());

/// ros
std::vector<message_filters::Cache<sensor_msgs::Image>> g_cache_image(g_cam_ids.size());
message_filters::Cache<pcl::PointCloud<pcl::PointXYZI>> g_cache_lidarall;
message_filters::Cache<pcl::PointCloud<pcl::PointXYZI>> g_cache_lidarall_nonground;
message_filters::Cache<pcl::PointCloud<pcl::PointXYZIL>> g_cache_lidar_ssn;
std::vector<std::string> g_cam_topic_names(g_cam_ids.size());
std::vector<std::string> g_bbox_topic_names(g_cam_ids.size());

/// image
int g_image_w = camera::image_width;
int g_image_h = camera::image_height;
int g_raw_image_w = camera::raw_image_width;
int g_raw_image_h = camera::raw_image_height;
std::vector<cv::Mat> g_mats(g_cam_ids.size());
std::vector<cv::Mat> g_mats_process(g_cam_ids.size());

/// lidar
pcl::PointCloud<pcl::PointXYZI>::Ptr g_lidarall_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_lidarall_nonground_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_lidarall_ptr_process(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZIL>::Ptr g_lidar_ssn_ptr(new pcl::PointCloud<pcl::PointXYZIL>);
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> g_cams_points_ptr(g_cam_ids.size());
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> g_cams_bbox_points_ptr(g_cam_ids.size());
std::vector<pcl::visualization::Camera> g_cam;

/// object
ros::Publisher g_object_pub;
std::vector<msgs::DetectedObjectArray> g_object_arrs(g_cam_ids.size());
std::vector<msgs::DetectedObjectArray> g_object_arrs_process(g_cam_ids.size());
int g_object_wait_frame = 5;
std::vector<std::vector<msgs::DetectedObjectArray>> g_object_buffer_arrs(g_cam_ids.size());

/// sync camera and lidar
int g_buffer_size = 180;
std::vector<std::vector<ros::Time>> g_cam_times(g_cam_ids.size());
std::vector<std::vector<ros::Time>> g_cam_single_times(g_cam_ids.size());
std::vector<ros::Time> g_lidarall_times;
std::vector<ros::Time> g_lidarall_nonground_times;
std::vector<ros::Time> g_lidar_ssn_times;
std::vector<ros::Time> g_lidarall_time_buffer;
std::vector<ros::Time> g_lidarall_nonground_time_buffer;
std::vector<ros::Time> g_lidar_ssn_time_buffer;

/// 3d cube
// std::vector<std::vector<MinMax3D>> g_cams_bboxs_cube_min_max(g_cam_ids.size()); // bbox - pcl
std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>>> g_cams_bboxs_points(g_cam_ids.size());

//////////////////// for camera image
void callback_cam_front_bottom_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_bottom_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  if (!g_data_sync)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    std::lock_guard<std::mutex> lock_cams(g_mutex_cams[cam_order]);
    g_mats[cam_order] = cv_ptr->image;
  }
  else
  {
    std::lock_guard<std::mutex> lock_cam_time(g_mutex_cam_time[cam_order]);
    g_cam_single_times[cam_order].push_back(msg->header.stamp);
    // std::cout << "camera time: " << g_cam_single_time[cam_order].sec << "." <<
    // g_cam_single_time[cam_order].nsec <<
    // std::endl;
  }
}

void callback_cam_front_top_far_30(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_top_far_30);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  if (!g_data_sync)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    std::lock_guard<std::mutex> lock_cams(g_mutex_cams[cam_order]);
    g_mats[cam_order] = cv_ptr->image;
  }
  else
  {
    std::lock_guard<std::mutex> lock_cam_time(g_mutex_cam_time[cam_order]);
    g_cam_single_times[cam_order].push_back(msg->header.stamp);
    // std::cout << "camera time: " << g_cam_single_time[cam_order].sec << "." <<
    // g_cam_single_time[cam_order].nsec <<
    // std::endl;
  }
}

void callback_cam_right_back_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_back_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  if (!g_data_sync)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    std::lock_guard<std::mutex> lock_cams(g_mutex_cams[cam_order]);
    g_mats[cam_order] = cv_ptr->image;
  }
  else
  {
    std::lock_guard<std::mutex> lock_cam_time(g_mutex_cam_time[cam_order]);
    g_cam_single_times[cam_order].push_back(msg->header.stamp);
    // std::cout << "camera time: " << g_cam_single_time[cam_order].sec << "." <<
    // g_cam_single_time[cam_order].nsec <<
    // std::endl;
  }
}

void callback_cam_left_back_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_back_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  if (!g_data_sync)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    std::lock_guard<std::mutex> lock_cams(g_mutex_cams[cam_order]);
    g_mats[cam_order] = cv_ptr->image;
  }
  else
  {
    std::lock_guard<std::mutex> lock_cam_time(g_mutex_cam_time[cam_order]);
    g_cam_single_times[cam_order].push_back(msg->header.stamp);
    // std::cout << "camera time: " << g_cam_single_time[cam_order].sec << "." <<
    // g_cam_single_time[cam_order].nsec <<
    // std::endl;
  }
}

void callback_cam_right_front_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_front_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  if (!g_data_sync)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    std::lock_guard<std::mutex> lock_cams(g_mutex_cams[cam_order]);
    g_mats[cam_order] = cv_ptr->image;
  }
  else
  {
    std::lock_guard<std::mutex> lock_cam_time(g_mutex_cam_time[cam_order]);
    g_cam_single_times[cam_order].push_back(msg->header.stamp);
    // std::cout << "camera time: " << g_cam_single_time[cam_order].sec << "." <<
    // g_cam_single_time[cam_order].nsec <<
    // std::endl;
  }
}

void callback_cam_left_front_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_front_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  if (!g_data_sync)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    std::lock_guard<std::mutex> lock_cams(g_mutex_cams[cam_order]);
    g_mats[cam_order] = cv_ptr->image;
  }
  else
  {
    std::lock_guard<std::mutex> lock_cam_time(g_mutex_cam_time[cam_order]);
    g_cam_single_times[cam_order].push_back(msg->header.stamp);
    // std::cout << "camera time: " << g_cam_single_time[cam_order].sec << "." <<
    // g_cam_single_time[cam_order].nsec <<
    // std::endl;
  }
}
//////////////////// for camera object
void callback_object_cam_front_bottom_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_bottom_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  static int object_wait_frame_count = 0;
  
  std::lock_guard<std::mutex> lock_objects(g_mutex_objects);
  if (object_wait_frame_count < g_object_wait_frame)
  {
    g_object_buffer_arrs[cam_order].push_back(*msg);
    object_wait_frame_count = object_wait_frame_count + 1;
  }
  else
  {
    g_object_buffer_arrs[cam_order].push_back(*msg);	 
    g_object_arrs[cam_order] = g_object_buffer_arrs[cam_order].front();
    g_is_object_update[cam_order] = true;
    g_object_buffer_arrs[cam_order].erase(g_object_buffer_arrs[cam_order].begin());     
  }
  // std::cout << "camera object: " << msg->header.stamp.sec << "." << msg->header.stamp.nsec <<
  // std::endl;
}

void callback_object_cam_front_top_far_30(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_top_far_30);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  static int object_wait_frame_count = 0;
  
  std::lock_guard<std::mutex> lock_objects(g_mutex_objects);
  if (object_wait_frame_count < g_object_wait_frame)
  {
    g_object_buffer_arrs[cam_order].push_back(*msg);
    object_wait_frame_count = object_wait_frame_count + 1;
  }
  else
  {
    g_object_buffer_arrs[cam_order].push_back(*msg);	 
    g_object_arrs[cam_order] = g_object_buffer_arrs[cam_order].front();
    g_is_object_update[cam_order] = true;
    g_object_buffer_arrs[cam_order].erase(g_object_buffer_arrs[cam_order].begin());     
  }
  // std::cout << "camera object: " << msg->header.stamp.sec << "." << msg->header.stamp.nsec <<
  // std::endl;
}

void callback_object_cam_right_back_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_back_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  static int object_wait_frame_count = 0;
  
  std::lock_guard<std::mutex> lock_objects(g_mutex_objects);
  if (object_wait_frame_count < g_object_wait_frame)
  {
    g_object_buffer_arrs[cam_order].push_back(*msg);
    object_wait_frame_count = object_wait_frame_count + 1;
  }
  else
  {
    g_object_buffer_arrs[cam_order].push_back(*msg);	 
    g_object_arrs[cam_order] = g_object_buffer_arrs[cam_order].front();
    g_is_object_update[cam_order] = true;
    g_object_buffer_arrs[cam_order].erase(g_object_buffer_arrs[cam_order].begin());     
  }
  // std::cout << "camera object: " << msg->header.stamp.sec << "." << msg->header.stamp.nsec <<
  // std::endl;
}


void callback_object_cam_left_back_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_back_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  static int object_wait_frame_count = 0;
  
  std::lock_guard<std::mutex> lock_objects(g_mutex_objects);
  if (object_wait_frame_count < g_object_wait_frame)
  {
    g_object_buffer_arrs[cam_order].push_back(*msg);
    object_wait_frame_count = object_wait_frame_count + 1;
  }
  else
  {
    g_object_buffer_arrs[cam_order].push_back(*msg);	 
    g_object_arrs[cam_order] = g_object_buffer_arrs[cam_order].front();
    g_is_object_update[cam_order] = true;
    g_object_buffer_arrs[cam_order].erase(g_object_buffer_arrs[cam_order].begin());     
  }
  // std::cout << "camera object: " << msg->header.stamp.sec << "." << msg->header.stamp.nsec <<
  // std::endl;
}

void callback_object_cam_right_front_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_front_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  static int object_wait_frame_count = 0;
  
  std::lock_guard<std::mutex> lock_objects(g_mutex_objects);
  if (object_wait_frame_count < g_object_wait_frame)
  {
    g_object_buffer_arrs[cam_order].push_back(*msg);
    object_wait_frame_count = object_wait_frame_count + 1;
  }
  else
  {
    g_object_buffer_arrs[cam_order].push_back(*msg);	 
    g_object_arrs[cam_order] = g_object_buffer_arrs[cam_order].front();
    g_is_object_update[cam_order] = true;
    g_object_buffer_arrs[cam_order].erase(g_object_buffer_arrs[cam_order].begin());     
  }
  // std::cout << "camera object: " << msg->header.stamp.sec << "." << msg->header.stamp.nsec <<
  // std::endl;
}

void callback_object_cam_left_front_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_front_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  static int object_wait_frame_count = 0;
  
  std::lock_guard<std::mutex> lock_objects(g_mutex_objects);
  if (object_wait_frame_count < g_object_wait_frame)
  {
    g_object_buffer_arrs[cam_order].push_back(*msg);
    object_wait_frame_count = object_wait_frame_count + 1;
  }
  else
  {
    g_object_buffer_arrs[cam_order].push_back(*msg);	 
    g_object_arrs[cam_order] = g_object_buffer_arrs[cam_order].front();
    g_is_object_update[cam_order] = true;
    g_object_buffer_arrs[cam_order].erase(g_object_buffer_arrs[cam_order].begin());     
  }
  // std::cout << "camera object: " << msg->header.stamp.sec << "." << msg->header.stamp.nsec <<
  // std::endl;
}
//////////////////// for LiDAR data
void callback_lidarall(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  if (!g_data_sync)
  {
    std::lock_guard<std::mutex> lock(g_mutex_lidar_raw);
    *g_lidarall_ptr = *msg;
    // std::cout << "lidarall: " << msg->header.stamp.sec << "." << msg->header.stamp.nsec <<
    // std::endl;
  }
  else
  {
    ros::Time header_time;
    pcl_conversions::fromPCL(msg->header.stamp, header_time);
    std::lock_guard<std::recursive_mutex> lock_lidar_time(g_mutex_lidar_time);
    g_lidarall_time_buffer.push_back(header_time);  
    // std::cout << "lidarall: " << header_time.sec << "." << header_time.nsec <<
    // std::endl;
  }
  // std::cout << "Point cloud size: " << g_lidarall_ptr->size() << std::endl;
  // std::cout << "Lidar x: " << g_lidarall_ptr->points[0].x << ", y: " << g_lidarall_ptr->points[0].y << ", z: " <<
  // g_lidarall_ptr->points[0].z << std::endl;
}

void callback_lidarall_nonground(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  if (!g_data_sync)
  {
    std::lock_guard<std::mutex> lock(g_mutex_lidar_nonground);
    *g_lidarall_nonground_ptr = *msg;
  }
  else
  {
    ros::Time header_time;
    pcl_conversions::fromPCL(msg->header.stamp, header_time);
    std::lock_guard<std::recursive_mutex> lock_lidar_nonground_time(g_mutex_lidar_nonground_time);
    g_lidarall_nonground_time_buffer.push_back(header_time);
    // ROS_INFO("lidarall nonground timestamp: %d.%d", header_time.sec, header_time.nsec);
  }
}

void callback_ssn(const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr& msg)
{
  if (!g_data_sync)
  {
    std::lock_guard<std::mutex> lock(g_mutex_lidar_ssn);
    *g_lidar_ssn_ptr = *msg;
  }
  else
  {
    ros::Time header_time;
    pcl_conversions::fromPCL(msg->header.stamp, header_time);
    std::lock_guard<std::recursive_mutex> lock_lidar_ssn_time(g_mutex_lidar_ssn_time);
    g_lidar_ssn_time_buffer.push_back(header_time);
    // std::cout << "lidar ssn: " << header_time.sec << "." << header_time.nsec <<
    // std::endl;
  }
}

void object_publisher(std::vector<msgs::DetectedObjectArray>& objects_2d_bbox,
                       std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>>>& cams_bboxs_points,
                       /*std::vector<std::vector<MinMax3D>>& cams_bboxs_cube_min_max,*/ std_msgs::Header msg_header)
{
  msgs::DetectedObjectArray msg_det_obj_arr;
  std::vector<msgs::DetectedObject> msg_objs;
  float min_z = -3;
  float max_z = -1.5;

#pragma omp parallel for
  for (size_t cam_order = 0; cam_order < cams_bboxs_points.size(); cam_order++)
  {
    for (size_t obj_index = 0; obj_index < cams_bboxs_points[cam_order].size(); obj_index++)
    {
      msgs::DetectedObject msg_obj;
      msg_obj.header = objects_2d_bbox[cam_order].objects[obj_index].header;
      msg_obj.classId = objects_2d_bbox[cam_order].objects[obj_index].classId;
      msg_obj.camInfo = objects_2d_bbox[cam_order].objects[obj_index].camInfo;
      msg_obj.fusionSourceId = sensor_msgs_itri::FusionSourceId::Camera;
      msg_obj.distance = 0;
      pcl::PointCloud<pcl::PointXYZI> points = cams_bboxs_points[cam_order][obj_index];

      /// bbox- pcl
      // MinMax3D cube = cams_bboxs_cube_min_max[cam_order][obj_index];
      // msg_obj.bPoint = g_object_generator.minMax3dToBBox(cube);

      /// bbox- L-shape
      msgs::BoxPoint box_point;
      box_point = g_object_generator.pointsToLShapeBBox(points, msg_obj.classId);
      if (!(box_point.p0.x == 0 && box_point.p0.y == 0 && box_point.p0.z == 0
          && box_point.p6.x == 0 && box_point.p6.y == 0 && box_point.p6.z == 0))
      {
        msg_obj.bPoint = box_point;
      }
      else
      {
        /// bbox - pcl
        MinMax3D cube_min_max;  // object min and max point
        pcl::getMinMax3D(points, cube_min_max.p_min, cube_min_max.p_max);
        msg_obj.bPoint = g_object_generator.minMax3dToBBox(cube_min_max);
      }
      
      /// polygon - ApproxMVBB
      pcl::PointCloud<pcl::PointXYZ> convex_points;
      convex_points = g_object_generator.pointsToPolygon(points);

      /// polygon to DetectedObj.cPoint
      if (!convex_points.empty())
      {
        msg_obj.cPoint.objectHigh = max_z - min_z;
        for (auto& point : convex_points)
        {
          msgs::PointXYZ convex_point;
          convex_point.x = point.x;
          convex_point.y = point.y;
          convex_point.z = min_z;
          msg_obj.cPoint.lowerAreaPoints.push_back(convex_point);
        }
#pragma omp critical
        {
          msg_objs.push_back(msg_obj);
        }
      }
    }
  }
  msg_det_obj_arr.header = std::move(msg_header);
  msg_det_obj_arr.header.frame_id = "lidar";  // mapping to lidar coordinate
  msg_det_obj_arr.objects = msg_objs;
  g_object_pub.publish(msg_det_obj_arr);
}

void pclViewerInitializer(const boost::shared_ptr<pcl::visualization::PCLVisualizer>& pcl_viewer) /*,
                           std::vector<std::string> window_name, int window_count = 3)*/
{
  // if (window_name.size() < 3)
  // {
  //   window_name.clear();
  //   window_name.emplace_back("raw_data");
  //   window_name.emplace_back("image fov");
  //   window_name.emplace_back("object");
  // }
  // if (window_count < 3)
  // {
  //   window_count = 3;
  // }

  // int v1 = 1, v2 = 2, v3 = 3;
  // pcl_viewer->createViewPort(0.0, 0.0, 0.33, 1.0, v1);
  // pcl_viewer->createViewPort(0.33, 0.0, 0.66, 1.0, v2);
  // pcl_viewer->createViewPort(0.66, 0.0, 1.0, 1.0, v3);
  pcl_viewer->initCameraParameters();
  pcl_viewer->addCoordinateSystem(3.0, 0, 0, 0);
  pcl_viewer->setCameraPosition(0, 0, 20, 0.2, 0, 0);
  pcl_viewer->setShowFPS(false);
  // for (int count = 1; count < window_count + 1; count++)
  // {
  //   pcl_viewer->setBackgroundColor(0, 0, 0, count);
  //   pcl_viewer->addText(window_name[count - 1], 10, 10, window_name[count - 1], count);
  // }
}

void cvViewerInitializer()
{
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    cv::namedWindow(g_cam_topic_names[cam_order], cv::WINDOW_NORMAL);
    cv::resizeWindow(g_cam_topic_names[cam_order], 360, 270);
    cv::moveWindow(g_cam_topic_names[cam_order], 380 * cam_order, 30);
  }
}

void pclInitializer(std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& points_ptr)
{
  for (auto& point_ptr : points_ptr)
  {
    point_ptr = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  }
}

void pointsColorInit(std::vector<pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>>& rgb_points,
                     std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> points_ptr)
{
  cv::Scalar point_color = CvColor::white_;
  for (size_t index = 0; index < points_ptr.size(); index++)
  {
    point_color = intToColor(static_cast<int>(index));
    rgb_points.emplace_back(pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>(
        points_ptr[index], point_color[0], point_color[1], point_color[2]));
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

void drawBoxOnImages(std::vector<cv::Mat>& mats, std::vector<msgs::DetectedObjectArray>& objects)
{
  // std::cout << "===== drawBoxOnImages... =====" << std::endl;
  for (size_t cam_order = 0; cam_order < mats.size(); cam_order++)
  {
    g_visualization.drawBoxOnImage(mats[cam_order], objects[cam_order].objects);
  }
}
void drawPointCloudOnImages(std::vector<cv::Mat>& mats, std::vector<std::vector<PixelPosition>>& cam_pixels,
                            std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_points_ptr)
{
  // std::cout << "===== drawPointCloudOnImages... =====" << std::endl;
  pcl::PointCloud<pcl::PointXYZI> point_cloud;
  for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
  {
    point_cloud = *cams_points_ptr[cam_order];
    for (size_t i = 0; i < point_cloud.size(); i++)
    {
      int point_u = cam_pixels[cam_order][i].u;
      int point_v = cam_pixels[cam_order][i].v;
      float point_x = point_cloud[i].x;
      g_visualization.drawPointCloudOnImage(mats[cam_order], point_u, point_v, point_x);
    }
  }
}

void getPointCloudInAllImageFOV(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidarall_ptr,
                                std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_points_ptr,
                                /*std::vector<std::vector<PixelPosition>>& cam_pixels,*/ int image_w, int image_h)
{
// std::cout << "===== getPointCloudInImageFOV... =====" << std::endl;
#pragma omp parallel for
  for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
  {
    getPointCloudInImageFOV(lidarall_ptr, cams_points_ptr[cam_order] /*, cam_pixels[cam_order]*/, image_w, image_h,
                            g_alignments[cam_order]);
  }
}

void getPointCloudInAllBoxFOV(const std::vector<msgs::DetectedObjectArray>& objects,
                              std::vector<msgs::DetectedObjectArray>& remaining_objects,
                              const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_points_ptr,
                              std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_bbox_points_ptr,
                              std::vector<std::vector<PixelPosition>>& cam_pixels,
                              std::vector<msgs::DetectedObjectArray>& objects_2d_bbox,
                              /*std::vector<std::vector<MinMax3D>>& cams_bboxs_cube_min_max,*/// bbox - pcl
                              std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>>>& cams_bboxs_points)
{
// std::cout << "===== getPointCloudInAllBoxFOV... =====" << std::endl;
#pragma omp parallel for
  for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
  {
    getPointCloudInBoxFOV(objects[cam_order], remaining_objects[cam_order], cams_points_ptr[cam_order],
                          cams_bbox_points_ptr[cam_order], cam_pixels[cam_order], objects_2d_bbox[cam_order],
                          /*cams_bboxs_cube_min_max[cam_order],*/ cams_bboxs_points[cam_order], g_alignments[cam_order],
                          g_cloud_clusters[cam_order], g_is_enable_default_3d_bbox, g_do_clustering);
  }
}
void getPointCloudInAllBoxFOV(const std::vector<msgs::DetectedObjectArray>& remaining_objects,
                              const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_points_ptr,
                              std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_bbox_points_ptr,
                              std::vector<std::vector<PixelPosition>>& cam_pixels,
                              std::vector<msgs::DetectedObjectArray>& objects_2d_bbox,
                              /*std::vector<std::vector<MinMax3D>>& cams_bboxs_cube_min_max,*/ // bbox - pcl
                              std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>>>& cams_bboxs_points)
{
// std::cout << "===== getPointCloudInAllBoxFOV... =====" << std::endl;
#pragma omp parallel for
  for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
  {
    getPointCloudInBoxFOV(remaining_objects[cam_order], cams_points_ptr[cam_order], cams_bbox_points_ptr[cam_order],
                          cam_pixels[cam_order], objects_2d_bbox[cam_order], /*cams_bboxs_cube_min_max[cam_order],*/
                          cams_bboxs_points[cam_order], g_alignments[cam_order], g_cloud_clusters[cam_order],
                          g_is_enable_default_3d_bbox, g_do_clustering);
  }
}
template <typename T>
void release(std::vector<T>& input_vector)
{
  for (size_t index = 0; index < input_vector.size(); index++)
  {
    input_vector[index].clear();
  }
}
void displayLidarData()
{
  std::cout << "===== displayLidarData... =====" << std::endl;
  /// create variable
  boost::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer(
      new pcl::visualization::PCLVisualizer("Cloud_Viewer"));
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidarall_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_points_ptr(g_cams_points_ptr.size());
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_bbox_points_ptr(g_cams_bbox_points_ptr.size());
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb_lidarall(g_lidarall_ptr_process, 255, 255, 255);
  std::vector<pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>> rgb_cams_points;
  std::vector<pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>> rgb_cams_bbox_points;
  // std::vector<int> viewports{ 1, 2, 3 };
  // std::vector<std::string> view_name{ "raw data", "image fov", "object" };

  /// init
  pclViewerInitializer(pcl_viewer);  //, view_name, static_cast<int>(viewports.size()));
  pclInitializer(cams_points_ptr);
  pclInitializer(cams_bbox_points_ptr);
  pointsColorInit(rgb_cams_points, g_cams_points_ptr);
  pointsColorInit(rgb_cams_bbox_points, g_cams_bbox_points_ptr);

  MinMax3D point_50m, point_40m, point_30m, point_20m, point_10m;
  cv::Scalar color_50m, color_40m, color_30m, color_20m, color_10m;
  float x_dist = 50;
  float y_dist = 50;
  float z_dist = -3;
  point_50m = g_visualization.getDistLinePoint(x_dist, y_dist, z_dist);
  color_50m = g_visualization.getDistColor(x_dist);
  x_dist -= 10;
  point_40m = g_visualization.getDistLinePoint(x_dist, y_dist, z_dist);
  color_40m = g_visualization.getDistColor(x_dist);
  x_dist -= 10;
  point_30m = g_visualization.getDistLinePoint(x_dist, y_dist, z_dist);
  color_30m = g_visualization.getDistColor(x_dist);
  x_dist -= 10;
  point_20m = g_visualization.getDistLinePoint(x_dist, y_dist, z_dist);
  color_20m = g_visualization.getDistColor(x_dist);
  x_dist -= 10;
  point_10m = g_visualization.getDistLinePoint(x_dist, y_dist, z_dist);
  color_10m = g_visualization.getDistColor(x_dist);

  /// main loop
  ros::Rate loop_rate(10);
  while (ros::ok() && !pcl_viewer->wasStopped())
  {
    /// remove points on pcl viewer
    pcl_viewer->removePointCloud("Cloud viewer");  //, viewports[0]);
    // for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    // {
    //   pcl_viewer->removePointCloud(g_cam_topic_names[cam_order], viewports[1]);
    //   pcl_viewer->removePointCloud(g_bbox_topic_names[cam_order], viewports[2]);
    // }
    pcl_viewer->removeAllShapes();

    /// draw points on pcl viewer
    std::lock_guard<std::mutex> lock_lidar_process(g_mutex_lidar_process);
    pcl_viewer->addPointCloud<pcl::PointXYZI>(g_lidarall_ptr_process, rgb_lidarall, "Cloud viewer");  //, viewports[0]);

    pcl_viewer->addLine<pcl::PointXYZI>(point_50m.p_min, point_50m.p_max, color_50m[2], color_50m[1], color_50m[0],
                                        "line-50m");
    pcl_viewer->addLine<pcl::PointXYZI>(point_40m.p_min, point_40m.p_max, color_40m[2], color_40m[1], color_40m[0],
                                        "line-40m");
    pcl_viewer->addLine<pcl::PointXYZI>(point_30m.p_min, point_30m.p_max, color_30m[2], color_30m[1], color_30m[0],
                                        "line-30m");
    pcl_viewer->addLine<pcl::PointXYZI>(point_20m.p_min, point_20m.p_max, color_20m[2], color_20m[1], color_20m[0],
                                        "line-20m");
    pcl_viewer->addLine<pcl::PointXYZI>(point_10m.p_min, point_10m.p_max, color_10m[2], color_10m[1], color_10m[0],
                                        "line-10m");

    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      // std::lock_guard<std::mutex> lock_cams_points(g_mutex_cams_points); // mutex camera points
      // pcl_viewer->addPointCloud<pcl::PointXYZI>(g_cams_points_ptr[cam_order], rgb_cams_points[cam_order],
      //                                           g_cam_topic_names[cam_order], viewports[1]);
      // std::lock_guard<std::mutex> lock_objects_points(g_mutex_objects_points); // mutex objects points
      // pcl_viewer->addPointCloud<pcl::PointXYZI>(g_cams_bbox_points_ptr[cam_order], rgb_cams_bbox_points[cam_order],
      //                                           g_bbox_topic_names[cam_order], viewports[2]);

      /// bbox - pcl
      // std::lock_guard<std::mutex> lock_cube(g_mutex_cube);  // mutex camera cube
      // if (!g_cams_bboxs_cube_min_max[cam_order].empty())
      // {
      //   int cube_cout = 0;
      //   for (const auto& cube : g_cams_bboxs_cube_min_max[cam_order])
      //   {
      //     std::string cube_id = "cube_cam" + std::to_string(cam_order) + "_" + std::to_string(cube_cout);
      //     cv::Scalar cube_color = CvColor::white_;
      //     cube_color = intToColor(static_cast<int>(cam_order));

      //     pcl_viewer->addCube(cube.p_min.x, cube.p_max.x, cube.p_min.y, cube.p_max.y, cube.p_min.z, cube.p_max.z,
      //                         cube_color[0], cube_color[1], cube_color[2], cube_id);  //, viewports[0]);
      //     pcl_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
      //                                             pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, cube_id);
      //     cube_cout++;
      //   }
      // }
      std::lock_guard<std::mutex> lock_polygon(g_mutex_polygon);  // mutex camera polygon
      if (!g_cams_bboxs_points[cam_order].empty())
      {
        int polygon_cout = 0;
        for (const auto& points : g_cams_bboxs_points[cam_order])
        {
          pcl::PointCloud<pcl::PointXYZI>::Ptr points_ptr = points.makeShared();
          std::string polygon_id = "polygon_cam" + std::to_string(cam_order) + "_" + std::to_string(polygon_cout);
          cv::Scalar polygon_color = CvColor::white_;
          polygon_color = intToColor(static_cast<int>(cam_order));
          pcl_viewer->addPolygon<pcl::PointXYZI>(points_ptr, 1, 0, 0, polygon_id);  //, viewports[0]);
          polygon_cout++;
        }
      }
    }
    pcl_viewer->spinOnce();
    loop_rate.sleep();
  }
  std::cout << "===== displayLidarData close =====" << std::endl;
}
void displayCameraData()
{
  std::cout << "===== displayCameraData... =====" << std::endl;
  /// init
  cvViewerInitializer();

  /// main loop
  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      if (!g_mats_process[cam_order].empty())
      {
        std::lock_guard<std::mutex> lock_cams_process(g_mutex_cams_process);  // mutex camera
        cv::imshow(g_cam_topic_names[cam_order], g_mats_process[cam_order]);
      }
    }
    cv::waitKey(1);
    loop_rate.sleep();
  }
  std::cout << "===== displayCameraData close =====" << std::endl;
}

void getSyncLidarCameraData()
{
  std::cout << "getSyncLidarCameraData start." << std::endl;
  bool is_camera_update = false;
  bool is_lidar_update = false;
  bool is_lidarall_nonground_update = false;
  bool is_lidar_ssn_update = false;
  std::vector<std::vector<ros::Time>> cam_times_tmp(g_cam_ids.size());
  std::vector<ros::Time> lidarall_times_tmp;
  std::vector<ros::Time> lidarall_nonground_times_tmp;
  std::vector<ros::Time> lidar_ssn_times_tmp;
  std::vector<ros::Time> objects_time(g_cam_ids.size());
  ros::Time object_past_time = ros::Time(0);
  ros::Duration duration_time(3);

  ros::Rate loop_rate(20);
  while (ros::ok())
  {
    if (!g_cam_times[0].empty() && !g_lidarall_times.empty() && !g_lidarall_nonground_times.empty() &&
        !g_lidar_ssn_times.empty() && !g_is_data_sync)
    {
      if (g_is_object_update[0])
      {
        is_camera_update = false;
        is_lidar_update = false;
        is_lidarall_nonground_update = false;
        is_lidar_ssn_update = false;
        g_is_object_update[0] = false;

        // message sync
        std::lock_guard<std::mutex> lock_objects(g_mutex_objects);
        for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
        {
          objects_time[cam_order] = g_object_arrs[cam_order].header.stamp;
        }
        std::lock_guard<std::mutex> lock_objects_process(g_mutex_objects_process);
        g_object_arrs_process = g_object_arrs;

        // copy header timestamp vector
        std::lock_guard<std::recursive_mutex> lock_cam_times(g_mutex_cam_times);
        cam_times_tmp = g_cam_times;

        std::lock_guard<std::recursive_mutex> lock_lidar_time(g_mutex_lidar_time);
        lidarall_times_tmp = g_lidarall_times;

        std::lock_guard<std::recursive_mutex> lock_lidar_nonground_time(g_mutex_lidar_nonground_time);
        lidarall_nonground_times_tmp = g_lidarall_nonground_times;

        std::lock_guard<std::recursive_mutex> lock_lidar_ssn_time(g_mutex_lidar_ssn_time);
        lidar_ssn_times_tmp = g_lidar_ssn_times;

        // show camera and lidar buffer time
        // std::cout << "--------------------------------------------------" << std::endl;
        if (objects_time[0] != ros::Time(0) && objects_time[0] != object_past_time)
        {
          int sync_time_index = 0;
          std::vector<ros::Time>::iterator sync_times_it;
          std::vector<bool> is_cameras_update (g_cam_ids.size(), false);
          for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
          {
            // std::cout << "objects_time[" << cam_order << "]: " << objects_time[cam_order].sec << "." << objects_time[cam_order].nsec << std::endl;
            sync_times_it = std::find(cam_times_tmp[cam_order].begin(), cam_times_tmp[cam_order].end(), objects_time[cam_order]);
            if (cam_order == 0)
            {
              sync_time_index = std::distance(cam_times_tmp[cam_order].begin(), sync_times_it);
            }
            int time_index = std::distance(cam_times_tmp[cam_order].begin(), sync_times_it);
            ros::Time cam_time = cam_times_tmp[cam_order][time_index];
            if (sync_times_it != cam_times_tmp[cam_order].end() && cam_time != ros::Time(0))
            {
              // std::cout << "sync_camera_time: " << cam_time.sec << "." <<
              // cam_time.nsec <<
              // std::endl;

              cv::Mat message_mat;
              message_mat = getSpecificTimeCameraMessage(g_cache_image[cam_order], objects_time[cam_order], duration_time);
              
              if (!message_mat.empty())
              {
                std::lock_guard<std::mutex> lock_cams(g_mutex_cams[cam_order]);
                g_mats[cam_order] = message_mat;
                is_cameras_update[cam_order] = true;
              }
            }
            else
            {
              std::cout << "Not found the same timestamp in camera " << camera::names[g_cam_ids[cam_order]] <<  " time buffer." << std::endl;
            }
          }
          if (std::all_of(is_cameras_update.begin(), is_cameras_update.end(), [](bool v) { return v; }))
          {
            is_camera_update = true;
          }
          
          /// lidar
          ros::Time sync_lidar_time = lidarall_times_tmp[sync_time_index];
          // std::cout << "sync_lidar_time: " << sync_lidar_time.sec << "." << sync_lidar_time.nsec << std::endl;
          if (sync_lidar_time == ros::Time(0))
          {
            for (size_t index = sync_time_index; index < lidarall_times_tmp.size(); index++)
            {
              if (lidarall_times_tmp[index] != ros::Time(0))
              {
                sync_lidar_time = lidarall_times_tmp[index];
                break;
              }
            }
          }
          if (sync_lidar_time == ros::Time(0))
          {
            is_lidar_update = false;
          }
          else
          {
            pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_ptr =
                getSpecificTimeLidarMessage(g_cache_lidarall, sync_lidar_time, duration_time);
            if (lidar_ptr != nullptr)
            {
              std::lock_guard<std::mutex> lock_lidar_raw(g_mutex_lidar_raw);
              *g_lidarall_ptr = *lidar_ptr;
              is_lidar_update = true;
            }
          }
          /// lidar nonground
          sync_times_it =
              std::find(lidarall_nonground_times_tmp.begin(), lidarall_nonground_times_tmp.end(), sync_lidar_time);
          sync_time_index = std::distance(lidarall_nonground_times_tmp.begin(), sync_times_it);
          // std::cout << "lidarall_nonground_times_tmp[sync_time_index]: " <<
          // lidarall_nonground_times_tmp[sync_time_index] << std::endl;
          if (sync_times_it != lidarall_nonground_times_tmp.end())
          {
            ros::Time sync_lidarall_nonground_time = lidarall_nonground_times_tmp[sync_time_index];
            // std::cout << "sync_lidarall_nonground_time: " << sync_lidarall_nonground_time.sec << "." <<
            // sync_lidarall_nonground_time.nsec <<
            // std::endl;

            if (sync_lidarall_nonground_time == ros::Time(0))
            {
              for (size_t index = sync_time_index; index < lidarall_nonground_times_tmp.size(); index++)
              {
                if (lidarall_nonground_times_tmp[index] != ros::Time(0))
                {
                  sync_lidarall_nonground_time = lidarall_nonground_times_tmp[index];
                  break;
                }
              }
            }
            if (sync_lidarall_nonground_time == ros::Time(0))
            {
              is_lidarall_nonground_update = false;
            }
            else
            {
              pcl::PointCloud<pcl::PointXYZI>::Ptr lidarall_nonground_ptr = getSpecificTimeLidarMessage(
                  g_cache_lidarall_nonground, sync_lidarall_nonground_time, duration_time);
              if (lidarall_nonground_ptr != nullptr)
              {
                std::lock_guard<std::mutex> lock_lidar_nonground(g_mutex_lidar_nonground);
                *g_lidarall_nonground_ptr = *lidarall_nonground_ptr;
                is_lidarall_nonground_update = true;
              }
            }
          }
          /// lidar ssn
          sync_times_it = std::find(lidar_ssn_times_tmp.begin(), lidar_ssn_times_tmp.end(), sync_lidar_time);
          sync_time_index = std::distance(lidar_ssn_times_tmp.begin(), sync_times_it);
          // std::cout << "lidar_ssn_times_tmp[sync_time_index]: " << lidar_ssn_times_tmp[sync_time_index] <<
          // std::endl;
          if (sync_times_it != lidar_ssn_times_tmp.end())
          {
            ros::Time sync_lidar_ssn_time = lidar_ssn_times_tmp[sync_time_index];
            // std::cout << "sync_lidar_ssn_time: " << sync_lidar_ssn_time.sec << "." << sync_lidar_ssn_time.nsec <<
            // std::endl;

            if (sync_lidar_ssn_time == ros::Time(0))
            {
              for (size_t index = sync_time_index; index < lidar_ssn_times_tmp.size(); index++)
              {
                if (lidar_ssn_times_tmp[index] != ros::Time(0))
                {
                  sync_lidar_ssn_time = lidar_ssn_times_tmp[index];
                  break;
                }
              }
            }
            if (sync_lidar_ssn_time == ros::Time(0))
            {
              is_lidar_ssn_update = false;
            }
            else
            {
              pcl::PointCloud<pcl::PointXYZIL>::Ptr lidar_ssn_ptr =
                  getSpecificTimeLidarMessage(g_cache_lidar_ssn, sync_lidar_ssn_time, duration_time);
              if (lidar_ssn_ptr != nullptr)
              {
                std::lock_guard<std::mutex> lock_lidar_ssn(g_mutex_lidar_ssn);
                *g_lidar_ssn_ptr = *lidar_ssn_ptr;
                is_lidar_ssn_update = true;
              }
            }
          }
          else
          {
            std::cout << "Not found the same timestamp in lidar ssn time buffer." << std::endl;
          }
        }
        else
        {
          std::cout << "Not found the same timestamp in camera time buffer." << std::endl;
        }
        object_past_time = objects_time[0];
        if (is_camera_update && is_lidar_update && is_lidarall_nonground_update && is_lidar_ssn_update)
        {
          g_is_data_sync = true;
        }
      }
    }
    loop_rate.sleep();
  }
  std::cout << "getSyncLidarCameraData close." << std::endl;
}
void runInference()
{
  std::cout << "===== runInference... =====" << std::endl;

  /// create variable
  std::vector<bool> is_object_update(g_cam_ids.size());
  bool is_data_ready = true;
  std::vector<cv::Mat> cam_mats(g_cam_ids.size());
  std::vector<msgs::DetectedObjectArray> object_arrs(g_cam_ids.size());
  std::vector<msgs::DetectedObjectArray> remaining_object_arrs(g_cam_ids.size());
  std::vector<msgs::DetectedObjectArray> objects_2d_bbox_arrs(g_cam_ids.size());
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidarall_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidarall_nonground_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_ssn_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_points_ptr(g_cam_ids.size());
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_raw_points_ptr(g_cam_ids.size());
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_bbox_points_ptr(g_cam_ids.size());
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_bbox_raw_points_ptr(g_cam_ids.size());
  std::vector<std::vector<PixelPosition>> cam_pixels(g_cam_ids.size());
  std::vector<std::vector<int>> cam_bboxs_class_id(g_cam_ids.size());
  std::vector<std::vector<int>> cam_bboxs_class_id_raw(g_cam_ids.size());
  std::vector<std::vector<MinMax3D>> cams_bboxs_cube_min_max(g_cam_ids.size());
  std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>>> cams_bboxs_points(g_cam_ids.size());

  /// init
  pclInitializer(cams_raw_points_ptr);
  pclInitializer(cams_points_ptr);
  pclInitializer(cams_bbox_points_ptr);
  pclInitializer(cams_bbox_raw_points_ptr);

  /// main loop
  ros::Rate loop_rate(20);
  while (ros::ok())
  {
    is_data_ready = true;
    if (!g_data_sync || g_is_data_sync)
    {
      /// copy camera data
      for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
      {
        std::lock_guard<std::mutex> lock_cams(g_mutex_cams[cam_order]);
        cam_mats[cam_order] = g_mats[cam_order].clone();

        if (!g_data_sync)
        {
          if (cam_mats[cam_order].empty())
          {
            is_data_ready = false;
          }
        }
      }
      if (!g_data_sync)
      {
        std::lock_guard<std::mutex> lock_objects(g_mutex_objects);
        object_arrs = g_object_arrs;
        is_object_update = g_is_object_update;
        for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
        {
          if (!is_object_update[cam_order])
          {
            is_data_ready = false;
          }
        }
      }
      else
      {
        std::lock_guard<std::mutex> lock_objects_process(g_mutex_objects_process);
        object_arrs = g_object_arrs_process;
        is_object_update = g_is_object_update;
      }
      remaining_object_arrs = object_arrs;
      objects_2d_bbox_arrs = object_arrs;
      for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
      {
        remaining_object_arrs[cam_order].objects.clear();
        objects_2d_bbox_arrs[cam_order].objects.clear();
      }

      /// copy lidar data
      std::lock_guard<std::mutex> lock_lidar_raw(g_mutex_lidar_raw);
      pcl::copyPointCloud(*g_lidarall_ptr, *lidarall_ptr);

      std::lock_guard<std::mutex> lock_lidar_nonground(g_mutex_lidar_nonground);
      pcl::copyPointCloud(*g_lidarall_nonground_ptr, *lidarall_nonground_ptr);

      std::lock_guard<std::mutex> lock_lidar_ssn(g_mutex_lidar_ssn);
      pcl::copyPointCloud(*g_lidar_ssn_ptr, *lidar_ssn_ptr);

      if (!g_data_sync)
      {
        if (lidarall_ptr->empty() && lidar_ssn_ptr->empty())
        {
          is_data_ready = false;
        }
      }
      if (g_data_sync || is_data_ready)
      {
        g_is_data_sync = false;
        std::cout << "===== doInference once =====" << std::endl;
        /// get results
        std::thread get_point_in_image_fov_thread_1(getPointCloudInAllImageFOV, lidar_ssn_ptr,
                                                    std::ref(cams_points_ptr) /*, cam_pixels*/, g_image_w, g_image_h);
        std::thread get_point_in_image_fov_thread_2(getPointCloudInAllImageFOV, lidarall_nonground_ptr,
                                                    std::ref(cams_raw_points_ptr) /*, cam_pixels*/, g_image_w,
                                                    g_image_h);
        get_point_in_image_fov_thread_1.join();
        getPointCloudInAllBoxFOV(object_arrs, remaining_object_arrs, cams_points_ptr, cams_bbox_points_ptr, cam_pixels,
                                 objects_2d_bbox_arrs, /*cams_bboxs_cube_min_max,*/ cams_bboxs_points);

        get_point_in_image_fov_thread_2.join();
        getPointCloudInAllBoxFOV(remaining_object_arrs, cams_raw_points_ptr, cams_bbox_raw_points_ptr, cam_pixels,
                                 objects_2d_bbox_arrs, /*cams_bboxs_cube_min_max,*/ cams_bboxs_points);

        if (g_is_display)
        {
          for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
          {
            *cams_bbox_points_ptr[cam_order] += *cams_bbox_raw_points_ptr[cam_order];
            *cams_points_ptr[cam_order] += *cams_raw_points_ptr[cam_order];
          }
          /// draw results on image
          drawPointCloudOnImages(cam_mats, cam_pixels, cams_bbox_points_ptr);
          drawBoxOnImages(cam_mats, object_arrs);

          /// prepare point cloud visualization
          std::lock_guard<std::mutex> lock_lidar_process(g_mutex_lidar_process);
          pcl::copyPointCloud(*lidarall_ptr, *g_lidarall_ptr_process);

          std::lock_guard<std::mutex> lock_cams_points(g_mutex_cams_points);
          std::lock_guard<std::mutex> lock_objects_points(g_mutex_objects_points);
          for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
          {
            pcl::copyPointCloud(*cams_points_ptr[cam_order], *g_cams_points_ptr[cam_order]);
            pcl::copyPointCloud(*cams_bbox_points_ptr[cam_order], *g_cams_bbox_points_ptr[cam_order]);
          }

          // std::lock_guard<std::mutex> lock_cube(g_mutex_cube);
          // g_cams_bboxs_cube_min_max = cams_bboxs_cube_min_max;
          std::lock_guard<std::mutex> lock_polygon(g_mutex_polygon);
          g_cams_bboxs_points = cams_bboxs_points;

          /// prepare image visualization
          std::lock_guard<std::mutex> lock_cams_process(g_mutex_cams_process);
          for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
          {
            g_mats_process[cam_order] = cam_mats[cam_order].clone();
          }
        }

        object_publisher(objects_2d_bbox_arrs, cams_bboxs_points, /*cams_bboxs_cube_min_max,*/ object_arrs[0].header);

        release(cam_pixels);
        release(cam_bboxs_class_id);
        release(cams_bboxs_cube_min_max);
        release(cams_bboxs_points);
        release(cam_bboxs_class_id_raw);
      }
    }
    loop_rate.sleep();
  }
  std::cout << "===== runInference close =====" << std::endl;
}
void buffer_monitor()
{
  std::vector<ros::Time> cam_single_time_last(g_cam_ids.size());
  ros::Time lidarall_time_last;
  ros::Time lidarall_nonground_time_last;
  ros::Time lidar_ssn_time_last;
  bool cam_single_time_last_updated = false;
  bool lidarall_time_last_updated = false;
  bool lidarall_nonground_time_last_updated = false;
  bool lidar_ssn_time_last_updated = false;
  /// main loop
  ros::Rate loop_rate(80);
  while (ros::ok())
  {
    if (g_data_sync)
    {
      // Add buffer
      std::lock_guard<std::recursive_mutex> lock_cam_times(g_mutex_cam_times);
      for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
      {
        std::lock_guard<std::mutex> lock_cam_time(g_mutex_cam_time[cam_order]);

        if (!g_cam_single_times[cam_order].empty())
        {
          cam_single_time_last[cam_order] = g_cam_single_times[cam_order].front();  /* store last timestamp */
          
          // std::cout  <<"cam_single_time_last[cam_order]:    " << cam_single_time_last[cam_order].sec << "." <<
          // cam_single_time_last[cam_order].nsec << " store" << 
          // std::endl;
          
          if (!cam_single_time_last_updated)
          {
            cam_single_time_last_updated = true;
          }
          g_cam_times[cam_order].push_back(g_cam_single_times[cam_order].front());			
          g_cam_single_times[cam_order].erase(g_cam_single_times[cam_order].begin());
        }
        else
        {
          /* if empty, then use last timestamp */
          if (cam_single_time_last_updated)
          {
            g_cam_times[cam_order].push_back(cam_single_time_last[cam_order]);
            
            // std::cout  <<"cam_single_time_last[cam_order]:    " << cam_single_time_last[cam_order].sec << "." <<
            // cam_single_time_last[cam_order].nsec << 
            // std::endl;
          }
          else
          {
            g_cam_times[cam_order].push_back(ros::Time(0));
          }
        }
      }

      // std::cout << "buffer camera object: " << g_object_arrs[0].header.stamp.sec << "." <<
      // g_object_arrs[0].header.stamp.nsec <<
      // std::endl;

      std::lock_guard<std::recursive_mutex> lock_lidar_time(g_mutex_lidar_time);
      if (!g_lidarall_time_buffer.empty())
      {
        lidarall_time_last = g_lidarall_time_buffer.front();  // store last timestamp 
        
        // std::cout  <<"lidarall_time_last:    " << g_lidarall_time_buffer.front().sec << "." <<
        // g_lidarall_time_buffer.front().nsec << " store" << 
        // std::endl;
        
        if (!lidarall_time_last_updated)
        {
          lidarall_time_last_updated = true;
        }
        
        g_lidarall_times.push_back(g_lidarall_time_buffer.front());
        g_lidarall_time_buffer.erase(g_lidarall_time_buffer.begin());
      }
      else
      {
        // if empty, then use last timestamp 
        if (lidarall_time_last_updated)
        {
          g_lidarall_times.push_back(lidarall_time_last);				
          // std::cout  <<"lidarall_time_last:    " << lidarall_time_last.sec << "." <<
          // lidarall_time_last.nsec << " empty " <<
          // std::endl; 
        }
        else
        {
          g_lidarall_times.push_back(ros::Time(0));
        }
      }
      std::lock_guard<std::recursive_mutex> lock_lidar_nonground_time(g_mutex_lidar_nonground_time);
      if (!g_lidarall_nonground_time_buffer.empty())
      {
        lidarall_nonground_time_last = g_lidarall_nonground_time_buffer.front();  // store last timestamp 
        // std::cout  <<"lidarall_nonground_time_last:    " << lidarall_nonground_time_last.sec << "." <<
        // lidarall_nonground_time_last.nsec << " store" << 
        // std::endl;
        
        if (!lidarall_nonground_time_last_updated)
        {
          lidarall_nonground_time_last_updated = true;
        }
        
        g_lidarall_nonground_times.push_back(g_lidarall_nonground_time_buffer.front());
        g_lidarall_nonground_time_buffer.erase(g_lidarall_nonground_time_buffer.begin());
      }	
      else
      {
        // if empty, then use last timestamp 
        if (lidarall_nonground_time_last_updated)
        {
          g_lidarall_nonground_times.push_back(lidarall_nonground_time_last);
          // std::cout  <<"lidarall_nonground_time_last:    " << lidarall_nonground_time_last.sec << "." <<
          // lidarall_nonground_time_last.nsec << 
          // std::endl; 
        }
        else
        {
          g_lidarall_nonground_times.push_back(ros::Time(0));
        }
      }
      std::lock_guard<std::recursive_mutex> lock_lidar_ssn_time(g_mutex_lidar_ssn_time);
      if (!g_lidar_ssn_time_buffer.empty())
      {
        lidar_ssn_time_last = g_lidar_ssn_time_buffer.front();  //store last timestamp 
        // std::cout  <<"lidar_ssn_time_last:    " << lidar_ssn_time_last.sec << "." <<
        // lidar_ssn_time_last.nsec << " store" << 
        // std::endl;
        
        if (!lidar_ssn_time_last_updated)
        {
          lidar_ssn_time_last_updated = true;
        }
        
        g_lidar_ssn_times.push_back(g_lidar_ssn_time_buffer.front());
        g_lidar_ssn_time_buffer.erase(g_lidar_ssn_time_buffer.begin());
      } 	  
      else
      {
        // if empty, then use last timestamp 
        if (lidar_ssn_time_last_updated)
        {
          g_lidar_ssn_times.push_back(lidar_ssn_time_last);
          // std::cout  <<"lidar_ssn_time_last:    " << lidar_ssn_time_last.sec << "." <<
          // lidar_ssn_time_last.nsec << 
          // std::endl;      
        }
        else
        {
          g_lidar_ssn_times.push_back(ros::Time(0));
        }
      }

      // Clear buffer
      if (static_cast<int>(g_cam_times[0].size()) >= g_buffer_size)
      {
        std::lock_guard<std::recursive_mutex> lock_cam_times(g_mutex_cam_times);
        for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
        {
          g_cam_times[cam_order].erase(g_cam_times[cam_order].begin(),
                                       g_cam_times[cam_order].begin() + g_buffer_size / 3);
        }
        std::lock_guard<std::recursive_mutex> lock_lidar_time(g_mutex_lidar_time);
        g_lidarall_times.erase(g_lidarall_times.begin(), g_lidarall_times.begin() + g_buffer_size / 3);

        std::lock_guard<std::recursive_mutex> lock_lidar_nonground_time(g_mutex_lidar_nonground_time);
        g_lidarall_nonground_times.erase(g_lidarall_nonground_times.begin(),
                                         g_lidarall_nonground_times.begin() + g_buffer_size / 3);

        std::lock_guard<std::recursive_mutex> lock_lidar_ssn_time(g_mutex_lidar_ssn_time);
        g_lidar_ssn_times.erase(g_lidar_ssn_times.begin(), g_lidar_ssn_times.begin() + g_buffer_size / 3);
      }
    }
    loop_rate.sleep();
  }
}
int main(int argc, char** argv)
{
  std::cout << "===== Multi_sensor_3d_object_detection startup. =====" << std::endl;
  ros::init(argc, argv, "Multi_sensor_3d_object_detection");
  ros::NodeHandle nh;

  /// ros Subscriber
  std::vector<ros::Subscriber> cam_subs(g_cam_ids.size());
  std::vector<ros::Subscriber> object_subs(g_cam_ids.size());
  ros::Subscriber lidarall;
  ros::Subscriber lidar_ssn_sub;
  ros::Subscriber lidarall_nonground;

  /// message_filters Subscriber
  std::vector<message_filters::Subscriber<sensor_msgs::Image>> cam_filter_subs(g_cam_ids.size());
  std::vector<message_filters::Subscriber<msgs::DetectedObjectArray>> object_filter_subs(g_cam_ids.size());
  message_filters::Subscriber<pcl::PointCloud<pcl::PointXYZI>> sub_filter_lidarall;
  message_filters::Subscriber<pcl::PointCloud<pcl::PointXYZI>> sub_filter_lidarall_nonground;
  message_filters::Subscriber<pcl::PointCloud<pcl::PointXYZIL>> sub_filter_lidar_ssn;

  /// get callback function
  static void (*f_callbacks_cam[])(const sensor_msgs::Image::ConstPtr&) = {
    callback_cam_front_bottom_60, callback_cam_front_top_far_30, callback_cam_right_back_60, callback_cam_left_back_60,
    callback_cam_right_front_60, callback_cam_left_front_60
  };
  static void (*f_callbacks_object[])(const msgs::DetectedObjectArray::ConstPtr&) = {
    callback_object_cam_front_bottom_60, callback_object_cam_front_top_far_30, callback_object_cam_right_back_60,
    callback_object_cam_left_back_60, callback_object_cam_right_front_60, callback_object_cam_left_front_60
  };

  /// set topic name
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    g_cam_topic_names[cam_order] = camera::topics[g_cam_ids[cam_order]];
    g_bbox_topic_names[cam_order] = camera::topics_obj[g_cam_ids[cam_order]];
  }

  if (!g_data_sync)
  {
    /// ros Subscriber
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      cam_subs[cam_order] = nh.subscribe(g_cam_topic_names[cam_order], 1, f_callbacks_cam[cam_order]);
      object_subs[cam_order] = nh.subscribe(g_bbox_topic_names[cam_order], 1, f_callbacks_object[cam_order]);
    }

    lidarall = nh.subscribe("/LidarAll", 1, callback_lidarall);
    lidarall_nonground = nh.subscribe("/LidarAll/NonGround", 1, callback_lidarall_nonground);
    lidar_ssn_sub = nh.subscribe("/squ_seg/result_cloud", 1, callback_ssn);
  }
  else
  {
    /// message_filters Subscriber
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      cam_filter_subs[cam_order].subscribe(nh, g_cam_topic_names[cam_order], 1);
      g_cache_image[cam_order].connectInput(cam_filter_subs[cam_order]);
      g_cache_image[cam_order].registerCallback(f_callbacks_cam[cam_order]);
      g_cache_image[cam_order].setCacheSize(g_buffer_size);

      object_filter_subs[cam_order].subscribe(nh, g_bbox_topic_names[cam_order], 1);
      object_filter_subs[cam_order].registerCallback(f_callbacks_object[cam_order]);
    }
    sub_filter_lidarall.subscribe(nh, "/LidarAll", 1);
    g_cache_lidarall.setCacheSize(g_buffer_size);
    g_cache_lidarall.connectInput(sub_filter_lidarall);
    g_cache_lidarall.registerCallback(callback_lidarall);

    sub_filter_lidarall_nonground.subscribe(nh, "/LidarAll/NonGround", 1);
    g_cache_lidarall_nonground.setCacheSize(g_buffer_size);
    g_cache_lidarall_nonground.connectInput(sub_filter_lidarall_nonground);
    g_cache_lidarall_nonground.registerCallback(callback_lidarall_nonground);

    sub_filter_lidar_ssn.subscribe(nh, "/squ_seg/result_cloud", 1);
    g_cache_lidar_ssn.setCacheSize(g_buffer_size);
    g_cache_lidar_ssn.connectInput(sub_filter_lidar_ssn);
    g_cache_lidar_ssn.registerCallback(callback_ssn);

    /// object publisher
    g_object_pub = nh.advertise<msgs::DetectedObjectArray>(camera::detect_result, 8);
  }

  /// class init
  alignmentInitializer();
  pclInitializer(g_cams_points_ptr);
  pclInitializer(g_cams_bbox_points_ptr);

  /// visualization
  std::thread display_lidar_thread;
  std::thread display_camera_thread;
  if (g_is_display)
  {
    display_lidar_thread = std::thread(displayLidarData);
    display_camera_thread = std::thread(displayCameraData);
  }

  /// sync
  std::thread sync_data_thread;
  std::thread buffer_monitor_thread;
  if (g_data_sync)
  {
    buffer_monitor_thread = std::thread(buffer_monitor);
    sync_data_thread = std::thread(getSyncLidarCameraData);
  }
  /// main loop start
  std::thread main_thread(runInference);
  int thread_count = int(g_cam_ids.size()) * 2 + 3;  /// camera raw + object + lidar raw + ssn + nonground
  ros::MultiThreadedSpinner spinner(thread_count);
  spinner.spin();
  std::cout << "===== Multi_sensor_3d_object_detection running... =====" << std::endl;

  /// main loop end
  if (g_is_display)
  {
    display_lidar_thread.join();
    display_camera_thread.join();
  }
  if (g_data_sync)
  {
    sync_data_thread.join();
    buffer_monitor_thread.join();
  }
  main_thread.join();
  std::cout << "===== Multi_sensor_3d_object_detection shutdown. =====" << std::endl;
  return 0;
}
