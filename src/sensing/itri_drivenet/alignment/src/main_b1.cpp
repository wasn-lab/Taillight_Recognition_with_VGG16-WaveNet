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
#include "alignment.h"
#include "visualization_util.h"
#include <drivenet/object_label_util.h>
#include "point_preprocessing.h"
#include "ssn_util.h"

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
#if CAR_MODEL_IS_B1
const std::vector<camera::id> g_cam_ids{ camera::id::front_60, camera::id::top_front_120, camera::id::left_60,
                                         camera::id::right_60 };
#else
#error "car model is not well defined"
#endif

/// class
std::vector<Alignment> g_alignments(g_cam_ids.size());
Visualization g_visualization;

/// thread
std::vector<std::mutex> g_sync_lock_cams(g_cam_ids.size());
std::mutex g_sync_lock_cams_process;
std::vector<std::mutex> g_sync_lock_objects(g_cam_ids.size());
std::mutex g_sync_lock_objects_process;
std::mutex g_sync_lock_lidar_raw;
std::mutex g_sync_lock_lidar_process;
std::mutex g_sync_lock_lidar_ssn;
std::mutex g_sync_lock_cams_points;
std::mutex g_sync_lock_objects_points;
std::vector<std::mutex> g_sync_lock_times(g_cam_ids.size());
std::mutex g_sync_lock_lidar_time;
std::mutex g_sync_lock_lidar_ssn_time;
std::vector<std::mutex> g_sync_lock_cam_time(g_cam_ids.size());
std::mutex g_sync_lock_cube;
/// params
bool g_is_enable_default_3d_bbox = false;
bool g_data_sync = true;  // trun on or trun off data sync function

/// inference params
bool g_is_data_sync = false;
std::vector<bool> g_is_object_update(g_cam_ids.size());

/// ros
std::vector<message_filters::Cache<sensor_msgs::Image>> g_cache_image(g_cam_ids.size());
message_filters::Cache<pcl::PointCloud<pcl::PointXYZI>> g_cache_lidar;
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
pcl::PointCloud<pcl::PointXYZI>::Ptr g_lidarall_ptr_process(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZIL>::Ptr g_lidar_ssn_ptr(new pcl::PointCloud<pcl::PointXYZIL>);
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> g_cams_points_ptr(g_cam_ids.size());
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> g_cams_bbox_points_ptr(g_cam_ids.size());
std::vector<pcl::visualization::Camera> g_cam;

/// object
std::vector<msgs::DetectedObjectArray> g_object_arrs(g_cam_ids.size());
std::vector<msgs::DetectedObjectArray> g_object_arrs_process(g_cam_ids.size());
/// sync camera and lidar
int g_buffer_size = 60;
std::vector<std::vector<ros::Time>> g_cam_times(g_cam_ids.size());
std::vector<ros::Time> g_cam_single_time(g_cam_ids.size());
std::vector<ros::Time> g_lidarall_times;
std::vector<ros::Time> g_lidar_ssn_times;
ros::Time g_lidarall_time;
ros::Time g_lidar_ssn_time;
/// 3d cube
std::vector<std::vector<pcl_cube>> g_cams_bboxs_cube_min_max(g_cam_ids.size());

//////////////////// for camera image
void callback_cam_front_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  if (!g_data_sync)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    g_sync_lock_cams[cam_order].lock();
    g_mats[cam_order] = cv_ptr->image;
    g_sync_lock_cams[cam_order].unlock();
  }
  else
  {
    g_sync_lock_cam_time[cam_order].lock();
    g_cam_single_time[cam_order] = msg->header.stamp;
    g_sync_lock_cam_time[cam_order].unlock();

    for (size_t index = 0; index < g_cam_ids.size(); index++)
    {
      g_sync_lock_times[index].lock();
      g_cam_times[index].push_back(g_cam_single_time[index]);
      g_sync_lock_times[index].unlock();
    }
    g_sync_lock_lidar_time.lock();
    g_lidarall_times.push_back(g_lidarall_time);
    g_sync_lock_lidar_time.unlock();

    g_sync_lock_lidar_ssn_time.lock();
    g_lidar_ssn_times.push_back(g_lidar_ssn_time);
    g_sync_lock_lidar_ssn_time.unlock();
    // ROS_INFO("cam_front_60 timestamp: %d.%d", msg->header.stamp.sec, msg->header.stamp.nsec);
  }
}
void callback_cam_top_front_120(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::top_front_120);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  if (!g_data_sync)
  {
    g_sync_lock_cams[cam_order].lock();
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    g_mats[cam_order] = cv_ptr->image;
    g_sync_lock_cams[cam_order].unlock();
  }
  else
  {
    g_sync_lock_cam_time[cam_order].lock();
    g_cam_single_time[cam_order] = msg->header.stamp;
    g_sync_lock_cam_time[cam_order].unlock();
  }
}

void callback_cam_left_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  if (!g_data_sync)
  {
    g_sync_lock_cams[cam_order].lock();
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    g_mats[cam_order] = cv_ptr->image;
    g_sync_lock_cams[cam_order].unlock();
  }
  else
  {
    g_sync_lock_cam_time[cam_order].lock();
    g_cam_single_time[cam_order] = msg->header.stamp;
    g_sync_lock_cam_time[cam_order].unlock();
  }
}
void callback_cam_right_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  if (!g_data_sync)
  {
    g_sync_lock_cams[cam_order].lock();
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    g_mats[cam_order] = cv_ptr->image;
    g_sync_lock_cams[cam_order].unlock();
  }
  else
  {
    g_sync_lock_cam_time[cam_order].lock();
    g_cam_single_time[cam_order] = msg->header.stamp;
    g_sync_lock_cam_time[cam_order].unlock();
  }
}

//////////////////// for camera object
void callback_object_cam_front_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_objects[cam_order].lock();
  g_is_object_update[cam_order] = true;
  g_object_arrs[cam_order] = *msg;
  g_sync_lock_objects[cam_order].unlock();
}
void callback_object_cam_top_front_120(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::top_front_120);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_objects[cam_order].lock();
  g_is_object_update[cam_order] = true;
  g_object_arrs[cam_order] = *msg;
  g_sync_lock_objects[cam_order].unlock();
}
void callback_object_cam_left_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_objects[cam_order].lock();
  g_is_object_update[cam_order] = true;
  g_object_arrs[cam_order] = *msg;
  g_sync_lock_objects[cam_order].unlock();
}
void callback_object_cam_right_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_objects[cam_order].lock();
  g_is_object_update[cam_order] = true;
  g_object_arrs[cam_order] = *msg;
  g_sync_lock_objects[cam_order].unlock();
}
void callback_lidarall(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  if (!g_data_sync)
  {
    g_sync_lock_lidar_raw.lock();
    *g_lidarall_ptr = *msg;
    g_sync_lock_lidar_raw.unlock();
  }
  else
  {
    ros::Time header_time;
    pcl_conversions::fromPCL(msg->header.stamp, header_time);
    g_sync_lock_lidar_time.lock();
    g_lidarall_time = header_time;
    g_sync_lock_lidar_time.unlock();
    // ROS_INFO("lidarall timestamp: %d.%d", header_time.sec, header_time.nsec);
  }
  // std::cout << "Point cloud size: " << g_lidarall_ptr->size() << std::endl;
  // std::cout << "Lidar x: " << g_lidarall_ptr->points[0].x << ", y: " << g_lidarall_ptr->points[0].y << ", z: " <<
  // g_lidarall_ptr->points[0].z << std::endl;
}

void callback_ssn(const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr& msg)
{
  if (!g_data_sync)
  {
    g_sync_lock_lidar_ssn.lock();
    *g_lidar_ssn_ptr = *msg;
    g_sync_lock_lidar_ssn.unlock();
  }
  else
  {
    ros::Time header_time;
    pcl_conversions::fromPCL(msg->header.stamp, header_time);
    g_sync_lock_lidar_ssn_time.lock();
    g_lidar_ssn_time = header_time;
    g_sync_lock_lidar_ssn_time.unlock();
    // ROS_INFO("lidarall timestamp: %d.%d", header_time.sec, header_time.nsec);
  }
}

void pclViewerInitializer(boost::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer,
                          std::vector<std::string> window_name, int window_count = 3)
{
  if (window_name.size() < 3)
  {
    window_name.clear();
    window_name.push_back("raw_data");
    window_name.push_back("image fov");
    window_name.push_back("object");
  }
  if (window_count < 3)
  {
    window_count = 3;
  }

  int v1 = 1, v2 = 2, v3 = 3;
  pcl_viewer->createViewPort(0.0, 0.0, 0.33, 1.0, v1);
  pcl_viewer->createViewPort(0.33, 0.0, 0.66, 1.0, v2);
  pcl_viewer->createViewPort(0.66, 0.0, 1.0, 1.0, v3);
  pcl_viewer->initCameraParameters();
  pcl_viewer->addCoordinateSystem(3.0, 0, 0, 0);
  pcl_viewer->setCameraPosition(0, 0, 20, 0.2, 0, 0);
  pcl_viewer->setShowFPS(false);
  for (int count = 1; count < window_count + 1; count++)
  {
    pcl_viewer->setBackgroundColor(0, 0, 0, count);
    pcl_viewer->addText(window_name[count - 1], 10, 10, window_name[count - 1], count);
  }
}

void cvViewerInitializer()
{
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    cv::namedWindow(g_cam_topic_names[cam_order], cv::WINDOW_NORMAL);
    cv::resizeWindow(g_cam_topic_names[cam_order], 360, 270);
    cv::moveWindow(g_cam_topic_names[cam_order], 380*cam_order, 30);
  }
}

void pclInitializer(std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& points_ptr)
{
  for (size_t i = 0; i < points_ptr.size(); i++)
  {
    points_ptr[i] = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  }
}

void pointsColorInit(std::vector<pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>>& rgb_points,
                     std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> points_ptr)
{
  cv::Scalar point_color = CvColor::white_;
  for (size_t index = 0; index < points_ptr.size(); index++)
  {
    point_color = intToColor(static_cast<int>(index));
    rgb_points.push_back(pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>(
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
void getPointCloudIn3DBox(const pcl::PointCloud<pcl::PointXYZI> cloud_src, int object_class_id,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_ptr)
{
  // std::cout << "===== getPointCloudIn3DBox... =====" << std::endl;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointXYZI minPt, maxPt;

  /// get the box length of object
  pcl::getMinMax3D(cloud_src, minPt, maxPt);
  object_box bbox;
  bbox = getDefaultObjectBox(object_class_id);

  /// build the condition
  pcl::ConditionAnd<pcl::PointXYZI>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZI>());
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZI>("x", pcl::ComparisonOps::GT, minPt.x)));
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZI>("x", pcl::ComparisonOps::LT, minPt.x + bbox.length)));

  /// build the filter
  pcl::ConditionalRemoval<pcl::PointXYZI> condrem;
  condrem.setCondition(range_cond);
  cloud_ptr = cloud_src.makeShared();
  condrem.setInputCloud(cloud_ptr);
  condrem.setKeepOrganized(false);

  /// apply filter
  condrem.filter(*cloud_filtered_ptr);
}
void getPointCloudInImageFOV(pcl::PointCloud<pcl::PointXYZI>::Ptr lidarall_ptr,
                             std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_points_ptr,
                             std::vector<std::vector<PixelPosition>>& cam_pixels, int image_w, int image_h)
{
  // std::cout << "===== getPointCloudInImageFOV... =====" << std::endl;
  /// create variable
  std::vector<pcl::PointCloud<pcl::PointXYZI>> cam_points(cams_points_ptr.size());
  std::vector<int> cloud_sizes(cams_points_ptr.size(), 0);
  std::vector<std::vector<std::vector<pcl::PointXYZI>>> point_cloud(
      cams_points_ptr.size(), std::vector<std::vector<pcl::PointXYZI>>(image_w, std::vector<pcl::PointXYZI>(image_h)));

  /// copy from source
  for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
  {
    pcl::copyPointCloud(*lidarall_ptr, *cams_points_ptr[cam_order]);
    cam_points[cam_order] = *cams_points_ptr[cam_order];
  }
  /// find 3d points in image coverage
  for (size_t i = 0; i < lidarall_ptr->size(); i++)
  {
    if (lidarall_ptr->points[i].x > 0)
    {
      for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
      {
        PixelPosition pixel_position{ -1, -1 };
        pixel_position = g_alignments[cam_order].projectPointToPixel(lidarall_ptr->points[i]);
        if (pixel_position.u >= 0 && pixel_position.v >= 0)
        {
          if (point_cloud[cam_order][pixel_position.u][pixel_position.v].x > lidarall_ptr->points[i].x ||
              point_cloud[cam_order][pixel_position.u][pixel_position.v].x == 0)
          {
            point_cloud[cam_order][pixel_position.u][pixel_position.v] = lidarall_ptr->points[i];
          }
        }
      }
    }
  }
  /// record the 2d points and 3d points.
  for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
  {
    for (int u = 0; u < image_w; u++)
    {
      for (int v = 0; v < image_h; v++)
      {
        PixelPosition pixel_position{ -1, -1 };
        pixel_position.u = u;
        pixel_position.v = v;
        if (point_cloud[cam_order][u][v].x != 0 && point_cloud[cam_order][u][v].y != 0 &&
            point_cloud[cam_order][u][v].z != 0)
        {
          // cam_pixels[cam_order].push_back(pixel_position);
          cam_points[cam_order].points[cloud_sizes[cam_order]] = point_cloud[cam_order][u][v];
          cloud_sizes[cam_order]++;
        }
      }
    }
  }
  /// copy to destination
  for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
  {
    cam_points[cam_order].resize(cloud_sizes[cam_order]);
    *cams_points_ptr[cam_order] = cam_points[cam_order];
  }
}

void getPointCloudInBoxFOV(std::vector<msgs::DetectedObjectArray>& objects,
                           std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_points_ptr,
                           std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_bbox_points_ptr,
                           std::vector<std::vector<PixelPosition>>& cam_pixels,
                           std::vector<std::vector<pcl_cube>>& cams_bboxs_cube_min_max)
{
  // std::cout << "===== getPointCloudInBoxFOV... =====" << std::endl;
  /// create variable
  std::vector<pcl::PointCloud<pcl::PointXYZI>> cam_points(cams_points_ptr.size());
  std::vector<int> cloud_sizes(cam_points.size(), 0);
  pcl::PointCloud<pcl::PointXYZI> point_cloud_src;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<pcl::PointXYZI> point_vector_object;
  std::vector<pcl::PointXYZI> point_vector_objects;
  std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>>> cams_bboxs_points;

  /// copy from source
  for (size_t cam_order = 0; cam_order < cam_points.size(); cam_order++)
  {
    pcl::copyPointCloud(*cams_points_ptr[cam_order], *cams_bbox_points_ptr[cam_order]);
    cam_points[cam_order] = *cams_bbox_points_ptr[cam_order];
  }

  /// main
  for (size_t cam_order = 0; cam_order < cam_points.size(); cam_order++)
  {
    std::vector<pcl_cube> bboxs_cube_min_max;
    for (const auto& obj : objects[cam_order].objects)
    {
      pcl::PointCloud<pcl::PointXYZI> bbox_points;
      pcl_cube cube_min_max; // object min and max point
      for (size_t i = 0; i < cam_points[cam_order].points.size(); i++)
      {
        // get the 2d box
        std::vector<PixelPosition> bbox_positions(2);
        bbox_positions[0].u = obj.camInfo.u;
        bbox_positions[0].v = obj.camInfo.v;
        bbox_positions[1].u = obj.camInfo.u + obj.camInfo.width;
        bbox_positions[1].v = obj.camInfo.v + obj.camInfo.height;
        transferPixelScaling(bbox_positions);

        // get points in the 2d box
        PixelPosition pixel_position{ -1, -1 };
        pixel_position = g_alignments[cam_order].projectPointToPixel(cam_points[cam_order].points[i]);
        if (pixel_position.u >= bbox_positions[0].u && pixel_position.v >= bbox_positions[0].v &&
            pixel_position.u <= bbox_positions[1].u && pixel_position.v <= bbox_positions[1].v)
        {
          cam_pixels[cam_order].push_back(pixel_position);
          point_vector_object.push_back(cam_points[cam_order].points[i]);
          bbox_points.push_back(cam_points[cam_order].points[i]);
        }
      }
      // vector to point cloud
      pcl::PointCloud<pcl::PointXYZI> point_cloud_object;
      point_cloud_object.points.resize(point_vector_object.size());
      for (size_t i = 0; i < point_vector_object.size(); i++)
      {
        point_cloud_object.points[i] = point_vector_object[i];
      }
      point_vector_object.clear();

      // get points in the 3d box
      if (g_is_enable_default_3d_bbox)
      {
        getPointCloudIn3DBox(point_cloud_object, obj.classId, cloud_filtered_ptr);
      }
      else
      {
        cloud_filtered_ptr = point_cloud_object.makeShared();
      }

      // point cloud to vector
      for (const auto& point : cloud_filtered_ptr->points)
      {
        point_vector_object.push_back(point);
      }

      // concatenate the points of objects
      point_vector_objects.insert(point_vector_objects.begin(), point_vector_object.begin(), point_vector_object.end());
      point_vector_object.clear();
      pcl::getMinMax3D (bbox_points, cube_min_max.p_min, cube_min_max.p_max);
      object_box bbox;
      bbox = getDefaultObjectBox(obj.classId);
      cube_min_max.p_max.x = cube_min_max.p_min.x + bbox.length;
      cube_min_max.p_max.y = cube_min_max.p_min.y + bbox.width;
      cube_min_max.p_max.z = cube_min_max.p_min.z + bbox.height;
      bboxs_cube_min_max.push_back(cube_min_max);
    }
    removeDuplePoints(point_vector_objects);
    for (size_t i = 0; i < point_vector_objects.size(); i++)
    {
      cam_points[cam_order].points[i] = point_vector_objects[i];
      cloud_sizes[cam_order]++;
    }
    point_vector_objects.clear();
    cams_bboxs_cube_min_max[cam_order] = bboxs_cube_min_max;
  }
  /// copy to destination
  for (size_t cam_order = 0; cam_order < cam_points.size(); cam_order++)
  {
    cam_points[cam_order].resize(cloud_sizes[cam_order]);
    *cams_bbox_points_ptr[cam_order] = cam_points[cam_order];
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
  // std::cout << "===== displayLidarData... =====" << std::endl;
  /// create variable
  boost::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer(
      new pcl::visualization::PCLVisualizer("Cloud_Viewer"));
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidarall_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_points_ptr(g_cams_points_ptr.size());
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_bbox_points_ptr(g_cams_bbox_points_ptr.size());
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb_lidarall(g_lidarall_ptr_process, 255, 255, 255);
  std::vector<pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>> rgb_cams_points;
  std::vector<pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>> rgb_cams_bbox_points;
  std::vector<int> viewports{ 1, 2, 3 };
  std::vector<std::string> view_name{ "raw data", "image fov", "object" };

  /// init
  pclViewerInitializer(pcl_viewer, view_name, static_cast<int>(viewports.size()));
  pclInitializer(cams_points_ptr);
  pclInitializer(cams_bbox_points_ptr);
  pointsColorInit(rgb_cams_points, g_cams_points_ptr);
  pointsColorInit(rgb_cams_bbox_points, g_cams_bbox_points_ptr);

  /// main loop
  ros::Rate loop_rate(10);
  while (ros::ok() && !pcl_viewer->wasStopped())
  {
    /// remove points on pcl viewer
    pcl_viewer->removePointCloud("Cloud viewer", viewports[0]);
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      pcl_viewer->removePointCloud(g_cam_topic_names[cam_order], viewports[1]);
      pcl_viewer->removePointCloud(g_bbox_topic_names[cam_order], viewports[2]);
    }
    pcl_viewer->removeAllShapes();

    /// draw points on pcl viewer
    g_sync_lock_lidar_process.lock();
    pcl_viewer->addPointCloud<pcl::PointXYZI>(g_lidarall_ptr_process, rgb_lidarall, "Cloud viewer", viewports[0]);
    g_sync_lock_lidar_process.unlock();
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      g_sync_lock_cams_points.lock();  // mutex camera points
      pcl_viewer->addPointCloud<pcl::PointXYZI>(g_cams_points_ptr[cam_order], rgb_cams_points[cam_order],
                                                g_cam_topic_names[cam_order], viewports[1]);
      g_sync_lock_cams_points.unlock();   // mutex camera points
      g_sync_lock_objects_points.lock();  // mutex camera object points
      pcl_viewer->addPointCloud<pcl::PointXYZI>(g_cams_bbox_points_ptr[cam_order], rgb_cams_bbox_points[cam_order],
                                                g_bbox_topic_names[cam_order], viewports[2]);
      g_sync_lock_objects_points.unlock();  // mutex camera object points

      g_sync_lock_cube.lock();  // mutex camera points
      if(g_cams_bboxs_cube_min_max[cam_order].size() > 0)
      {
        int cube_cout = 0;
        for(const auto& cube: g_cams_bboxs_cube_min_max[cam_order])
        {
          std::string cube_id = "cube_cam" + std::to_string(cam_order) + "_"+ std::to_string(cube_cout);
          cv::Scalar cube_color = CvColor::white_;
          cube_color = intToColor(static_cast<int>(cam_order));

          pcl_viewer->addCube(cube.p_min.x, cube.p_max.x, cube.p_min.y, cube.p_max.y, cube.p_min.z, cube.p_max.z, cube_color[0], cube_color[1], cube_color[2], cube_id, viewports[2]);
          pcl_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, cube_id);
          cube_cout ++;
        }
      }
      g_sync_lock_cube.unlock();  // mutex camera points

    }
    pcl_viewer->spinOnce();
    loop_rate.sleep();
  }
}
void displayCameraData()
{
  // std::cout << "===== displayCameraData... =====" << std::endl;
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
        g_sync_lock_cams_process.lock();  // mutex camera
        cv::imshow(g_cam_topic_names[cam_order], g_mats_process[cam_order]);
        g_sync_lock_cams_process.unlock();  // mutex camera
      }
    }
    cv::waitKey(1);
    loop_rate.sleep();
  }
}
cv::Mat getSpecificTimeCameraMessage(message_filters::Cache<sensor_msgs::Image>& cache_image, ros::Time target_time,
                                     ros::Duration duration_time)
{
  ros::Time begin_time = target_time - duration_time;
  ros::Time end_time = target_time + duration_time;
  std::vector<sensor_msgs::Image::ConstPtr> images = cache_image.getInterval(begin_time, end_time);
  cv::Mat out_mat;
  if (images.size() > 0)
  {
    std::vector<ros::Time> images_time;
    for (const auto& msg : images)
    {
      images_time.push_back(msg->header.stamp);
    }
    std::vector<ros::Time>::iterator it;
    it = std::find(images_time.begin(), images_time.end(), target_time);
    if (it != images_time.end())
    {
      int time_index = std::distance(images_time.begin(), it);
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(images[time_index], sensor_msgs::image_encodings::BGR8);
      out_mat = cv_ptr->image;
    }
    else if (images.size() == 1)
    {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(images[0], sensor_msgs::image_encodings::BGR8);
      out_mat = cv_ptr->image;
    }
    else
    {
      std::cout << "Not found the same timestamp in camera buffer." << std::endl;
    }
  }
  else
  {
    std::cout << "Not found any message in camera buffer." << std::endl;
  }
  return out_mat;
}
pcl::PointCloud<pcl::PointXYZI>::Ptr
getSpecificTimeLidarMessage(message_filters::Cache<pcl::PointCloud<pcl::PointXYZI>>& cache_lidar, ros::Time target_time,
                            ros::Duration duration_time)
{
  ros::Time begin_time = target_time - duration_time;
  ros::Time end_time = target_time + duration_time;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::ConstPtr> lidars = cache_lidar.getInterval(begin_time, end_time);
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_ptr(new pcl::PointCloud<pcl::PointXYZI>);

  if (lidars.size() > 0)
  {
    std::vector<ros::Time> lidars_time;
    for (const auto& msg : lidars)
    {
      ros::Time header_time;
      pcl_conversions::fromPCL(msg->header.stamp, header_time);
      lidars_time.push_back(header_time);
    }
    std::vector<ros::Time>::iterator it;
    it = std::find(lidars_time.begin(), lidars_time.end(), target_time);
    if (it != lidars_time.end())
    {
      int time_index = std::distance(lidars_time.begin(), it);
      *lidar_ptr = *lidars[time_index];
    }
    else if (lidars.size() == 1)
    {
      *lidar_ptr = *lidars[0];
    }
    else
    {
      std::cout << "Not found the same timestamp in lidar buffer." << std::endl;
    }
  }
  else
  {
    std::cout << "Not found any message in lidar buffer." << std::endl;
  }
  return lidar_ptr;
}
pcl::PointCloud<pcl::PointXYZIL>::Ptr
getSpecificTimeLidarMessage(message_filters::Cache<pcl::PointCloud<pcl::PointXYZIL>>& cache_lidar, ros::Time target_time,
                            ros::Duration duration_time)
{
  ros::Time begin_time = target_time - duration_time;
  ros::Time end_time = target_time + duration_time;
  std::vector<pcl::PointCloud<pcl::PointXYZIL>::ConstPtr> lidars = cache_lidar.getInterval(begin_time, end_time);
  pcl::PointCloud<pcl::PointXYZIL>::Ptr lidar_ptr(new pcl::PointCloud<pcl::PointXYZIL>);

  if (lidars.size() > 0)
  {
    std::vector<ros::Time> lidars_time;
    for (const auto& msg : lidars)
    {
      ros::Time header_time;
      pcl_conversions::fromPCL(msg->header.stamp, header_time);
      lidars_time.push_back(header_time);
    }
    std::vector<ros::Time>::iterator it;
    it = std::find(lidars_time.begin(), lidars_time.end(), target_time);
    if (it != lidars_time.end())
    {
      int time_index = std::distance(lidars_time.begin(), it);
      *lidar_ptr = *lidars[time_index];
    }
    else if (lidars.size() == 1)
    {
      *lidar_ptr = *lidars[0];
    }
    else
    {
      std::cout << "Not found the same timestamp in lidar buffer." << std::endl;
    }
  }
  else
  {
    std::cout << "Not found any message in lidar buffer." << std::endl;
  }
  return lidar_ptr;
}
void getSyncLidarCameraData()
{
  std::cout << "getSyncLidarCameraData start." << std::endl;
  bool is_camera_update = false;
  bool is_lidar_update = false;
  bool is_lidar_ssn_update = false;
  std::vector<std::vector<ros::Time>> cam_times_tmp(g_cam_ids.size());
  std::vector<ros::Time> lidarall_times_tmp;
  std::vector<ros::Time> lidar_ssn_times_tmp;
  ros::Time object_time = ros::Time(0);
  ros::Time object_past_time = ros::Time(0);

  ros::Rate loop_rate(20);
  while (ros::ok())
  {
    if (g_cam_times[0].size() > 0 && g_lidarall_times.size() > 0 && g_lidar_ssn_times.size() > 0 && !g_is_data_sync)
    {
      if (g_is_object_update[0])
      {
        is_camera_update = false;
        is_lidar_update = false;
        is_lidar_ssn_update = false;
        g_is_object_update[0] = false;

        // message sync
        g_sync_lock_objects[0].lock();
        object_time = g_object_arrs[0].header.stamp;
        g_sync_lock_objects_process.lock();
        g_object_arrs_process = g_object_arrs;
        g_sync_lock_objects_process.unlock();
        g_sync_lock_objects[0].unlock();

        // copy header timestamp vector
        for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
        {
          g_sync_lock_times[cam_order].lock();
        }
        cam_times_tmp = g_cam_times;
        for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
        {
          g_sync_lock_times[cam_order].unlock();
        }
        g_sync_lock_lidar_time.lock();
        lidarall_times_tmp = g_lidarall_times;
        g_sync_lock_lidar_time.unlock();

        g_sync_lock_lidar_ssn_time.lock();
        lidar_ssn_times_tmp = g_lidar_ssn_times;
        g_sync_lock_lidar_ssn_time.unlock();

        // show camera and lidar buffer time
        // std::cout << "--------------------------------------------------" << std::endl;
        // std::cout << "object_time: " << object_time.sec << "." << object_time.nsec << std::endl;
        if (object_time != ros::Time(0) && object_time != object_past_time)
        {
          std::vector<ros::Time>::iterator sync_times_it;
          sync_times_it = std::find(cam_times_tmp[0].begin(), cam_times_tmp[0].end(), object_time);
          int sync_time_index = std::distance(cam_times_tmp[0].begin(), sync_times_it);
          if (sync_times_it != cam_times_tmp[0].end())
          {
            ros::Duration duration_time(1);
            /// camera front center 60
            for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
            {
              cv::Mat message_mat;
              if (cam_order == 0)
              {
                message_mat = getSpecificTimeCameraMessage(g_cache_image[cam_order], object_time, duration_time);
              }
              else
              {
                ros::Time sync_camera_time = cam_times_tmp[cam_order][sync_time_index];
                if (sync_camera_time == ros::Time(0))
                {
                  for (size_t index = sync_time_index; index < cam_times_tmp[cam_order].size(); index++)
                  {
                    if (cam_times_tmp[cam_order][index] != ros::Time(0))
                    {
                      sync_camera_time = cam_times_tmp[cam_order][index];
                      break;
                    }
                  }
                }
                if(sync_camera_time != ros::Time(0))
                {
                  message_mat = getSpecificTimeCameraMessage(g_cache_image[cam_order], sync_camera_time, duration_time);
                }
              }

              if (!message_mat.empty())
              {
                g_sync_lock_cams[cam_order].lock();
                g_mats[cam_order] = message_mat;
                g_sync_lock_cams[cam_order].unlock();
                is_camera_update = true;
              }
            }
            /// lidar
            ros::Time sync_lidar_time = lidarall_times_tmp[sync_time_index];
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
            pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_ptr =
                getSpecificTimeLidarMessage(g_cache_lidar, sync_lidar_time, duration_time);
            if (lidar_ptr != NULL)
            {
              g_sync_lock_lidar_raw.lock();
              *g_lidarall_ptr = *lidar_ptr;
              g_sync_lock_lidar_raw.unlock();
              is_lidar_update = true;
            }

            /// lidar ssn
            ros::Time sync_lidar_ssn_time = lidar_ssn_times_tmp[sync_time_index];
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
            pcl::PointCloud<pcl::PointXYZIL>::Ptr lidar_ssn_ptr =
                getSpecificTimeLidarMessage(g_cache_lidar_ssn, sync_lidar_ssn_time, duration_time);
            if (lidar_ssn_ptr != NULL)
            {
              g_sync_lock_lidar_ssn.lock();
              *g_lidar_ssn_ptr = *lidar_ssn_ptr;
              g_sync_lock_lidar_ssn.unlock();
              is_lidar_ssn_update = true;
            }
          }
          else
          {
            std::cout << "Not found the same timestamp in camera time buffer." << std::endl;
          }
        }
        object_past_time = object_time;
        if (is_lidar_ssn_update && is_lidar_update && is_camera_update)
        {
          g_is_data_sync = true;
        }
      }
    }
    loop_rate.sleep();
  }
}
void runInference()
{
  std::cout << "===== runInference... =====" << std::endl;

  /// create variable
  std::vector<bool> is_object_update(g_cam_ids.size());
  std::vector<bool> is_mat_empty(g_cam_ids.size());
  std::vector<cv::Mat> cam_mats(g_cam_ids.size());
  std::vector<msgs::DetectedObjectArray> object_arrs(g_cam_ids.size());
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidarall_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_ssn_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_points_ptr(g_cam_ids.size());
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_bbox_points_ptr(g_cam_ids.size());
  std::vector<std::vector<PixelPosition>> cam_pixels(g_cam_ids.size());
  std::vector<std::vector<pcl_cube>> cams_bboxs_cube_min_max(g_cam_ids.size());

  /// init
  pclInitializer(cams_points_ptr);
  pclInitializer(cams_bbox_points_ptr);

  /// main loop
  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    if (!g_data_sync || g_is_data_sync)
    {
      g_is_data_sync = false;
      /// copy camera data
      for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
      {
        g_sync_lock_cams[cam_order].lock();  // mutex camera
        cam_mats[cam_order] = g_mats[cam_order].clone();
        g_sync_lock_cams[cam_order].unlock();  // mutex camera
        if (cam_mats[cam_order].empty())
        {
          is_mat_empty[cam_order] = true;
        }
      }
      if (!g_is_data_sync)
      {
        for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
        {
          g_sync_lock_objects[cam_order].lock();  // mutex object
        }
        object_arrs = g_object_arrs;
        is_object_update = g_is_object_update;
        for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
        {
          g_sync_lock_objects[cam_order].unlock();  // mutex object
        }
      }
      else
      {
        g_sync_lock_objects_process.lock();
        object_arrs = g_object_arrs_process;
        is_object_update = g_is_object_update;
        g_sync_lock_objects_process.unlock();
      }

      /// copy lidar data
      g_sync_lock_lidar_raw.lock();  // mutex lidar
      pcl::copyPointCloud(*g_lidarall_ptr, *lidarall_ptr);
      g_sync_lock_lidar_raw.unlock();  // mutex lidar

      g_sync_lock_lidar_ssn.lock();  // mutex lidar_ssn
      pcl::copyPointCloud(*g_lidar_ssn_ptr, *lidar_ssn_ptr);
      g_sync_lock_lidar_ssn.unlock();  // mutex lidar_ssn

      /// check object message
      // for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
      // {
      //   if (!is_object_update[cam_order])
      //   {
      //     object_arrs[cam_order].objects.clear();
      //   }
      // }

      if (lidarall_ptr->size() > 0)
      {
        /// get results
        getPointCloudInImageFOV(lidar_ssn_ptr, cams_points_ptr, cam_pixels, g_image_w, g_image_h);
        getPointCloudInBoxFOV(object_arrs, cams_points_ptr, cams_bbox_points_ptr, cam_pixels, cams_bboxs_cube_min_max);

        /// draw results on image
        drawPointCloudOnImages(cam_mats, cam_pixels, cams_bbox_points_ptr);
        drawBoxOnImages(cam_mats, object_arrs);

        /// prepare point cloud visualization
        g_sync_lock_lidar_process.lock();
        pcl::copyPointCloud(*lidarall_ptr, *g_lidarall_ptr_process);
        g_sync_lock_lidar_process.unlock();

        for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
        {
          g_sync_lock_cams_points.lock();  // mutex camera points
          pcl::copyPointCloud(*cams_points_ptr[cam_order], *g_cams_points_ptr[cam_order]);
          g_sync_lock_cams_points.unlock();  // mutex camera points

          g_sync_lock_objects_points.lock();  // mutex camera object points
          pcl::copyPointCloud(*cams_bbox_points_ptr[cam_order], *g_cams_bbox_points_ptr[cam_order]);
          g_sync_lock_objects_points.unlock();  // mutex camera object points
        }

        g_sync_lock_cube.lock();  // mutex camera points
        g_cams_bboxs_cube_min_max = cams_bboxs_cube_min_max;
        g_sync_lock_cube.unlock();  // mutex camera points

        /// prepare image visualization
        for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
        {
          g_sync_lock_cams_process.lock();  // mutex camera
          g_mats_process[cam_order] = cam_mats[cam_order].clone();
          g_sync_lock_cams_process.unlock();  // mutex camera
        }

        release(cam_pixels);
        release(cams_bboxs_cube_min_max);
      }
    }
    loop_rate.sleep();
  }
}
void buffer_monitor()
{
  /// main loop
  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    if (g_data_sync)
    {
      if (static_cast<int>(g_cam_times[0].size()) >= g_buffer_size)
      {
        for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
        {
          g_sync_lock_times[cam_order].lock();
          g_cam_times[cam_order].erase(g_cam_times[cam_order].begin(),
                                       g_cam_times[cam_order].begin() + g_buffer_size / 2);
          g_sync_lock_times[cam_order].unlock();
        }
        g_sync_lock_lidar_time.lock();
        g_lidarall_times.erase(g_lidarall_times.begin(), g_lidarall_times.begin() + g_buffer_size / 2);
        g_lidar_ssn_times.erase(g_lidar_ssn_times.begin(), g_lidar_ssn_times.begin() + g_buffer_size / 2);
        g_sync_lock_lidar_time.unlock();
      }
    }
    loop_rate.sleep();
  }
}
int main(int argc, char** argv)
{
  std::cout << "===== Alignment startup. =====" << std::endl;
  ros::init(argc, argv, "Alignment");
  ros::NodeHandle nh;

  /// ros Subscriber
  std::vector<ros::Subscriber> cam_subs(g_cam_ids.size());
  std::vector<ros::Subscriber> object_subs(g_cam_ids.size());
  ros::Subscriber lidarall;
  ros::Subscriber lidar_ssn_sub;

  /// message_filters Subscriber
  std::vector<message_filters::Subscriber<sensor_msgs::Image>> cam_filter_subs(g_cam_ids.size());
  std::vector<message_filters::Subscriber<msgs::DetectedObjectArray>> object_filter_subs(g_cam_ids.size());
  message_filters::Subscriber<pcl::PointCloud<pcl::PointXYZI>> sub_filter_lidar;
  message_filters::Subscriber<pcl::PointCloud<pcl::PointXYZIL>> sub_filter_lidar_ssn;

  /// get callback function
  static void (*f_callbacks_cam[])(const sensor_msgs::Image::ConstPtr&) = {
    callback_cam_front_60, callback_cam_top_front_120, callback_cam_left_60, callback_cam_right_60
  };
  static void (*f_callbacks_object[])(
      const msgs::DetectedObjectArray::ConstPtr&) = { callback_object_cam_front_60, callback_object_cam_top_front_120,
                                                      callback_object_cam_left_60, callback_object_cam_right_60 };

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
    lidar_ssn_sub = nh.subscribe("/squ_seg/result_cloud", 1, callback_ssn);
  }
  else
  {
    /// message_filters Subscriber
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      cam_filter_subs[cam_order].subscribe(nh, g_cam_topic_names[cam_order], g_buffer_size / 2);
      g_cache_image[cam_order].connectInput(cam_filter_subs[cam_order]);
      g_cache_image[cam_order].registerCallback(f_callbacks_cam[cam_order]);
      g_cache_image[cam_order].setCacheSize(g_buffer_size);

      object_filter_subs[cam_order].subscribe(nh, g_bbox_topic_names[cam_order], g_buffer_size / 2);
      object_filter_subs[cam_order].registerCallback(f_callbacks_object[cam_order]);
    }
    sub_filter_lidar.subscribe(nh, "/LidarAll", 1);
    g_cache_lidar.setCacheSize(g_buffer_size);
    g_cache_lidar.connectInput(sub_filter_lidar);
    g_cache_lidar.registerCallback(callback_lidarall);

    sub_filter_lidar_ssn.subscribe(nh, "/squ_seg/result_cloud", 1);
    g_cache_lidar_ssn.setCacheSize(g_buffer_size);
    g_cache_lidar_ssn.connectInput(sub_filter_lidar_ssn);
    g_cache_lidar_ssn.registerCallback(callback_ssn);    
  }

  /// class init
  alignmentInitializer();
  pclInitializer(g_cams_points_ptr);
  pclInitializer(g_cams_bbox_points_ptr);

  /// visualization
  std::thread display_lidar_thread(displayLidarData);
  std::thread display_camera_thread(displayCameraData);

  /// main loop start
  std::thread sync_data_thread;
  std::thread buffer_monitor_thread;
  if (g_data_sync)
  {
    sync_data_thread = std::thread(getSyncLidarCameraData);
    buffer_monitor_thread = std::thread(buffer_monitor);
  }
  std::thread main_thread(runInference);
  int thread_count = int(g_cam_ids.size()) * 2 + 1;  /// camera raw + object + lidar raw
  ros::MultiThreadedSpinner spinner(thread_count);
  spinner.spin();
  std::cout << "===== Alignment running... =====" << std::endl;

  /// main loop end
  display_lidar_thread.join();
  display_camera_thread.join();
  buffer_monitor_thread.join();
  if (g_data_sync)
  {
    sync_data_thread.join();
  }
  main_thread.join();
  std::cout << "===== Alignment shutdown. =====" << std::endl;
  return 0;
}
