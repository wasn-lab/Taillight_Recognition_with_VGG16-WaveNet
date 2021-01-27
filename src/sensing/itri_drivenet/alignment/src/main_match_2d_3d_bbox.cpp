/// standard
#include <iostream>
#include <thread>

/// ros
#include "ros/ros.h"
#include <std_msgs/Empty.h>
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
#include "points_in_image_area.h"
#include "point_preprocessing.h"
#include "sync_message.h"
#include "roi_fusion.h"

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
#if CAR_MODEL_IS_B1_V2 || CAR_MODEL_IS_C1
const std::vector<camera::id> g_cam_ids{ camera::id::front_bottom_60 };
#else
#error "car model is not well defined"
#endif

/// class
std::vector<Alignment> g_alignments(g_cam_ids.size());
Visualization g_visualization;
RoiFusion g_roi_fusion;

/// thread
std::vector<std::mutex> g_mutex_cams(g_cam_ids.size());
std::mutex g_mutex_data;
std::mutex g_mutex_cams_process;
std::mutex g_mutex_lidar_raw;
std::mutex g_mutex_lidar_process;
std::mutex g_mutex_cams_points;
std::mutex g_mutex_cam_times;
std::mutex g_mutex_lidar_time;
std::mutex g_mutex_lidar_detection_time;
std::vector<std::mutex> g_mutex_cam_time(g_cam_ids.size());
std::vector<std::mutex> g_mutex_cam_object_time(g_cam_ids.size());

/// params
bool g_is_data_sync = false;
bool g_is_enable_default_3d_bbox = true;
bool g_is_display = false;
bool g_img_result_publish = false;

/// ros
std::vector<message_filters::Cache<sensor_msgs::Image>> g_cache_image(g_cam_ids.size());
std::vector<message_filters::Cache<msgs::DetectedObjectArray>> g_cache_cam_object(g_cam_ids.size());
message_filters::Cache<pcl::PointCloud<pcl::PointXYZI>> g_cache_lidarall;
message_filters::Cache<msgs::DetectedObjectArray> g_cache_lidar_detection;
std::vector<std::string> g_cam_topic_names(g_cam_ids.size());
std::vector<std::string> g_bbox_topic_names(g_cam_ids.size());
std::vector<image_transport::Publisher> g_img_pubs(g_cam_ids.size());

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
pcl::PointCloud<pcl::PointXYZIL>::Ptr g_lidar_detection_ptr(new pcl::PointCloud<pcl::PointXYZIL>);
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> g_cams_points_ptr(g_cam_ids.size());
std::vector<pcl::visualization::Camera> g_cam;
ros::Time g_lidar_header_time;

/// object
ros::Publisher g_object_pub;
ros::Publisher g_heartbeat_pub;
msgs::DetectedObjectArray g_object_lidar;
std::vector<msgs::DetectedObjectArray> g_object_arrs(g_cam_ids.size());
std::vector<msgs::DetectedObjectArray> g_object_arrs_process(g_cam_ids.size());
bool g_is_lidar_object_update = false;
int g_object_wait_frame = 5;
std::vector<std::vector<msgs::DetectedObjectArray>> g_object_buffer_arrs(g_cam_ids.size());
std::vector<bool> g_is_object_update(g_cam_ids.size());

/// sync camera and lidar
int g_buffer_size = 180;
std::vector<std::vector<ros::Time>> g_cam_time_buffer(g_cam_ids.size());
std::vector<std::vector<ros::Time>> g_cam_object_time_buffer(g_cam_ids.size());
std::vector<ros::Time> g_lidarall_time_buffer;
std::vector<ros::Time> g_lidar_detection_time_buffer;
std::vector<std::vector<ros::Time>> g_cam_times(g_cam_ids.size());
std::vector<std::vector<ros::Time>> g_cam_object_times(g_cam_ids.size());
std::vector<ros::Time> g_lidarall_times;
std::vector<ros::Time> g_lidar_detection_times;

/// 3d cube
std::vector<std::vector<MinMax3D>> g_cams_bboxs_cube_min_max(g_cam_ids.size());
std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>>> g_cams_bboxs_points(g_cam_ids.size());

//////////////////// for camera data
void callback_cam_front_bottom_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_bottom_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  std::lock_guard<std::mutex> lock_cam_time(g_mutex_cam_time[cam_order]);
  g_cam_time_buffer[cam_order].push_back(msg->header.stamp);
  // std::cout << camera::topics[g_cam_ids[cam_order]] << " time: " << g_cam_time_buffer[cam_order].back().sec << "."
  // <<
  // g_cam_time_buffer[cam_order].back().nsec << std::endl;
}

void callback_object_cam_front_bottom_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_bottom_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  static int object_wait_frame_count = 0;
  if (object_wait_frame_count < g_object_wait_frame)
  {
    g_object_buffer_arrs[cam_order].push_back(*msg);
    object_wait_frame_count = object_wait_frame_count + 1;
  }
  else
  {
    g_object_buffer_arrs[cam_order].push_back(*msg);
    std::unique_lock<std::mutex> lock_cam_object_time(g_mutex_cam_object_time[cam_order], std::adopt_lock);
    g_object_arrs[cam_order] = g_object_buffer_arrs[cam_order].front();
    lock_cam_object_time.unlock();
    g_is_object_update[cam_order] = true;
    g_object_buffer_arrs[cam_order].erase(g_object_buffer_arrs[cam_order].begin());
  }
  // std::cout << camera::topics_obj[g_cam_ids[cam_order]] << " time: " << msg->header.stamp.sec << "." <<
  // msg->header.stamp.nsec << std::endl;
}

//////////////////// for LiDAR data
void callback_lidarall(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  ros::Time header_time;
  pcl_conversions::fromPCL(msg->header.stamp, header_time);
  std::lock_guard<std::mutex> lock_lidar_time(g_mutex_lidar_time);
  g_lidarall_time_buffer.push_back(header_time);
  // std::cout << "lidarall time: " << header_time.sec << "." << header_time.nsec <<
  // std::endl;
}

void callback_lidar_detection(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock_lidar_detection_time(g_mutex_lidar_detection_time);
  g_lidar_detection_time_buffer.push_back(msg->header.stamp);
  // std::cout << "lidar detection time: " << msg->header.stamp.sec << "." << msg->header.stamp.nsec <<
  // std::endl;
}

void pclViewerInitializer(const boost::shared_ptr<pcl::visualization::PCLVisualizer>& pcl_viewer)
{
  pcl_viewer->initCameraParameters();
  pcl_viewer->addCoordinateSystem(3.0, 0, 0, 0);
  pcl_viewer->setCameraPosition(0, 0, 20, 0.2, 0, 0);
  pcl_viewer->setShowFPS(false);
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
void drawLidarBoxOnImage(cv::Mat& mats, std::vector<MinMax2D>& lidar_pixels_obj)
{
  // std::cout << "===== drawLidarBoxOnImage... =====" << std::endl;
  g_visualization.drawBoxOnImage(mats, lidar_pixels_obj, sensor_msgs_itri::FusionSourceId::Lidar);
}

void drawLidarCubeOnImage(cv::Mat& mats, std::vector<std::vector<PixelPosition>>& lidar_pixels_obj)
{
  // std::cout << "===== drawLidarBoxOnImage... =====" << std::endl;
  g_visualization.drawCubeOnImage(mats, lidar_pixels_obj);
}

void drawPointCloudOnImages(std::vector<cv::Mat>& mats, std::vector<std::vector<PixelPosition>>& cam_pixels,
                            std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_points_ptr)
{
// std::cout << "===== drawPointCloudOnImages... =====" << std::endl;
#pragma omp parallel for
  for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
  {
    pcl::PointCloud<pcl::PointXYZI> point_cloud;
    point_cloud = *cams_points_ptr[cam_order];
    std::cout << "point_cloud.size(): " << point_cloud.size() << std::endl;
    for (size_t i = 0; i < point_cloud.size(); i++)
    {
      int point_u = cam_pixels[cam_order][i].u;
      int point_v = cam_pixels[cam_order][i].v;
      float point_x = point_cloud[i].x;
      g_visualization.drawPointCloudOnImage(mats[cam_order], point_u, point_v, point_x);
    }
  }
}
void drawBoxOnImages(std::vector<cv::Mat>& mats, const std::vector<msgs::DetectedObjectArray>& objects)
{
  // std::cout << "===== drawBoxOnImages... =====" << std::endl;
  for (size_t cam_order = 0; cam_order < mats.size(); cam_order++)
  {
    g_visualization.drawBoxOnImage(mats[cam_order], objects[cam_order].objects);
  }
}
void drawBoxOnImages(std::vector<cv::Mat>& mats, const std::vector<MinMax2D>& min_max_2d_bbox)
{
  // std::cout << "===== drawBoxOnImages... =====" << std::endl;
  for (size_t cam_order = 0; cam_order < mats.size(); cam_order++)
  {
    g_visualization.drawBoxOnImage(mats[cam_order], min_max_2d_bbox, sensor_msgs_itri::FusionSourceId::Camera);
  }
}
void getPointCloudInAllImageFOV(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidarall_ptr,
                                std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_points_ptr,
                                std::vector<std::vector<PixelPosition>>& cam_pixels, int image_w, int image_h)
{
// std::cout << "===== getPointCloudInImageFOV... =====" << std::endl;
#pragma omp parallel for
  for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
  {
    getPointCloudInImageFOV(lidarall_ptr, cams_points_ptr[cam_order], cam_pixels[cam_order], image_w, image_h,
                            g_alignments[cam_order]);
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
  boost::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer(new pcl::visualization::PCLVisualizer("Cloud_"
                                                                                                        "Viewer"));
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidarall_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_points_ptr(g_cams_points_ptr.size());
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb_lidarall(g_lidarall_ptr_process, 255, 255, 255);
  std::vector<pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>> rgb_cams_points;

  /// init
  pclViewerInitializer(pcl_viewer);
  pclInitializer(cams_points_ptr);
  pointsColorInit(rgb_cams_points, g_cams_points_ptr);

  MinMax3D point_50m, point_40m, point_30m, point_20m, point_10m, point_0m, point_negative_10m;
  cv::Scalar color_50m, color_40m, color_30m, color_20m, color_10m, color_0m, color_negative_10m;
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
  x_dist -= 10;
  point_0m = g_visualization.getDistLinePoint(x_dist, y_dist, z_dist);
  color_0m = g_visualization.getDistColor(x_dist);
  x_dist -= 10;
  point_negative_10m = g_visualization.getDistLinePoint(x_dist, y_dist, z_dist);
  color_negative_10m = g_visualization.getDistColor(x_dist);

  /// main loop
  ros::Rate loop_rate(10);
  while (ros::ok() && !pcl_viewer->wasStopped())
  {
    /// remove points on pcl viewer
    // pcl_viewer->removePointCloud("Cloud viewer");
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      pcl_viewer->removePointCloud(g_cam_topic_names[cam_order]);
    }
    pcl_viewer->removeAllShapes();

    /// draw points on pcl viewer
    // std::lock_guard<std::mutex> lock_lidar_process(g_mutex_lidar_process);
    // pcl_viewer->addPointCloud<pcl::PointXYZI>(g_lidarall_ptr_process, rgb_lidarall, "Cloud viewer");  //,
    // viewports[0]);

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
    pcl_viewer->addLine<pcl::PointXYZI>(point_0m.p_min, point_0m.p_max, color_0m[2], color_0m[1], color_0m[0],
                                        "line-"
                                        "0m");
    pcl_viewer->addLine<pcl::PointXYZI>(point_negative_10m.p_min, point_negative_10m.p_max, color_negative_10m[2],
                                        color_negative_10m[1], color_negative_10m[0], "line-negative-10m");
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      std::lock_guard<std::mutex> lock_cams_points(g_mutex_cams_points);  // mutex camera points
      pcl_viewer->addPointCloud<pcl::PointXYZI>(g_cams_points_ptr[cam_order], rgb_cams_points[cam_order],
                                                g_cam_topic_names[cam_order]);
    }
    pcl_viewer->spinOnce();
    loop_rate.sleep();
  }
  // pcl_viewer->close();
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

void projectLidarBBoxOntoImage(cv::Mat& mats, const msgs::DetectedObjectArray& objects_array,
                               std::vector<msgs::DetectedObject>& objects, std::vector<MinMax2D>& lidar_pixels_obj)
{
  std::vector<std::vector<PixelPosition>> lidar_pixels_obj_cube;
  getBoxInImageFOV(objects_array, objects, lidar_pixels_obj, g_alignments[0]);

  // if (g_is_display)
  // {
  //   drawLidarBoxOnImage(mats, lidar_pixels_obj);
  // }
}

void getSyncLidarCameraData()
{
  std::cout << "getSyncLidarCameraData start." << std::endl;
  bool is_camera_update = false;
  bool is_lidar_update = false;
  bool is_lidar_detection_update = false;
  bool use_system_time = false;
  std::vector<std::vector<ros::Time>> cam_times_tmp(g_cam_ids.size());
  std::vector<ros::Time> lidarall_times_tmp;
  std::vector<ros::Time> lidar_detection_times_tmp;
  std::vector<ros::Time> objects_time(g_cam_ids.size());
  ros::Time object_past_time = ros::Time(0);
  ros::Duration duration_time(3);
  ros::Duration diff_max_time(0.1);
  ros::Duration nsec_max_time(0.999999999);

  ros::Rate loop_rate(20);
  while (ros::ok())
  {
    if (!g_cam_times[0].empty() && !g_lidarall_times.empty() && !g_lidar_detection_times.empty() &&
        g_is_object_update[0] && !g_is_data_sync)
    {
      g_is_object_update[0] = false;
      is_camera_update = false;
      is_lidar_update = false;
      is_lidar_detection_update = false;

      // get time message
      for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
      {
        std::unique_lock<std::mutex> lock_cam_object_time(g_mutex_cam_object_time[cam_order], std::adopt_lock);
        g_object_arrs_process[cam_order] = g_object_arrs[cam_order];
        lock_cam_object_time.unlock();
        objects_time[cam_order] = g_object_arrs_process[cam_order].header.stamp;
      }
      std::unique_lock<std::mutex> lock_data(g_mutex_data, std::adopt_lock);
      cam_times_tmp = g_cam_times;
      lidarall_times_tmp = g_lidarall_times;
      lidar_detection_times_tmp = g_lidar_detection_times;
      lock_data.unlock();

      /// find sync data
      // std::cout << "--------------------------------------------------" << std::endl;
      /// camera
      if (objects_time[0] != ros::Time(0) && objects_time[0] != object_past_time)
      {
        ros::Time sync_cam_time = ros::Time(0);
        size_t sync_time_index = 0;
        std::vector<ros::Time>::iterator sync_times_it;
        std::vector<bool> is_cameras_update(g_cam_ids.size(), false);
        for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
        {
          // std::cout << "objects_time[" << camera::names[g_cam_ids[cam_order]] << "]: " <<
          // objects_time[cam_order].sec << "." <<
          // objects_time[cam_order].nsec << std::endl;
          sync_times_it =
              std::find(cam_times_tmp[cam_order].begin(), cam_times_tmp[cam_order].end(), objects_time[cam_order]);

          int time_index = std::distance(cam_times_tmp[cam_order].begin(), sync_times_it);
          ros::Time cam_time = cam_times_tmp[cam_order][time_index];
          if (sync_times_it != cam_times_tmp[cam_order].end() && cam_time != ros::Time(0))
          {
            // std::cout << "sync_camera_time: " << cam_time.sec << "." <<
            // cam_time.nsec <<
            // std::endl;

            if (cam_order == 0)
            {
              sync_time_index = time_index;
              sync_cam_time = cam_time;
            }

            /// get camera image
            cv::Mat message_mat;
            message_mat = getSpecificTimeCameraMessage(g_cache_image[cam_order], cam_time, duration_time);
            if (!message_mat.empty())
            {
              g_mats[cam_order] = message_mat;
              is_cameras_update[cam_order] = true;
            }

            // /// get camera object
            // g_object_camera_arrs[cam_order] =
            //     getSpecificTimeCameraObjectMessage(g_cache_cam_object[cam_order], cam_time, duration_time);
            // std::unique_lock<std::mutex> lock_data(g_mutex_data, std::adopt_lock);
            // g_cam_object_times.erase(g_cam_object_times.begin(), g_cam_object_times.begin() + 1);
            // lock_data.unlock();
          }
          else
          {
            std::cout << "Not found the same timestamp in camera " << camera::names[g_cam_ids[cam_order]]
                      << " time buffer." << std::endl;
          }
        }
        if (std::all_of(is_cameras_update.begin(), is_cameras_update.end(), [](bool v) { return v; }))
        {
          is_camera_update = true;
        }

        /// lidar
        if (is_camera_update)
        {
          ros::Time sync_lidar_time;
          if (use_system_time)
          {
            sync_lidar_time = lidarall_times_tmp[sync_time_index];
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
          }
          else
          {
            sync_lidar_time = sync_cam_time;
            ros::Duration diff_time_lidar(0);
            bool have_candidate_lidar = false;
            std::vector<int> diff_time_lidar_nsec(lidarall_times_tmp.size());
            for (size_t i = 0; i < lidarall_times_tmp.size(); i++)
            {
              diff_time_lidar = ros::Duration(0);
              diff_time_lidar_nsec[i] = nsec_max_time.nsec;

              if (lidarall_times_tmp[i] <= sync_lidar_time)
              {
                diff_time_lidar = sync_lidar_time - lidarall_times_tmp[i];
              }
              else if (lidarall_times_tmp[i] > sync_lidar_time)
              {
                diff_time_lidar = lidarall_times_tmp[i] - sync_lidar_time;
              }

              if (diff_time_lidar.sec == 0 && diff_time_lidar.nsec != 0)
              {
                if (diff_time_lidar < diff_max_time)
                {
                  // std::cout << "index: " << i << ", lidarall_times_tmp[i]: " << lidarall_times_tmp[i] << ",
                  // diff_time_lidar.nsec: " << diff_time_lidar.nsec << std::endl;
                  diff_time_lidar_nsec[i] = diff_time_lidar.nsec;
                  have_candidate_lidar = true;
                }
              }
            }
            if (have_candidate_lidar)
            {
              std::vector<int>::iterator result_iter =
                  std::min_element(diff_time_lidar_nsec.begin(), diff_time_lidar_nsec.end());
              sync_time_index = std::distance(diff_time_lidar_nsec.begin(), result_iter);
              sync_lidar_time = lidarall_times_tmp[sync_time_index];
            }
            else
            {
              sync_lidar_time = ros::Time(0);
            }
          }

          if (sync_lidar_time == ros::Time(0))
          {
            is_lidar_update = false;
            std::cout << "Not found the same timestamp in lidar time buffer." << std::endl;
          }
          else
          {
            // std::cout << "sync_lidar_time: " << sync_time_index << " : " << sync_lidar_time.sec << "." <<
            // sync_lidar_time.nsec << std::endl;
            pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_ptr =
                getSpecificTimeLidarMessage(g_cache_lidarall, sync_lidar_time, duration_time);
            if (lidar_ptr != nullptr)
            {
              *g_lidarall_ptr = *lidar_ptr;
              is_lidar_update = true;
            }
          }

          /// lidar detection
          if (is_lidar_update)
          {
            sync_times_it =
                std::find(lidar_detection_times_tmp.begin(), lidar_detection_times_tmp.end(), sync_lidar_time);
            sync_time_index = std::distance(lidar_detection_times_tmp.begin(), sync_times_it);

            if (sync_time_index == lidar_detection_times_tmp.size())
            {
              ros::Duration diff_time_detection;
              bool have_candidate_detection = false;
              std::vector<int> diff_time_nsec_detection(lidar_detection_times_tmp.size());
              for (size_t i = 0; i < lidar_detection_times_tmp.size(); i++)
              {
                diff_time_nsec_detection[i] = nsec_max_time.nsec;
                if (lidar_detection_times_tmp[i] <= sync_lidar_time)
                {
                  diff_time_detection = sync_lidar_time - lidar_detection_times_tmp[i];
                  if (diff_time_detection.sec == 0)
                  {
                    if (diff_time_detection < diff_max_time)
                    {
                      // std::cout << "index: " << i << ", lidar_detection_times_tmp[i]: "
                      // << lidar_detection_times_tmp[i] << ", diff_time_detection.nsec: " << diff_time_detection.nsec
                      // << std::endl;
                      diff_time_nsec_detection[i] = diff_time_detection.nsec;
                      have_candidate_detection = true;
                    }
                  }
                }
              }
              if (have_candidate_detection)
              {
                std::vector<int>::iterator result_iter =
                    std::min_element(diff_time_nsec_detection.begin(), diff_time_nsec_detection.end());
                sync_time_index = std::distance(diff_time_nsec_detection.begin(), result_iter);
              }
            }

            if (sync_time_index < lidar_detection_times_tmp.size())
            {
              ros::Time sync_lidar_detection_time = lidar_detection_times_tmp[sync_time_index];

              if (sync_lidar_detection_time == ros::Time(0))
              {
                for (size_t index = sync_time_index; index < lidar_detection_times_tmp.size(); index++)
                {
                  if (lidar_detection_times_tmp[index] != ros::Time(0))
                  {
                    sync_lidar_detection_time = lidar_detection_times_tmp[index];
                    sync_time_index = index;
                    break;
                  }
                }
              }
              if (sync_lidar_detection_time == ros::Time(0))
              {
                is_lidar_detection_update = false;
              }
              else
              {
                // std::cout << "sync_lidar_detection_time: " << sync_lidar_detection_time.sec << "." <<
                // sync_lidar_detection_time.nsec
                // <<
                // std::endl;
                g_object_lidar = getSpecificTimeLidarObjectMessage(g_cache_lidar_detection, sync_lidar_detection_time,
                                                                   duration_time);
                g_lidar_header_time = sync_lidar_detection_time;
                is_lidar_detection_update = true;
              }
            }
            else
            {
              std::cout << "Not found the same timestamp in lidar detection time buffer." << std::endl;
            }
          }
        }
      }
      else
      {
        std::cout << "Not found the same timestamp in camera time buffer." << std::endl;
      }
      object_past_time = objects_time[0];
      if (is_camera_update && is_lidar_update && is_lidar_detection_update)
      {
        g_is_data_sync = true;
      }
    }
    loop_rate.sleep();
  }
  std::cout << "getSyncLidarCameraData close." << std::endl;
}

void image_publisher(const std::vector<cv::Mat>& image, const std_msgs::Header& header)
{
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    sensor_msgs::ImagePtr img_msg;
    img_msg = cv_bridge::CvImage(header, "bgr8", image[cam_order]).toImageMsg();
    g_img_pubs[cam_order].publish(img_msg);
  }
}
void object_publisher(const msgs::DetectedObjectArray& object_array_camera,
                      const msgs::DetectedObjectArray& object_array_lidar,
                      const std::vector<msgs::DetectedObject>& object_array_lidar_filter,
                      const std::vector<std::pair<int, int>>& fusion_index)
{
  msgs::DetectedObjectArray msg_det_obj_arr;
  std::vector<msgs::DetectedObject> msg_objs;

  std_msgs::Header msg_header;
  // msg_header.stamp = ros_time;

  for (size_t pair_index = 0; pair_index < fusion_index.size(); pair_index++)
  {
    int camera_index = fusion_index[pair_index].first;
    int lidar_index = fusion_index[pair_index].second;
    // std::cout << "fusion_index[" << pair_index << "]: " << fusion_index[pair_index].first << ", " <<
    // fusion_index[pair_index].second << std::endl;

    msgs::DetectedObject msg_obj;
    msg_obj.header = object_array_lidar.header;
    msg_obj.fusionSourceId = sensor_msgs_itri::FusionSourceId::Camera;
    msg_obj.distance = 0;

    /// detection result
    msg_obj.classId = object_array_camera.objects[camera_index].classId;
    msg_obj.camInfo = object_array_camera.objects[camera_index].camInfo;
    msg_obj.bPoint = object_array_lidar_filter[lidar_index].bPoint;
    msg_obj.center_point = object_array_lidar_filter[lidar_index].center_point;
    msg_obj.heading = object_array_lidar_filter[lidar_index].heading;
    msg_obj.dimension = object_array_lidar_filter[lidar_index].dimension;

    msg_objs.push_back(msg_obj);
  }

  msg_det_obj_arr.header = std::move(msg_header);
  msg_det_obj_arr.header.frame_id = "lidar";  // mapping to lidar coordinate
  msg_det_obj_arr.objects = msg_objs;
  g_object_pub.publish(msg_det_obj_arr);

  std_msgs::Empty empty_msg;
  g_heartbeat_pub.publish(empty_msg);
}
void runInference()
{
  std::cout << "===== runInference... =====" << std::endl;

  /// create variable
  ros::Time lidar_header_time;
  std::vector<cv::Mat> cam_mats(g_cam_ids.size());
  std::vector<msgs::DetectedObjectArray> object_arrs(g_cam_ids.size());
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidarall_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  msgs::DetectedObjectArray object_lidar;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_points_ptr(g_cam_ids.size());
  std::vector<std::vector<PixelPosition>> cam_pixels(g_cam_ids.size());
  std::vector<msgs::DetectedObject> object_lidar_filter;
  std::vector<std::vector<MinMax2D>> lidar_pixels_obj(g_cam_ids.size());
  std::vector<std::vector<MinMax2D>> cam_pixels_obj(g_cam_ids.size());
  std::vector<std::vector<MinMax3D>> cams_bboxs_cube_min_max(g_cam_ids.size());

  /// init
  pclInitializer(cams_points_ptr);

  /// main loop
  ros::Rate loop_rate(20);
  while (ros::ok())
  {
    if (g_is_data_sync)
    {
      /// copy camera data
      for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
      {
        cam_mats[cam_order] = g_mats[cam_order].clone();
      }
      object_arrs = g_object_arrs_process;
      pcl::copyPointCloud(*g_lidarall_ptr, *lidarall_ptr);
      object_lidar = g_object_lidar;
      lidar_header_time = g_lidar_header_time;

      /// get results
      std::cout << "===== doInference once =====" << std::endl;
      g_is_data_sync = false;
      // getPointCloudInAllImageFOV(lidarall_ptr, cams_points_ptr, cam_pixels, g_image_w, g_image_h);
      projectLidarBBoxOntoImage(cam_mats[0], object_lidar, object_lidar_filter, lidar_pixels_obj[0]);
      std::vector<sensor_msgs::RegionOfInterest> object_lidar_roi = g_roi_fusion.getLidar2DROI(lidar_pixels_obj[0]);
      std::vector<sensor_msgs::RegionOfInterest> object_camera_roi = g_roi_fusion.getCam2DROI(object_arrs[0]);
      std::vector<std::pair<int, int>> fusion_index =
          g_roi_fusion.getRoiFusionResult(object_camera_roi, object_lidar_roi);
      g_roi_fusion.getFusionCamObj(object_arrs[0], fusion_index, cam_pixels_obj[0]);
      object_publisher(object_arrs[0], object_lidar, object_lidar_filter, fusion_index);

      if (g_img_result_publish)
      {
        drawBoxOnImages(cam_mats, object_arrs);        // camera detection result
        drawBoxOnImages(cam_mats, cam_pixels_obj[0]);  // fusion camera result
        image_publisher(cam_mats, object_lidar.header);
      }

      if (g_is_display)
      {
        /// draw results on image
        // drawPointCloudOnImages(cam_mats, cam_pixels, cams_points_ptr);
        // drawBoxOnImages(cam_mats, object_arrs);
        if (!g_img_result_publish)
        {
          drawBoxOnImages(cam_mats, cam_pixels_obj[0]);
        }

        /// prepare image visualization
        std::lock_guard<std::mutex> lock_cams_process(g_mutex_cams_process);
        for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
        {
          g_mats_process[cam_order] = cam_mats[cam_order].clone();
        }

        /// prepare point cloud visualization
        // std::lock_guard<std::mutex> lock_lidar_process(g_mutex_lidar_process);
        // pcl::copyPointCloud(*lidarall_ptr, *g_lidarall_ptr_process);

        std::lock_guard<std::mutex> lock_cams_points(g_mutex_cams_points);
        for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
        {
          pcl::copyPointCloud(*cams_points_ptr[cam_order], *g_cams_points_ptr[cam_order]);
        }
      }

      release(cam_pixels);
      release(lidar_pixels_obj);
      release(cam_pixels_obj);
      object_lidar_filter.clear();
    }
    loop_rate.sleep();
  }
  std::cout << "===== runInference close =====" << std::endl;
}

void buffer_monitor()
{
  std::vector<ros::Time> cam_time_last(g_cam_ids.size());
  std::vector<ros::Time> cam_object_time_last(g_cam_ids.size());
  ros::Time lidarall_time_last = ros::Time(0);
  ros::Time lidar_detection_time_last = ros::Time(0);
  bool cam_object_update = false;
  /// main loop
  ros::Rate loop_rate(20);
  while (ros::ok())
  {
    // Add buffer
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      std::unique_lock<std::mutex> lock_cam_time(g_mutex_cam_time[cam_order], std::adopt_lock);
      if (!g_cam_time_buffer[cam_order].empty())
      {
        cam_time_last[cam_order] = g_cam_time_buffer[cam_order].front(); /* store last timestamp */

        // std::cout  << camera::topics[g_cam_ids[cam_order]] << " cam_time_last: " <<
        // cam_time_last[cam_order].sec << "." <<
        // cam_time_last[cam_order].nsec << " store" <<
        // std::endl;
        g_cam_time_buffer[cam_order].erase(g_cam_time_buffer[cam_order].begin());
      }
      lock_cam_time.unlock();

      // std::unique_lock<std::mutex> lock_cam_object_time(g_mutex_cam_object_time[cam_order], std::adopt_lock);
      // if (!g_cam_object_time_buffer[cam_order].empty())
      // {
      //   cam_object_time_last[cam_order] = g_cam_object_time_buffer[cam_order].front(); /* store last timestamp */
      //   cam_object_update = true;
      //   // std::cout  << camera::topics[g_cam_ids[cam_order]] << " cam_object_time_last: " <<
      //   // cam_object_time_last[cam_order].sec << "." <<
      //   // cam_object_time_last[cam_order].nsec << " store" <<
      //   // std::endl;
      //   g_cam_object_time_buffer[cam_order].erase(g_cam_object_time_buffer[cam_order].begin());
      // }
      // else
      // {
      //   cam_object_update = false;
      // }

      // lock_cam_object_time.unlock();
    }

    std::unique_lock<std::mutex> lock_lidar_time(g_mutex_lidar_time, std::adopt_lock);
    if (!g_lidarall_time_buffer.empty())
    {
      lidarall_time_last = g_lidarall_time_buffer.front();  // store last timestamp

      // std::cout  <<"lidarall_time_last:    " << g_lidarall_time_buffer.front().sec << "." <<
      // g_lidarall_time_buffer.front().nsec << " store" <<
      // std::endl;
      g_lidarall_time_buffer.erase(g_lidarall_time_buffer.begin());
    }
    lock_lidar_time.unlock();

    std::unique_lock<std::mutex> lock_lidar_detection_time(g_mutex_lidar_detection_time, std::adopt_lock);
    if (!g_lidar_detection_time_buffer.empty())
    {
      lidar_detection_time_last = g_lidar_detection_time_buffer.front();  // store last timestamp
      // std::cout  <<"lidar_detection_time_last:    " << lidar_detection_time_last.sec << "." <<
      // lidar_detection_time_last.nsec << " store" <<
      // std::endl;

      g_lidar_detection_time_buffer.erase(g_lidar_detection_time_buffer.begin());
    }

    lock_lidar_detection_time.unlock();

    std::unique_lock<std::mutex> lock_data(g_mutex_data, std::adopt_lock);
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      g_cam_times[cam_order].push_back(cam_time_last[cam_order]);
      if (cam_object_update)
      {
        g_cam_object_times[cam_order].push_back(cam_object_time_last[cam_order]);
      }
    }
    g_lidarall_times.push_back(lidarall_time_last);
    g_lidar_detection_times.push_back(lidar_detection_time_last);
    lock_data.unlock();

    // Clear buffer
    if (static_cast<int>(g_cam_times[0].size()) >= g_buffer_size)
    {
      std::unique_lock<std::mutex> lock_data(g_mutex_data, std::adopt_lock);
      for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
      {
        g_cam_times[cam_order].erase(g_cam_times[cam_order].begin(),
                                     g_cam_times[cam_order].begin() + g_buffer_size / 3);
        // g_cam_object_times[cam_order].erase(g_cam_object_times[cam_order].begin(),
        //                               g_cam_object_times[cam_order].begin() + g_buffer_size / 3);
      }
      g_lidarall_times.erase(g_lidarall_times.begin(), g_lidarall_times.begin() + g_buffer_size / 3);
      g_lidar_detection_times.erase(g_lidar_detection_times.begin(),
                                    g_lidar_detection_times.begin() + g_buffer_size / 3);
      lock_data.unlock();
    }
    loop_rate.sleep();
  }
}
int main(int argc, char** argv)
{
  std::cout << "===== match_2d_3d_bbox startup. =====" << std::endl;
  ros::init(argc, argv, "match_2d_3d_bbox");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  ros::param::get(ros::this_node::getName() + "/display", g_is_display);
  ros::param::get(ros::this_node::getName() + "/imgResult_publish", g_img_result_publish);

  /// ros Subscriber
  std::vector<ros::Subscriber> cam_subs(g_cam_ids.size());
  std::vector<ros::Subscriber> object_subs(g_cam_ids.size());
  ros::Subscriber lidarall_sub;
  ros::Subscriber lidar_detection_sub;

  /// message_filters Subscriber
  std::vector<message_filters::Subscriber<sensor_msgs::Image>> cam_filter_subs(g_cam_ids.size());
  std::vector<message_filters::Subscriber<msgs::DetectedObjectArray>> object_filter_subs(g_cam_ids.size());
  message_filters::Subscriber<pcl::PointCloud<pcl::PointXYZI>> sub_filter_lidarall;
  message_filters::Subscriber<pcl::PointCloud<pcl::PointXYZI>> sub_filter_lidarall_nonground;
  message_filters::Subscriber<msgs::DetectedObjectArray> sub_filter_lidar_detection;

  /// get callback function
  static void (*f_callbacks_cam[])(const sensor_msgs::Image::ConstPtr&) = { callback_cam_front_bottom_60 };
  static void (*f_callbacks_object[])(
      const msgs::DetectedObjectArray::ConstPtr&) = { callback_object_cam_front_bottom_60 };

  /// set topic name
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    g_cam_topic_names[cam_order] = camera::topics[g_cam_ids[cam_order]];
    g_bbox_topic_names[cam_order] = camera::topics_obj[g_cam_ids[cam_order]];
  }

  /// message_filters Subscriber
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    cam_filter_subs[cam_order].subscribe(nh, g_cam_topic_names[cam_order], 1);
    g_cache_image[cam_order].connectInput(cam_filter_subs[cam_order]);
    g_cache_image[cam_order].registerCallback(f_callbacks_cam[cam_order]);
    g_cache_image[cam_order].setCacheSize(g_buffer_size);

    object_filter_subs[cam_order].subscribe(nh, g_bbox_topic_names[cam_order], 1);
    g_cache_cam_object[cam_order].connectInput(object_filter_subs[cam_order]);
    g_cache_cam_object[cam_order].registerCallback(f_callbacks_object[cam_order]);
    g_cache_cam_object[cam_order].setCacheSize(g_buffer_size);

    if (g_img_result_publish)
    {
      g_img_pubs[cam_order] = it.advertise(
          camera::detect_result + std::string("/") + g_cam_topic_names[cam_order] + std::string("/detect_image"), 1);
    }
  }

  sub_filter_lidarall.subscribe(nh, "/LidarAll", 1);
  g_cache_lidarall.setCacheSize(g_buffer_size);
  g_cache_lidarall.connectInput(sub_filter_lidarall);
  g_cache_lidarall.registerCallback(callback_lidarall);

  sub_filter_lidar_detection.subscribe(nh, "/LidarDetection", 1);
  g_cache_lidar_detection.setCacheSize(g_buffer_size);
  g_cache_lidar_detection.connectInput(sub_filter_lidar_detection);
  g_cache_lidar_detection.registerCallback(callback_lidar_detection);

  /// object publisher
  g_object_pub = nh.advertise<msgs::DetectedObjectArray>(camera::detect_result, 8);
  g_heartbeat_pub = nh.advertise<std_msgs::Empty>(camera::detect_result + std::string("/heartbeat"), 1);

  /// class init
  alignmentInitializer();
  pclInitializer(g_cams_points_ptr);

  /// visualization
  std::thread display_lidar_thread;
  std::thread display_camera_thread;
  if (g_is_display)
  {
    display_lidar_thread = std::thread(displayLidarData);
    display_camera_thread = std::thread(displayCameraData);
  }
  /// sync
  std::thread buffer_monitor_thread(buffer_monitor);
  std::thread sync_data_thread(getSyncLidarCameraData);

  /// main loop start
  std::thread main_thread(runInference);
  int thread_count = int(g_cam_ids.size()) * 2 + 2;  /// camera raw + lidar raw + objects
  ros::MultiThreadedSpinner spinner(thread_count);
  spinner.spin();
  std::cout << "===== match_2d_3d_bbox running... =====" << std::endl;

  /// main loop end
  if (g_is_display)
  {
    display_lidar_thread.join();
    display_camera_thread.join();
  }
  /// sync
  buffer_monitor_thread.join();
  sync_data_thread.join();

  main_thread.join();
  std::cout << "===== match_2d_3d_bbox shutdown. =====" << std::endl;
  return 0;
}