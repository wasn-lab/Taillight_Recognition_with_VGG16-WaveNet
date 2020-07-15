/// standard
#include <iostream>
#include <thread>

/// ros
#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

/// package
#include "camera_params.h"
#include "alignment.h"
#include "visualization_util.h"
#include <drivenet/object_label_util.h>
#include "point_preprocessing.h"
#include "points_in_image_area.h"
#include "point_preprocessing.h"

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
                                         camera::id::right_back_60, camera::id::left_back_60 };
#else
#error "car model is not well defined"
#endif

/// class
std::vector<Alignment> g_alignments(g_cam_ids.size());
Visualization g_visualization;

/// thread
std::vector<std::mutex> g_mutex_cams(g_cam_ids.size());
std::mutex g_mutex_cams_process;
std::mutex g_mutex_lidar_raw;
std::mutex g_mutex_lidar_process;
std::mutex g_mutex_cams_points;
std::mutex g_mutex_cam_times;
std::mutex g_mutex_lidar_time;
std::vector<std::mutex> g_mutex_cam_time(g_cam_ids.size());

/// params
bool g_is_enable_default_3d_bbox = true;
bool g_is_display = true;

/// ros
std::vector<std::string> g_cam_topic_names(g_cam_ids.size());

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
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> g_cams_points_ptr(g_cam_ids.size());
std::vector<pcl::visualization::Camera> g_cam;

/// object
ros::Publisher g_bbox_pub, g_polygon_pub;
std::vector<msgs::DetectedObjectArray> g_object_arrs(g_cam_ids.size());
std::vector<msgs::DetectedObjectArray> g_object_arrs_process(g_cam_ids.size());

/// sync camera and lidar
int g_buffer_size = 60;
std::vector<std::vector<ros::Time>> g_cam_times(g_cam_ids.size());
std::vector<ros::Time> g_cam_single_time(g_cam_ids.size());
std::vector<ros::Time> g_lidarall_times;
ros::Time g_lidarall_time;

/// 3d cube
std::vector<std::vector<MinMax3D>> g_cams_bboxs_cube_min_max(g_cam_ids.size());
std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>>> g_cams_bboxs_points(g_cam_ids.size());

//////////////////// for camera image
void callback_cam_front_bottom_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_bottom_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);

  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  std::lock_guard<std::mutex> lock_cams(g_mutex_cams[cam_order]);
  g_mats[cam_order] = cv_ptr->image;
}

void callback_cam_front_top_far_30(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_top_far_30);
  int cam_order = std::distance(g_cam_ids.begin(), it);

  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  std::lock_guard<std::mutex> lock_cams(g_mutex_cams[cam_order]);
  g_mats[cam_order] = cv_ptr->image;
}

void callback_cam_right_back_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_back_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);

  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  std::lock_guard<std::mutex> lock_cams(g_mutex_cams[cam_order]);
  g_mats[cam_order] = cv_ptr->image;
}

void callback_cam_left_back_60(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_back_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);

  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  std::lock_guard<std::mutex> lock_cams(g_mutex_cams[cam_order]);
  g_mats[cam_order] = cv_ptr->image;
}

void callback_lidarall(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(g_mutex_lidar_raw);
  *g_lidarall_ptr = *msg;
  // std::cout << "Point cloud size: " << g_lidarall_ptr->size() << std::endl;
  // std::cout << "Lidar x: " << g_lidarall_ptr->points[0].x << ", y: " << g_lidarall_ptr->points[0].y << ", z: " <<
  // g_lidarall_ptr->points[0].z << std::endl;
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
                                std::vector<std::vector<PixelPosition>>& cam_pixels, int image_w, int image_h)
{
  // std::cout << "===== getPointCloudInImageFOV... =====" << std::endl;
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
  boost::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer(
      new pcl::visualization::PCLVisualizer("Cloud_Viewer"));
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
    pcl_viewer->addLine<pcl::PointXYZI>(point_0m.p_min, point_0m.p_max, color_0m[2], color_0m[1], color_0m[0], "line-"
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

void runInference()
{
  std::cout << "===== runInference... =====" << std::endl;

  /// create variable
  bool is_data_ready = true;
  std::vector<cv::Mat> cam_mats(g_cam_ids.size());
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidarall_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_points_ptr(g_cam_ids.size());
  std::vector<std::vector<PixelPosition>> cam_pixels(g_cam_ids.size());
  std::vector<std::vector<MinMax3D>> cams_bboxs_cube_min_max(g_cam_ids.size());

  /// init
  pclInitializer(cams_points_ptr);

  /// main loop
  ros::Rate loop_rate(20);
  while (ros::ok())
  {
    is_data_ready = true;

    /// copy camera data
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      std::lock_guard<std::mutex> lock_cams(g_mutex_cams[cam_order]);
      cam_mats[cam_order] = g_mats[cam_order].clone();
      if (cam_mats[cam_order].empty())
      {
        is_data_ready = false;
        std::cout << "cam_mats " << cam_order << "is empty" << std::endl;
      }
    }

    /// copy lidar data
    std::lock_guard<std::mutex> lock_lidar_raw(g_mutex_lidar_raw);
    pcl::copyPointCloud(*g_lidarall_ptr, *lidarall_ptr);

    if (lidarall_ptr->empty())
    {
      is_data_ready = false;
      std::cout << "lidarall is empty" << std::endl;
    }

    if (is_data_ready)
    {
      std::cout << "===== doInference once =====" << std::endl;
      /// get results
      getPointCloudInAllImageFOV(lidarall_ptr, cams_points_ptr, cam_pixels, g_image_w, g_image_h);

      if (g_is_display)
      {
        /// draw results on image
        drawPointCloudOnImages(cam_mats, cam_pixels, cams_points_ptr);

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
    }
    loop_rate.sleep();
  }
  std::cout << "===== runInference close =====" << std::endl;
}

int main(int argc, char** argv)
{
  std::cout << "===== Alignment_visualization startup. =====" << std::endl;
  ros::init(argc, argv, "Alignment_visualization");
  ros::NodeHandle nh;

  /// ros Subscriber
  std::vector<ros::Subscriber> cam_subs(g_cam_ids.size());
  ros::Subscriber lidarall;

  /// get callback function
  static void (*f_callbacks_cam[])(const sensor_msgs::Image::ConstPtr&) = {
    callback_cam_front_bottom_60, callback_cam_front_top_far_30, callback_cam_right_back_60, callback_cam_left_back_60
  };

  /// set topic name
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    g_cam_topic_names[cam_order] = camera::topics[g_cam_ids[cam_order]];
  }

  /// ros Subscriber
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    cam_subs[cam_order] = nh.subscribe(g_cam_topic_names[cam_order], 1, f_callbacks_cam[cam_order]);
  }

  lidarall = nh.subscribe("/LidarAll", 1, callback_lidarall);

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

  /// main loop start
  std::thread main_thread(runInference);
  int thread_count = int(g_cam_ids.size()) * 2 + 1;  /// camera raw + object + lidar raw
  ros::MultiThreadedSpinner spinner(thread_count);
  spinner.spin();
  std::cout << "===== Alignment_visualization running... =====" << std::endl;

  /// main loop end
  if (g_is_display)
  {
    display_lidar_thread.join();
    display_camera_thread.join();
  }
  main_thread.join();
  std::cout << "===== Alignment_visualization shutdown. =====" << std::endl;
  return 0;
}