/// standard
#include <iostream>
#include <thread>

/// ros
#include "ros/ros.h"
#include <msgs/DetectedObjectArray.h>
#include <msgs/DetectedObject.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

/// package
#include "camera_params.h"
#include "alignment.h"
#include "visualization_util.h"
#include <drivenet/object_label_util.h>
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
mutex g_sync_lock_lidar_raw;
mutex g_sync_lock_lidar_process;
std::mutex g_sync_lock_cams_points;
std::mutex g_sync_lock_objects_points;

/// params
bool g_is_compressed = false;
bool g_is_enable_default_3d_bbox = false;
std::vector<bool> g_is_object_update(g_cam_ids.size());

/// image
int g_image_w = camera::image_width;
int g_image_h = camera::image_height;
int g_raw_image_w = camera::raw_image_width;
int g_raw_image_h = camera::raw_image_height;
std::vector<cv::Mat> g_mats(g_cam_ids.size());
std::vector<cv::Mat> g_mats_preocess(g_cam_ids.size());
std::vector<std::string> g_cam_topic_names(g_cam_ids.size());
std::vector<std::string> g_bbox_topic_names(g_cam_ids.size());

/// lidar
pcl::PointCloud<pcl::PointXYZI>::Ptr g_lidarall_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_lidarall_ptr_process(new pcl::PointCloud<pcl::PointXYZI>);
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> g_cams_points_ptr(g_cam_ids.size());
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> g_cams_bbox_points_ptr(g_cam_ids.size());
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_front_60(g_cam_front_60_ptr, 255, 255, 0);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_top_front_120(g_cam_top_front_120_ptr,
// 255,
//                                                                                          0, 0);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_left_60(g_cam_left_60_ptr, 0, 0, 255);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_right_60(g_cam_right_60_ptr, 0, 0, 255);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_front_60(g_cam_front_60_ptr, 255, 242,
// 230);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_top_front_120(g_cam_top_front_120_ptr,
// 179, 86, 0);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_left_60(g_cam_left_60_ptr, 255, 163, 77);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_right_60(g_cam_right_60_ptr, 255, 163,
// 77);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_front_60_bbox(g_cam_front_60_bbox_ptr,
// 255, 255, 255);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
// g_rgb_cam_top_front_120_bbox(g_cam_top_front_120_bbox_ptr, 255, 255, 255);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_left_60_bbox(g_cam_left_60_bbox_ptr, 255,
// 255, 255);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_right_60_bbox(g_cam_right_60_bbox_ptr,
// 255, 255, 255);
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
  g_sync_lock_objects[cam_order].lock();
  g_is_object_update[cam_order] = true;
  g_objects[cam_order] = msg->objects;
  g_sync_lock_objects[cam_order].unlock();
}
void callback_object_cam_top_front_120(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::top_front_120);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_objects[cam_order].lock();
  g_is_object_update[cam_order] = true;
  g_objects[cam_order] = msg->objects;
  g_sync_lock_objects[cam_order].unlock();
}
void callback_object_cam_left_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_objects[cam_order].lock();
  g_is_object_update[cam_order] = true;
  g_objects[cam_order] = msg->objects;
  g_sync_lock_objects[cam_order].unlock();
}
void callback_object_cam_right_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_objects[cam_order].lock();
  g_is_object_update[cam_order] = true;
  g_objects[cam_order] = msg->objects;
  g_sync_lock_objects[cam_order].unlock();
}
void callback_lidarall(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  g_sync_lock_lidar_raw.lock();
  *g_lidarall_ptr = *msg;
  g_sync_lock_lidar_raw.unlock();
  // std::cout << "Point cloud size: " << g_lidarall_ptr->size() << std::endl;
  // std::cout << "Lidar x: " << g_lidarall_ptr->points[0].x << ", y: " << g_lidarall_ptr->points[0].y << ", z: " <<
  // g_lidarall_ptr->points[0].z << std::endl;
}

void pclViewerInitializer(boost::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer,
                          std::vector<std::string> window_name, int window_count = 2)
{
  if (window_name.size() < 2)
  {
    window_name.clear();
    window_name.push_back("raw_data");
    window_name.push_back("object");
  }
  if (window_count < 2)
  {
    window_count = 2;
  }

  int v1 = 1, v2 = 2;
  pcl_viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
  pcl_viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
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
    cv::resizeWindow(g_cam_topic_names[cam_order], 480, 360);
    cv::moveWindow(g_cam_topic_names[cam_order], 1025, 30);
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

void drawBoxOnImages(std::vector<cv::Mat>& mats, std::vector<std::vector<msgs::DetectedObject>>& objects)
{
  // std::cout << "===== drawBoxOnImages... =====" << std::endl;
  for (size_t cam_order = 0; cam_order < mats.size(); cam_order++)
  {
    g_visualization.drawBoxOnImage(mats[cam_order], objects[cam_order]);
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
          cam_pixels[cam_order].push_back(pixel_position);
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

void getPointCloudInBoxFOV(std::vector<std::vector<msgs::DetectedObject>>& objects,
                           std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_points_ptr,
                           std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_bbox_points_ptr)
{
  // std::cout << "===== getPointCloudInBoxFOV... =====" << std::endl;
  /// create variable
  std::vector<pcl::PointCloud<pcl::PointXYZI>> cam_points(cams_points_ptr.size());
  std::vector<int> cloud_sizes(cam_points.size(), 0);
  pcl::PointCloud<pcl::PointXYZI> point_cloud_src;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<pcl::PointXYZI> point_vector_object;
  std::vector<pcl::PointXYZI> point_vector_objects;

  /// copy from source
  for (size_t cam_order = 0; cam_order < cam_points.size(); cam_order++)
  {
    pcl::copyPointCloud(*cams_points_ptr[cam_order], *cams_bbox_points_ptr[cam_order]);
    cam_points[cam_order] = *cams_bbox_points_ptr[cam_order];
  }

  /// main
  for (size_t cam_order = 0; cam_order < cam_points.size(); cam_order++)
  {
    for (const auto& obj : objects[cam_order])
    {
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
          point_vector_object.push_back(cam_points[cam_order].points[i]);
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
    }
    removeDuplePoints(point_vector_objects);
    for (size_t i = 0; i < point_vector_objects.size(); i++)
    {
      cam_points[cam_order].points[i] = point_vector_objects[i];
      cloud_sizes[cam_order]++;
    }
    point_vector_objects.clear();
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
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb_lidarall(g_lidarall_ptr_process, 0, 255, 255);
  std::vector<pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>> rgb_cams_points;
  std::vector<pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>> rgb_cams_bbox_points;
  std::vector<int> viewports{ 1, 2 };
  std::vector<std::string> view_name{ "raw data", "object" };

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
    // pcl_viewer->removePointCloud("Cloud viewer", viewports[0]);
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      pcl_viewer->removePointCloud(g_cam_topic_names[cam_order], viewports[0]);
      pcl_viewer->removePointCloud(g_bbox_topic_names[cam_order], viewports[1]);
    }

    /// draw points on pcl viewer
    // g_sync_lock_lidar_process.lock();
    // pcl_viewer->addPointCloud<pcl::PointXYZI>(g_lidarall_ptr_process, rgb_lidarall, "Cloud viewer", viewports[0]);
    // g_sync_lock_lidar_process.unlock();
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      g_sync_lock_cams_points.lock();  // mutex camera points
      pcl_viewer->addPointCloud<pcl::PointXYZI>(g_cams_points_ptr[cam_order], rgb_cams_points[cam_order],
                                                g_cam_topic_names[cam_order], viewports[0]);
      g_sync_lock_cams_points.unlock();   // mutex camera points
      g_sync_lock_objects_points.lock();  // mutex camera object points
      pcl_viewer->addPointCloud<pcl::PointXYZI>(g_cams_bbox_points_ptr[cam_order], rgb_cams_bbox_points[cam_order],
                                                g_bbox_topic_names[cam_order], viewports[1]);
      g_sync_lock_objects_points.unlock();  // mutex camera object points
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
      if (!g_mats_preocess[cam_order].empty())
      {
        g_sync_lock_cams_process.lock();  // mutex camera
        cv::imshow(g_cam_topic_names[cam_order], g_mats_preocess[cam_order]);
        g_sync_lock_cams_process.unlock();  // mutex camera
      }
    }
    cv::waitKey(1);
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
  std::vector<std::vector<msgs::DetectedObject>> objects(g_cam_ids.size());
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidarall_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_points_ptr(g_cam_ids.size());
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cams_bbox_points_ptr(g_cam_ids.size());
  std::vector<std::vector<PixelPosition>> cam_pixels(g_cam_ids.size());

  /// init
  pclInitializer(cams_points_ptr);
  pclInitializer(cams_bbox_points_ptr);

  /// main loop
  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    /// copy camera data
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      g_sync_lock_cams[cam_order].lock();  // mutex camera
      cam_mats[cam_order] = g_mats[cam_order].clone();
      if (cam_mats[cam_order].empty())
      {
        is_mat_empty[cam_order] = true;
      }
      g_sync_lock_cams[cam_order].unlock();  // mutex camera
    }
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      g_sync_lock_objects[cam_order].lock();  // mutex object
    }
    objects = g_objects;
    is_object_update = g_is_object_update;
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      g_sync_lock_objects[cam_order].unlock();  // mutex object
    }
    /// copy lidar data
    g_sync_lock_lidar_raw.lock();  // mutex lidar
    pcl::copyPointCloud(*g_lidarall_ptr, *lidarall_ptr);
    g_sync_lock_lidar_raw.unlock();  // mutex lidar

    /// check object message
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      if (!is_object_update[cam_order])
      {
        objects[cam_order].clear();
      }
    }

    if (g_lidarall_ptr->size() > 0)
    {
      /// get results
      getPointCloudInImageFOV(lidarall_ptr, cams_points_ptr, cam_pixels, g_image_w, g_image_h);
      getPointCloudInBoxFOV(objects, cams_points_ptr, cams_bbox_points_ptr);

      /// draw results on image
      drawPointCloudOnImages(cam_mats, cam_pixels, cams_points_ptr);
      drawBoxOnImages(cam_mats, objects);

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

      /// prepare image visualization
      for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
      {
        g_sync_lock_cams_process.lock();  // mutex camera
        g_mats_preocess[cam_order] = cam_mats[cam_order].clone();
        g_sync_lock_cams_process.unlock();  // mutex camera
      }

      release(cam_pixels);
    }
    loop_rate.sleep();
  }
}
int main(int argc, char** argv)
{
  std::cout << "===== Alignment startup. =====" << std::endl;
  ros::init(argc, argv, "Alignment");
  ros::NodeHandle nh;

  /// camera subscriber
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
    g_cam_topic_names[cam_order] = camera::topics[g_cam_ids[cam_order]];
    g_bbox_topic_names[cam_order] = camera::topics_obj[g_cam_ids[cam_order]];

    if (g_is_compressed)
    {
      cam_subs[cam_order] =
          nh.subscribe(g_cam_topic_names[cam_order] + std::string("/compressed"), 1, f_callbacks_cam_decode[cam_order]);
    }
    else
    {
      cam_subs[cam_order] = nh.subscribe(g_cam_topic_names[cam_order], 1, f_callbacks_cam[cam_order]);
    }
    object_subs[cam_order] = nh.subscribe(g_bbox_topic_names[cam_order], 1, f_callbacks_object[cam_order]);
  }

  /// lidar subscriber
  ros::Subscriber lidarall;
  lidarall = nh.subscribe("/LidarAll", 1, callback_lidarall);

  /// class init
  alignmentInitializer();
  pclInitializer(g_cams_points_ptr);
  pclInitializer(g_cams_bbox_points_ptr);

  // visualization
  std::thread display_lidar_thread(displayLidarData);
  std::thread display_camera_thread(displayCameraData);

  /// main loop start
  std::thread main_thread(runInference);
  int thread_count = int(g_cam_ids.size()) * 2 + 1;  /// camera raw + object + lidar raw
  ros::MultiThreadedSpinner spinner(thread_count);
  spinner.spin();
  std::cout << "===== Alignment running... =====" << std::endl;

  /// main loop end
  main_thread.join();
  display_lidar_thread.join();
  display_camera_thread.join();
  std::cout << "===== Alignment shutdown. =====" << std::endl;
  return 0;
}
