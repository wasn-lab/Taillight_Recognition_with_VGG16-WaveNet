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
std::vector<std::mutex> g_sync_lock_objects(g_cam_ids.size());
mutex g_sync_lock_lidar;

/// params
bool g_is_compressed = false;

/// image
int g_image_w = camera::image_width;
int g_image_h = camera::image_height;
int g_raw_image_w = camera::raw_image_width;
int g_raw_image_h = camera::raw_image_height;
std::vector<cv::Mat> g_mats(g_cam_ids.size());
std::vector<std::vector<PixelPosition>> g_cam_pixels(g_cam_ids.size());

/// lidar
pcl::PointCloud<pcl::PointXYZI>::Ptr g_lidarall_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cam_front_60_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cam_front_60_bbox_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cam_top_front_120_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cam_top_front_120_bbox_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cam_left_60_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cam_left_60_bbox_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cam_right_60_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cam_right_60_bbox_ptr(new pcl::PointCloud<pcl::PointXYZI>);
boost::shared_ptr<pcl::visualization::PCLVisualizer> g_viewer(new pcl::visualization::PCLVisualizer("Cloud_Viewer"));
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_lidarall(g_lidarall_ptr, 255, 255, 255);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_front_60(g_cam_front_60_ptr, 255, 255, 0);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_top_front_120(g_cam_top_front_120_ptr, 255,
                                                                                         0, 0);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_left_60(g_cam_left_60_ptr, 0, 0, 255);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_right_60(g_cam_right_60_ptr, 0, 0, 255);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_front_60(g_cam_front_60_ptr, 255, 242, 230);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_top_front_120(g_cam_top_front_120_ptr, 179, 86, 0);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_left_60(g_cam_left_60_ptr, 255, 163, 77);
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_right_60(g_cam_right_60_ptr, 255, 163, 77);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_front_60_bbox(g_cam_front_60_bbox_ptr, 255, 255, 255);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_top_front_120_bbox(g_cam_top_front_120_bbox_ptr, 255, 255, 255);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_left_60_bbox(g_cam_left_60_bbox_ptr, 255, 255, 255);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_cam_right_60_bbox(g_cam_right_60_bbox_ptr, 255, 255, 255);
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
  g_objects[cam_order] = msg->objects;
  g_sync_lock_objects[cam_order].unlock();
}
void callback_object_cam_top_front_120(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::top_front_120);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_objects[cam_order].lock();
  g_objects[cam_order] = msg->objects;
  g_sync_lock_objects[cam_order].unlock();
}
void callback_object_cam_left_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::left_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_objects[cam_order].lock();
  g_objects[cam_order] = msg->objects;
  g_sync_lock_objects[cam_order].unlock();
}
void callback_object_cam_right_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::right_60);
  int cam_order = std::distance(g_cam_ids.begin(), it);
  g_sync_lock_objects[cam_order].lock();
  g_objects[cam_order] = msg->objects;
  g_sync_lock_objects[cam_order].unlock();
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

void drawBoxOnImages()
{
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    g_visualization.drawBoxOnImage(g_mats[cam_order], g_objects[cam_order]);
  }
}
void drawPointCloudOnImages()
{
  pcl::PointCloud<pcl::PointXYZI> point_cloud;
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    if (cam_order == 0)
    {
      point_cloud = *g_cam_front_60_ptr;
    }
    else if (cam_order == 1)
    {
      point_cloud = *g_cam_top_front_120_ptr;
    }
    else if (cam_order == 2)
    {
      point_cloud = *g_cam_left_60_ptr;
    }
    else if (cam_order == 3)
    {
      point_cloud = *g_cam_right_60_ptr;
    }
    for (size_t i = 0; i < point_cloud.size(); i++)
    {
      int point_u = g_cam_pixels[cam_order][i].u;
      int point_v = g_cam_pixels[cam_order][i].v;
      float point_x = point_cloud[i].x;
      g_visualization.drawPointCloudOnImage(g_mats[cam_order], point_u, point_v, point_x);
    }
  }
}
void getPointCloudInImageFOV()
{
  pcl::copyPointCloud(*g_lidarall_ptr, *g_cam_front_60_ptr);
  pcl::copyPointCloud(*g_lidarall_ptr, *g_cam_top_front_120_ptr);
  pcl::copyPointCloud(*g_lidarall_ptr, *g_cam_left_60_ptr);
  pcl::copyPointCloud(*g_lidarall_ptr, *g_cam_right_60_ptr);
  std::vector<pcl::PointCloud<pcl::PointXYZI>> cam_points = { *g_cam_front_60_ptr, *g_cam_top_front_120_ptr,
                                                              *g_cam_left_60_ptr, *g_cam_right_60_ptr };
  std::vector<int> cloud_sizes(g_cam_ids.size(), 0);
  std::vector<std::vector<std::vector<pcl::PointXYZI>>> point_cloud(g_cam_ids.size() , std::vector<std::vector<pcl::PointXYZI>> (g_image_w, std::vector<pcl::PointXYZI>(g_image_h)));
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
          if(point_cloud[cam_order][pixel_position.u][pixel_position.v].x > g_lidarall_ptr->points[i].x || point_cloud[cam_order][pixel_position.u][pixel_position.v].x == 0)
          {
            point_cloud[cam_order][pixel_position.u][pixel_position.v] = g_lidarall_ptr->points[i];
          }
        }
      }
    }
  }
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    for(int u = 0; u < g_image_w; u++)
    {
      for(int v = 0; v < g_image_h; v++)
      {
        PixelPosition pixel_position{ -1, -1 };
        pixel_position.u = u;
        pixel_position.v = v;
        if(point_cloud[cam_order][u][v].x !=0 && point_cloud[cam_order][u][v].y !=0 && point_cloud[cam_order][u][v].z !=0)
        {
          g_cam_pixels[cam_order].push_back(pixel_position);
          cam_points[cam_order].points[cloud_sizes[cam_order]] = point_cloud[cam_order][u][v];
          cloud_sizes[cam_order]++;
        }
      }
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
void getPointCloudIn3DBox(const pcl::PointCloud<pcl::PointXYZI> cloud_src, int object_class_id, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_ptr)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointXYZI minPt, maxPt;

  // get the box length of object 
  pcl::getMinMax3D(cloud_src, minPt, maxPt);
  object_box bbox;
  bbox = getDefaultObjectBox(object_class_id);

  // build the condition
  pcl::ConditionAnd<pcl::PointXYZI>::Ptr range_cond (new pcl::ConditionAnd<pcl::PointXYZI> ());
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZI>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZI> ("x", pcl::ComparisonOps::GT, minPt.x)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZI>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZI> ("x", pcl::ComparisonOps::LT, minPt.x + bbox.length)));

  // build the filter
  pcl::ConditionalRemoval<pcl::PointXYZI> condrem;
  condrem.setCondition (range_cond);
  cloud_ptr = cloud_src.makeShared();
  condrem.setInputCloud (cloud_ptr);
  condrem.setKeepOrganized (false);

  // apply filter
  condrem.filter (*cloud_filtered_ptr);
}
void getPointCloudInBoxFOV()
{
  pcl::copyPointCloud(*g_cam_front_60_ptr, *g_cam_front_60_bbox_ptr);
  pcl::copyPointCloud(*g_cam_top_front_120_ptr, *g_cam_top_front_120_bbox_ptr);
  pcl::copyPointCloud(*g_cam_left_60_ptr, *g_cam_left_60_bbox_ptr);
  pcl::copyPointCloud(*g_cam_right_60_ptr, *g_cam_right_60_bbox_ptr);
  std::vector<pcl::PointCloud<pcl::PointXYZI>> cam_points = { *g_cam_front_60_bbox_ptr, *g_cam_top_front_120_bbox_ptr,
                                                              *g_cam_left_60_bbox_ptr, *g_cam_right_60_bbox_ptr };
  std::vector<int> cloud_sizes(g_cam_ids.size(), 0);
  pcl::PointCloud<pcl::PointXYZI> point_cloud_src;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<pcl::PointXYZI> point_vector_object;
  std::vector<pcl::PointXYZI> point_vector_objects;

  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    if (cam_order == 0)
    {
      point_cloud_src = *g_cam_front_60_bbox_ptr;
    }
    else if (cam_order == 1)
    {
      point_cloud_src = *g_cam_top_front_120_bbox_ptr;
    }
    else if (cam_order == 2)
    {
      point_cloud_src = *g_cam_left_60_bbox_ptr;
    }
    else if (cam_order == 3)
    {
      point_cloud_src = *g_cam_right_60_bbox_ptr;
    }
    for (const auto& obj: g_objects[cam_order])
    {
      for (size_t i = 0; i < point_cloud_src.points.size(); i++)
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
        pixel_position = g_alignments[cam_order].projectPointToPixel(point_cloud_src.points[i]);
        if (pixel_position.u >= bbox_positions[0].u && pixel_position.v >= bbox_positions[0].v && pixel_position.u <= bbox_positions[1].u && pixel_position.v <= bbox_positions[1].v)
        {
          point_vector_object.push_back(point_cloud_src.points[i]);
        }
      }

      // vector to point cloud
      pcl::PointCloud<pcl::PointXYZI> point_cloud_object;
      point_cloud_object.points.resize (point_vector_object.size());
      for (size_t i = 0; i < point_vector_object.size(); i++)
      {
        point_cloud_object.points[i] = point_vector_object[i];
      }
      point_vector_object.clear();
      
      // get points in the 3d box
      getPointCloudIn3DBox(point_cloud_object, obj.classId, cloud_filtered_ptr);

      // point cloud to vector
      for (const auto& point: cloud_filtered_ptr->points)
      {
        point_vector_object.push_back(point);
      }
    
      // Concatenate the points of objects
      point_vector_objects.insert (point_vector_objects.begin(), point_vector_object.begin(), point_vector_object.end());
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
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    cam_points[cam_order].resize(cloud_sizes[cam_order]);
  }
  *g_cam_front_60_bbox_ptr = cam_points[0];
  *g_cam_top_front_120_bbox_ptr = cam_points[1];
  *g_cam_left_60_bbox_ptr = cam_points[2];
  *g_cam_right_60_bbox_ptr = cam_points[3];
}
void release()
{
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    g_cam_pixels[cam_order].clear();
  }
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
    // g_viewer->removePointCloud("Cloud viewer");
    g_viewer->removePointCloud("Front 60 Cloud viewer");
    g_viewer->removePointCloud("Top Front 120 Cloud viewer");
    g_viewer->removePointCloud("Left 60 Cloud viewer");
    g_viewer->removePointCloud("Right 60 Cloud viewer");
    g_viewer->removePointCloud("Front 60 BBox Cloud viewer");
    g_viewer->removePointCloud("Top Front 120 BBox Cloud viewer");
    g_viewer->removePointCloud("Left 60 BBox Cloud viewer");
    g_viewer->removePointCloud("Right 60 BBox Cloud viewer");

    g_sync_lock_lidar.lock();  // mutex lidar
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      g_sync_lock_cams[cam_order].lock();  // mutex camera
      g_sync_lock_objects[cam_order].lock();
    }
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      if (!g_mats[cam_order].empty())
      {
        /// draw lidar on cv viewer
        getPointCloudInImageFOV();
        getPointCloudInBoxFOV();
        drawPointCloudOnImages();
        drawBoxOnImages();
        cv::imshow(cam_topic_names[cam_order], g_mats[cam_order]);
      }
    }
    /// draw points on pcl viewer
    // g_viewer->addPointCloud<pcl::PointXYZI>(g_lidarall_ptr, g_rgb_lidarall, "Cloud viewer");
    g_viewer->addPointCloud<pcl::PointXYZI>(g_cam_top_front_120_ptr, g_rgb_cam_top_front_120, "Top Front 120 Cloud viewer");
    g_viewer->addPointCloud<pcl::PointXYZI>(g_cam_front_60_ptr, g_rgb_cam_front_60, "Front 60 Cloud viewer");

    g_viewer->addPointCloud<pcl::PointXYZI>(g_cam_left_60_ptr, g_rgb_cam_left_60, "Left 60 Cloud viewer");
    g_viewer->addPointCloud<pcl::PointXYZI>(g_cam_right_60_ptr, g_rgb_cam_right_60, "Right 60 Cloud viewer");
    g_viewer->addPointCloud<pcl::PointXYZI>(g_cam_front_60_bbox_ptr, g_rgb_cam_front_60_bbox, "Front 60 BBox Cloud viewer");
    g_viewer->addPointCloud<pcl::PointXYZI>(g_cam_top_front_120_bbox_ptr, g_rgb_cam_top_front_120_bbox, "Top Front 120 BBox Cloud viewer");
    g_viewer->addPointCloud<pcl::PointXYZI>(g_cam_left_60_bbox_ptr, g_rgb_cam_left_60_bbox, "Left 60 BBox Cloud viewer");
    g_viewer->addPointCloud<pcl::PointXYZI>(g_cam_right_60_bbox_ptr, g_rgb_cam_right_60_bbox, "Right 60 BBox Cloud viewer");

    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      g_sync_lock_objects[cam_order].unlock();
      g_sync_lock_cams[cam_order].unlock();  // mutex camera
    }
    g_sync_lock_lidar.unlock();  // mutex lidar
    cv::waitKey(1);
    release();

    ros::spinOnce();
    g_viewer->spinOnce();
    loop_rate.sleep();
  }
  std::cout << "===== Alignment shutdown. =====" << std::endl;
  return 0;
}
