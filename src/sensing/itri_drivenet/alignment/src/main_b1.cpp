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

/// thread
#include <mutex>

/// class
Alignment g_alignment;

/// thread
mutex g_syncLock;

/// params
bool g_isCompressed = false;

/// image
int g_image_w_ = camera::image_width;
int g_image_h_ = camera::image_height;
cv::Mat g_mat60_1;
cv::Mat g_mat60_1_raw;

/// lidar
pcl::PointCloud<pcl::PointXYZI> g_lidarall_nonground;

/// object
std::vector<msgs::DetectedObject> g_object_60_1;

//////////////////// for camera image
void callback_60_1(const sensor_msgs::Image::ConstPtr& msg)
{
  g_syncLock.lock();
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_mat60_1 = cv_ptr->image;
  // cv::resize(g_mat60_1, g_mat60_1_raw, cv::Size(1920, 1208));
  std_msgs::Header h = msg->header;
  g_syncLock.unlock();
}

//////////////////// for camera image
void callback_60_1_decode(sensor_msgs::CompressedImage compressImg)
{
  g_syncLock.lock();
  cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(g_mat60_1);
  g_syncLock.unlock();
}

//////////////////// for camera object
void callback_object_60_1(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  g_syncLock.lock();
  g_object_60_1 = msg->objects;
  g_syncLock.unlock();
  // std::cout << camera::topics_obj[camera::id::front_60] << " size: " << g_object_60_1.size() << std::endl;
  // cout<< camera::topics_obj[camera::id::front_60] <<endl;
}

/// similar to above, this is just a backup and testing for printing lidar data ///
void lidarAllCallback (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  g_syncLock.lock();
  pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_cur_cloud (new pcl::PointCloud<pcl::PointXYZI>);
  *ptr_cur_cloud = *msg;
  g_lidarall_nonground = *ptr_cur_cloud;
  g_syncLock.unlock();
  // std::cout << "Point cloud size: " << g_lidarall_nonground.size() << std::endl;
  // std::cout << "Lidar x: " << g_lidarall_nonground.points[0].x << ", y: " << g_lidarall_nonground.points[0].y << ", z: " << g_lidarall_nonground.points[0].z << std::endl;
}

void drawPointCloudOnImage()
{
  for (size_t i = 0; i < g_lidarall_nonground.size(); i++)
  {
    if(g_lidarall_nonground.points[i].x > 0)
    {
      // std::cout << "Lidar x: " << g_lidarall_nonground.points[i].x << ", y: " << g_lidarall_nonground.points[i].y << ", z: " << g_lidarall_nonground.points[i].z << std::endl;
      PixelPosition pixel_position_;
      pixel_position_ = g_alignment.projectPointToPixel(g_lidarall_nonground.points[i]);
      if(pixel_position_.u >= 0 && pixel_position_.v >= 0)
      {
        cv::Point center_point_ = cv::Point(pixel_position_.u, pixel_position_.v);
        cv::circle(g_mat60_1, center_point_, 1, cv::Scalar(0, 255, 0), -1, LINE_8, 0);
        // std::cout << "Camera u: " << pixel_position_.u << ", v: " << pixel_position_.v << std::endl;
      }
    }
  }
}
int main (int argc, char **argv)
{
    std::cout << "===== Alignment startup. =====" << std::endl;
    ros::init(argc, argv, "Alignment");
    ros::NodeHandle nh;

    /// camera subscriber
    ros::Subscriber cam60_1;
    std::string cam60_1_topicName = camera::topics[camera::id::front_60];
    if (g_isCompressed)
    {
      cam60_1 = nh.subscribe(cam60_1_topicName + std::string("/compressed"), 1, callback_60_1_decode);
    }
    else
    {
      cam60_1 = nh.subscribe(cam60_1_topicName, 1, callback_60_1);
    }
    ros::Subscriber cam60_1_detection_sub;
    std::string cam60_1_object_topicName = camera::topics_obj[camera::id::front_60];
    cam60_1_detection_sub = nh.subscribe(cam60_1_object_topicName, 1, callback_object_60_1);

    /// lidar subscriber
    ros::Subscriber lidarall;
    lidarall = nh.subscribe("/LidarAll", 1, lidarAllCallback);

    /// init
    g_alignment.projectMatrixInit(camera::id::front_60);

    std::string window_name_cam60_1 = cam60_1_topicName;
    cv::namedWindow(window_name_cam60_1, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_cam60_1, 480, 360);
    cv::moveWindow(window_name_cam60_1, 1025, 30);

    ros::Rate loop_rate(30);
    std::cout << "===== Alignment running... =====" << std::endl;
    while (ros::ok())
    {
        ros::spinOnce();
        if(!g_mat60_1.empty())
        {
          g_syncLock.lock();
          drawPointCloudOnImage();
          cv::imshow(window_name_cam60_1, g_mat60_1);
          g_syncLock.unlock();
          cv::waitKey(1);
        }
        loop_rate.sleep();
    } 
    std::cout << "===== Alignment shutdown. =====" << std::endl;
    return 0;

}
