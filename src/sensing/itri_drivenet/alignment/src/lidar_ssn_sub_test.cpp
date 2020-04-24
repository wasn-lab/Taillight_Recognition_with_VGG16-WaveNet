/// standard
#include <iostream>
#include <mutex>
#include <thread>

/// ros
#include "ros/ros.h"

/// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>

/// lidar lib
#include "UserDefine.h"

/// thread
std::mutex g_sync_lock_ssn;

/// lidar
pcl::PointCloud<pcl::PointXYZIL>::Ptr g_ssn_ptr(new pcl::PointCloud<pcl::PointXYZIL>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_ssn_points_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_ssn_points(g_ssn_points_ptr, 255, 255, 255);

//////////////////// for camera image
void callback_SSN(const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr& msg)
{
  g_sync_lock_ssn.lock();
  *g_ssn_ptr = *msg;
  pcl::copyPointCloud(*g_ssn_ptr, *g_ssn_points_ptr);
  g_sync_lock_ssn.unlock();
}

void pclViewerInitializer(boost::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer)
{
  pcl_viewer->initCameraParameters();
  pcl_viewer->addCoordinateSystem(3.0, 0, 0, 0);       // Origin(0, 0, 0)
  pcl_viewer->setCameraPosition(0, 0, 20, 0.2, 0, 0);  // bird view
  pcl_viewer->setBackgroundColor(0, 0, 0);
  pcl_viewer->setShowFPS(false);
}

void displayLidarData()
{
  // std::cout << "===== displayLidarData... =====" << std::endl;
  /// create variable
  boost::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer(
      new pcl::visualization::PCLVisualizer("Cloud_Viewer"));
  pcl::PointCloud<pcl::PointXYZI>::Ptr ssn_points_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb_lidarall(ssn_points_ptr, 255, 255, 255);

  /// init
  pclViewerInitializer(pcl_viewer);

  /// main loop
  ros::Rate loop_rate(30);
  while (ros::ok() && !pcl_viewer->wasStopped())
  {
    /// remove points on pcl viewer
    pcl_viewer->removePointCloud("Cloud viewer");

    g_sync_lock_ssn.lock();  // mutex lidar
    pcl::copyPointCloud(*g_ssn_points_ptr, *ssn_points_ptr);
    g_sync_lock_ssn.unlock();  // mutex lidar

    /// draw points on pcl viewer
    pcl_viewer->addPointCloud<pcl::PointXYZI>(ssn_points_ptr, rgb_lidarall, "Cloud viewer");
    pcl_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Cloud viewer");
    
    pcl_viewer->spinOnce();
    loop_rate.sleep();
  }  
}
int main(int argc, char** argv)
{
  std::cout << "===== lidar_ssn_sub_test startup. =====" << std::endl;
  ros::init(argc, argv, "lidar_ssn_sub_test");
  ros::NodeHandle nh;

  /// lidar subscriber
  ros::Subscriber ssn_sub = nh.subscribe("/squ_seg/result_cloud", 1, callback_SSN);

  std::thread display_lidar_thread(displayLidarData);
  int thread_count = 2;  /// camera raw + object + lidar raw
  ros::MultiThreadedSpinner spinner(thread_count);
  spinner.spin();
  std::cout << "===== lidar_ssn_sub_test running... =====" << std::endl;

  display_lidar_thread.join();
  std::cout << "===== lidar_ssn_sub_test shutdown. =====" << std::endl;
  return 0;
}
