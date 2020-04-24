/// standard
#include <iostream>
#include <mutex>
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
boost::shared_ptr<pcl::visualization::PCLVisualizer> g_viewer(new pcl::visualization::PCLVisualizer("Cloud_Viewer"));
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> g_rgb_ssn_points(g_ssn_points_ptr, 255, 255, 255);

//////////////////// for camera image
void callback_SSN(const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr& msg)
{
  g_sync_lock_ssn.lock();
  *g_ssn_ptr = *msg;
  pcl::copyPointCloud(*g_ssn_ptr, *g_ssn_points_ptr);
  g_sync_lock_ssn.unlock();
}

void pclViewerInitializer()
{
  g_viewer->initCameraParameters();
  g_viewer->addCoordinateSystem(3.0, 0, 0, 0);       // Origin(0, 0, 0)
  g_viewer->setCameraPosition(0, 0, 20, 0.2, 0, 0);  // bird view
  g_viewer->setBackgroundColor(0, 0, 0);
  g_viewer->setShowFPS(false);
}

int main(int argc, char** argv)
{
  std::cout << "===== lidar_ssn_sub_test startup. =====" << std::endl;
  ros::init(argc, argv, "lidar_ssn_sub_test");
  ros::NodeHandle nh;

  /// lidar subscriber
  ros::Subscriber ssn_sub = nh.subscribe("/squ_seg/result_cloud", 1, callback_SSN);

  /// viwer init
  pclViewerInitializer();

  /// main loop
  ros::Rate loop_rate(30);
  std::cout << "===== lidar_ssn_sub_test running... =====" << std::endl;
  while (ros::ok())
  {
    /// remove points on pcl viewer
    g_viewer->removePointCloud("Cloud viewer");
    g_sync_lock_ssn.lock();  // mutex lidar
    /// draw points on pcl viewer
    g_viewer->addPointCloud<pcl::PointXYZI>(g_ssn_points_ptr, g_rgb_ssn_points, "Cloud viewer");
    g_sync_lock_ssn.unlock();  // mutex lidar
    g_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Cloud viewer");
    
    ros::spinOnce();
    g_viewer->spinOnce();
    loop_rate.sleep();
  }
  std::cout << "===== lidar_ssn_sub_test shutdown. =====" << std::endl;
  return 0;
}
