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

/// Camera lib
#include "drivenet/object_label_util.h"

/// namespace
using namespace DriveNet;

/// thread
std::mutex g_sync_lock_ssn;

/// lidar
pcl::PointCloud<pcl::PointXYZIL>::Ptr g_ssn_ptr(new pcl::PointCloud<pcl::PointXYZIL>);

//////////////////// for camera image
void callback_SSN(const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr& msg)
{
  g_sync_lock_ssn.lock();
  *g_ssn_ptr = *msg;
  g_sync_lock_ssn.unlock();
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

int transferCommonLabelToSSNLabel(common_type_id label_id)
{
  int nn_label_id = -1;
  switch (label_id)
  {
    case common_type_id::person:
      nn_label_id = nnClassID::Person;  // Person
      break;
    case common_type_id::bicycle:
      nn_label_id = nnClassID::Motobike;  // bicycle
      break;
    case common_type_id::motorbike:
      nn_label_id = nnClassID::Motobike;  // motobike
      break;
    case common_type_id::car:
      nn_label_id = nnClassID::Car;  // car
      break;
    case common_type_id::bus:
      nn_label_id = nnClassID::Car;  // bus
      break;
    case common_type_id::truck:
      nn_label_id = nnClassID::Car;  // truck
      break;
    default:
      break;
  }
  return nn_label_id;
}
pcl::PointCloud<pcl::PointXYZI>::Ptr getClassObjectPoint(pcl::PointCloud<pcl::PointXYZIL>::Ptr points_ptr, common_type_id label_id)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr out_points_ptr(new pcl::PointCloud<pcl::PointXYZI>); 
  int nn_label_id = -1;
  nn_label_id = transferCommonLabelToSSNLabel(label_id);
  for(size_t index = 0; index < points_ptr->size(); index++ )
  {
    if (points_ptr->points[index].label == nn_label_id)
    {
      pcl::PointXYZI point;
      point.x = points_ptr->points[index].x;
      point.y = points_ptr->points[index].y;
      point.z = points_ptr->points[index].z;
      point.intensity = points_ptr->points[index].intensity;
      out_points_ptr->push_back(point);
    }
  }
  return out_points_ptr;
}
pcl::PointCloud<pcl::PointXYZI>::Ptr getClassObjectPoint(pcl::PointCloud<pcl::PointXYZIL>::Ptr points_ptr, nnClassID label_id)
{ 
  pcl::PointCloud<pcl::PointXYZI>::Ptr out_points_ptr(new pcl::PointCloud<pcl::PointXYZI>); 
  int nn_label_id = -1;
  nn_label_id = label_id;
  for(size_t index = 0; index < points_ptr->size(); index++ )
  {
    if (points_ptr->points[index].label == nn_label_id)
    {
      pcl::PointXYZI point;
      point.x = points_ptr->points[index].x;
      point.y = points_ptr->points[index].y;
      point.z = points_ptr->points[index].z;
      point.intensity = points_ptr->points[index].intensity;
      out_points_ptr->push_back(point);
    }
  }
  return out_points_ptr;
}
void displayLidarData()
{
  // std::cout << "===== displayLidarData... =====" << std::endl;
  /// create variable
  boost::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer(
      new pcl::visualization::PCLVisualizer("Cloud_Viewer"));
  pcl::PointCloud<pcl::PointXYZIL>::Ptr ssn_ptr(new pcl::PointCloud<pcl::PointXYZIL>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr ssn_points_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr person_points_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr car_points_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr motorbike_points_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr rule_base_points_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb_ssn_points(ssn_points_ptr, 255, 255, 255);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb_person_points(ssn_points_ptr, 255, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb_car_points(ssn_points_ptr, 0, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb_motorbike_points(ssn_points_ptr, 0, 0, 255);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb_rule_base_points(ssn_points_ptr, 255, 255, 255);
  std::vector<int> viewports{ 1, 2 };
  std::vector<std::string> view_name{ "raw data", "object" };

  /// init
  pclViewerInitializer(pcl_viewer, view_name, static_cast<int>(viewports.size()));

  /// main loop
  ros::Rate loop_rate(30);
  while (ros::ok() && !pcl_viewer->wasStopped())
  {
    if(g_ssn_ptr->size() > 0)
    {
      g_sync_lock_ssn.lock();  // mutex lidar
      pcl::copyPointCloud(*g_ssn_ptr, *ssn_points_ptr);
      pcl::copyPointCloud(*g_ssn_ptr, *ssn_ptr);
      g_sync_lock_ssn.unlock();  // mutex lidar

      /// remove points on pcl viewer
      pcl_viewer->removePointCloud("Cloud viewer", viewports[0]);
      pcl_viewer->removePointCloud("Person cloud", viewports[1]);
      pcl_viewer->removePointCloud("Car cloud", viewports[1]);
      pcl_viewer->removePointCloud("Motorbike cloud", viewports[1]);
      pcl_viewer->removePointCloud("Rule_base cloud", viewports[1]);

      person_points_ptr = getClassObjectPoint(ssn_ptr, nnClassID::Person);
      car_points_ptr = getClassObjectPoint(ssn_ptr, nnClassID::Car);
      motorbike_points_ptr = getClassObjectPoint(ssn_ptr, nnClassID::Motobike);
      rule_base_points_ptr = getClassObjectPoint(ssn_ptr, nnClassID::Rule);

      /// draw points on pcl viewer
      pcl_viewer->addPointCloud<pcl::PointXYZI>(ssn_points_ptr, rgb_ssn_points, "Cloud viewer", viewports[0]);
      pcl_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Cloud viewer");
      pcl_viewer->addPointCloud<pcl::PointXYZI>(person_points_ptr, rgb_person_points, "Person cloud", viewports[1]);
      pcl_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Person cloud");
      pcl_viewer->addPointCloud<pcl::PointXYZI>(car_points_ptr, rgb_car_points, "Car cloud", viewports[1]);
      pcl_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Car cloud");
      pcl_viewer->addPointCloud<pcl::PointXYZI>(motorbike_points_ptr, rgb_motorbike_points, "Motorbike cloud", viewports[1]);
      pcl_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Motorbike cloud");
      pcl_viewer->addPointCloud<pcl::PointXYZI>(rule_base_points_ptr, rgb_rule_base_points, "Rule_base cloud", viewports[1]);
      pcl_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Rule_base cloud");

      pcl_viewer->spinOnce();
    }
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
