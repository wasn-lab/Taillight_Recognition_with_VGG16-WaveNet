#include "all_header.h"
#include "RosModuleHINO.hpp"
#include "CuboidFilter.h"
#include "VoxelGrid_CUDA.h"
#include "VoxelFilter_CUDA.h"
#include "PlaneGroundFilter.h"
#include "RayGroundFilter.h"
#include "extract_Indices.h"

#if ENABLE_DEBUG_MODE == true
#include "UI/QtViewer.h"
#endif

void callback_LidarAll(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  if (msg->size() > 100)
  {
    StopWatch stopWatch;

    pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_cur_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    *ptr_cur_cloud = *msg;
    // cout << "[raw data       ]:" << ptr_cur_cloud->size () << endl;

    *ptr_cur_cloud = CuboidFilter().pass_through_soild<PointXYZI>(ptr_cur_cloud, -30, 60, -30, 30, -5, 0);
    *ptr_cur_cloud = CuboidFilter().hollow_removal<PointXYZI>(ptr_cur_cloud, -4, 1.18, -1.3, 1.3, -5, 0);
    *ptr_cur_cloud = CuboidFilter().hollow_removal<PointXYZI>(ptr_cur_cloud, 1.3, 2.5, 1, 2, -1.5, 0);

    // cout << "[pass through   ]:" << ptr_cur_cloud->size () << "," << timer_algorithm_running.getTimeSeconds () << "s"
    // << endl;

    PointCloud<PointXYZI>::Ptr cloud_ground(new PointCloud<PointXYZI>);
    PointCloud<PointXYZI>::Ptr cloud_non_ground(new PointCloud<PointXYZI>);
    pcl::PointIndicesPtr indices_ground(new pcl::PointIndices);

    pcl::PointCloud<pcl::PointXYZ>::Ptr buff(new pcl::PointCloud<pcl::PointXYZ>);
    copyPointCloud(*ptr_cur_cloud, *buff);

    *indices_ground = RayGroundFilter(3.12, 5.8, 9.0, 0.01, 0.01, 0.15, 0.3, 0.8, 0.175).compute<PointXYZ>(buff);
    extract_Indices<PointXYZI>(ptr_cur_cloud, indices_ground, *cloud_ground, *cloud_non_ground);

    *indices_ground = PlaneGroundFilter().runMorphological<PointXYZI>(cloud_non_ground, 0.3, 2, 1, 0.9, 0.2, 0.3);
    extract_Indices<PointXYZI>(cloud_non_ground, indices_ground, *cloud_ground, *cloud_non_ground);

    // cout << "[remove ground]:" << timer_algorithm_running.getTimeSeconds () << "s"<< endl;

    if (cloud_ground->size() < 100)
    {
      RosModuleHINO::send_ErrorCode(0x4000);
      cout << "error: not find ground" << endl;
    }

    // RosModuleHINO::send_Rviz (*cloud_non_ground);
    RosModuleHINO::send_LidarAllNonGround(*cloud_non_ground, msg->header.stamp, msg->header.frame_id);

    if (stopWatch.getTimeSeconds() > 0.05)
    {
      cout << "[Preprocess]: slow" << stopWatch.getTimeSeconds() << "s" << endl << endl;
    }
  }
}

#if ENABLE_DEBUG_MODE == true
void UI(int argc, char** argv)
{
  QApplication a(argc, argv);
  QtViewer w;
  w.show();
  a.exec();
}
#endif

int main(int argc, char** argv)
{
  RosModuleHINO::initial("lidars_preprocessing", argc, argv);
  RosModuleHINO::RegisterCallBackLidarAll(callback_LidarAll);
  cout << "[" << ros::this_node::getName() << "] "
       << "----------------------------startup" << endl;

  cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  cout.precision(3);

#if ENABLE_DEBUG_MODE == true
  thread TheadDetection(UI, argc, argv);
#endif

  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }
  RosModuleHINO::send_ErrorCode(0x0006);

  cout << "[" << ros::this_node::getName() << "] "
       << "----------------------------end" << endl;
  return (0);
}
