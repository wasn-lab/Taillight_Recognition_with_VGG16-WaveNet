#include "all_header.h"
#include "RosModuleHINO.hpp"
#include "CuboidFilter.h"
#include "VoxelGrid_CUDA.h"
#include "VoxelFilter_CUDA.h"
#include "PlaneGroundFilter.h"
#include "extract_Indices.h"

void
callback_LidarAll (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  if (msg->size () > 100)
  {
    StopWatch stopWatch;

    pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_cur_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    *ptr_cur_cloud = *msg;
    //cout << "[raw data       ]:" << ptr_cur_cloud->size () << endl;

    *ptr_cur_cloud = CuboidFilter ().pass_through_soild<PointXYZI> (ptr_cur_cloud, -20, 60, -20, 20, -5, 0);
    *ptr_cur_cloud = CuboidFilter ().hollow_removal<PointXYZI> (ptr_cur_cloud, -4, 1.18, -1.3, 1.3, -5, 0);
    *ptr_cur_cloud = CuboidFilter ().hollow_removal<PointXYZI> (ptr_cur_cloud, 1.3, 2.5, 1, 2, -1.5, 0);

    //cout << "[pass through   ]:" << ptr_cur_cloud->size () << "," << timer_algorithm_running.getTimeSeconds () << "s" << endl;

    pcl::PointIndicesPtr indices_ground (new pcl::PointIndices);
    *indices_ground = PlaneGroundFilter ().runMorphological<PointXYZI> (ptr_cur_cloud, 0.3, 2, 1, 0.2, 0.4, 0.5);
    //*indices_ground = RayGroundFilter (3.6,5.8,9.0,0.01,0.01,0.15,0.3,0.8,0.175).compute (ptr_cur_cloud);
    PointCloud<PointXYZI>::Ptr cloud_ground (new PointCloud<PointXYZI>);
    PointCloud<PointXYZI>::Ptr cloud_non_ground (new PointCloud<PointXYZI>);
    extract_Indices<PointXYZI> (ptr_cur_cloud, indices_ground, *cloud_ground, *cloud_non_ground);
    //cout << "[remove ground]:" << timer_algorithm_running.getTimeSeconds () << "s"<< endl;

    if (cloud_ground->size () < 100)
    {
      RosModuleHINO::send_ErrorCode (0x4000);
      cout << "error: not find ground" << endl;
    }

    //RosModuleHINO::send_Rviz (*cloud_non_ground);
    RosModuleHINO::send_LidarAllNonGround (*cloud_non_ground, msg->header.stamp, msg->header.frame_id);

    //delete[] cur_cluster;

    if (stopWatch.getTimeSeconds () > 0.05)
    {
      cout << "[Preprocess slow]:" << stopWatch.getTimeSeconds () << "s" << endl << endl;
    }
  }
}

void
detection (int argc,
           char ** argv)
{
  cout.setf (std::ios_base::fixed, std::ios_base::floatfield);
  cout.precision (3);

  RosModuleHINO::initial ("lidars_preprocessing", argc, argv);
  RosModuleHINO::RegisterCallBackLidarAll (callback_LidarAll);

  ros::Rate loop_rate (10);
  while (ros::ok ())
  {
    ros::spinOnce ();
    loop_rate.sleep ();
  }
  RosModuleHINO::send_ErrorCode (0x0006);
  cout << "error: computational loop thread closed" << endl;
}

int
main (int argc,
      char ** argv)
{
  cout << "===================== lidars_preprocessing startup =====================" << endl;

  detection (argc, argv);

  cout << "===================== lidars_preprocessing end   =====================" << endl;
  return (0);
}
