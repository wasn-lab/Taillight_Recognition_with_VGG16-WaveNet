#include "all_header.h"
#include "RosModuleB1.h"
#include "CuboidFilter.h"
#include "VoxelGrid_CUDA.h"
#include "VoxelFilter_CUDA.h"
#include "PlaneGroundFilter.h"
#include "RayGroundFilter.h"
#include "extract_Indices.h"
#include "NoiseFilter.h"

#if ENABLE_DEBUG_MODE == true
#include "UI/QtViewer.h"
#endif

bool debug_output = false;
StopWatch stopWatch;

void callback_LidarAll(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
#if 0
  static pcl::uint64_t oldtimestamp;
  pcl::uint64_t intervaltime = msg->header.stamp - oldtimestamp;
  if(oldtimestamp!=0 && intervaltime > 100000L && intervaltime < 10000000L) // 100000 = 0.1s
  {
    cout << "[lidars_preprocessing]: missing data " << intervaltime << "," << msg->header.stamp <<endl;
  }
  oldtimestamp = msg->header.stamp;
#endif

  if (msg->size() > 100)
  {
    if (debug_output)
    {
      ros::Time rosTime;
      pcl_conversions::fromPCL(msg->header.stamp, rosTime);
      if ((ros::Time::now() - rosTime).toSec() < 3600)
      {
        cout << "[All->Pre]: " << (ros::Time::now() - rosTime).toSec() * 1000 << "ms" << endl;
      }
      stopWatch.reset();
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_cur_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    *ptr_cur_cloud = *msg;
    // cout << "[raw data       ]:" << ptr_cur_cloud->size () << endl;

    *ptr_cur_cloud = CuboidFilter().pass_through_soild<PointXYZI>(ptr_cur_cloud, -50, 50, -25, 25, -5, 1);
    *ptr_cur_cloud = CuboidFilter().hollow_removal<PointXYZI>(ptr_cur_cloud, -6.6, 0.9, -1.45, 1.45, -5, 1);
    // cout << "[pass through   ]:" << ptr_cur_cloud->size () << "," << timer_algorithm_running.getTimeSeconds () << "s"
    // << endl;
    

    PointCloud<PointXYZI>::Ptr cloud_ground(new PointCloud<PointXYZI>);
    PointCloud<PointXYZI>::Ptr cloud_non_ground(new PointCloud<PointXYZI>);
    pcl::PointIndicesPtr indices_ground(new pcl::PointIndices);

    // pcl::PointCloud<pcl::PointXYZ>::Ptr buff (new pcl::PointCloud<pcl::PointXYZ>);
    // copyPointCloud (*ptr_cur_cloud, *buff);
    //*indices_ground = RayGroundFilter (2.57, 2.8, 9.0, 0.01, 0.01, 0.15, 0.3, 0.8, 0.175).compute<PointXYZ> (buff);
    // extract_Indices<PointXYZI> (ptr_cur_cloud, indices_ground, *cloud_ground, *cloud_non_ground);

    *indices_ground = PlaneGroundFilter().runMorphological<PointXYZI>(ptr_cur_cloud, 0.3, 2, 1, 0.9, 0.32, 0.33);
    extract_Indices<PointXYZI>(ptr_cur_cloud, indices_ground, *cloud_ground, *cloud_non_ground);

    // cout << "[remove ground]:" << timer_algorithm_running.gcloud_non_ground
    if (cloud_ground->size() < 100)
    {
      RosModuleB1::send_ErrorCode(0x4000);
      cout << "error: not find ground" << endl;
    }

    RosModuleB1::send_LidarAllNonGround(*cloud_non_ground, msg->header.stamp, msg->header.frame_id);

    if (debug_output)
    {
      cout << "[Preprocess]: " << stopWatch.getTimeSeconds() << 's' << endl;
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
  RosModuleB1::initial("lidars_preprocessing", argc, argv);
  RosModuleB1::RegisterCallBackLidarAll(callback_LidarAll);
  cout << "[" << ros::this_node::getName() << "] "
       << "----------------------------startup" << endl;

  // check debug mode
  ros::param::get("/debug_output", debug_output);

  cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  cout.precision(3);

#if ENABLE_DEBUG_MODE == true
  thread TheadDetection(UI, argc, argv);
#endif

  ros::AsyncSpinner spinner(1);
  spinner.start();

  while (ros::ok())
  {
    ros::Rate(1).sleep();
  }
  RosModuleB1::send_ErrorCode(0x0006);

  cout << "[" << ros::this_node::getName() << "] "
       << "----------------------------end" << endl;
  return (0);
}
