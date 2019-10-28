#include "all_header.h"
#include "VoxelGrid_CUDA.h"
#include "VoxelFilter_CUDA.h"
#include "RosModuleHINO.hpp"
#include "S1Cluster.h"

#if ENABLE_DEBUG_MODE == true

#include "CompressFunction.h"

mutex mutex_LidarAll;

pcl::PointCloud<PointXYZI>::Ptr cloudPtr_LidarAll (new pcl::PointCloud<PointXYZI>);

void
callback_LidarAll (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  mutex_LidarAll.lock ();
  *cloudPtr_LidarAll = *msg;
  mutex_LidarAll.unlock ();
}

#endif

mutex mutex_SSN;
int heartBeat;
pcl::uint64_t pclTime;
string frameId;
std::atomic<float> LinearAcc[3];

pcl::PointCloud<PointXYZIL>::Ptr cloudPtr_SSN_N90deg (new pcl::PointCloud<PointXYZIL>);
pcl::PointCloud<PointXYZIL>::Ptr cloudPtr_SSN_P0deg (new pcl::PointCloud<PointXYZIL>);
pcl::PointCloud<PointXYZIL>::Ptr cloudPtr_SSN_P90deg (new pcl::PointCloud<PointXYZIL>);

void
callback_SSN_N90deg (const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr& msg)
{
  mutex_SSN.lock ();
  heartBeat = 0;
  pclTime = msg->header.stamp;
  frameId = msg->header.frame_id;
  *cloudPtr_SSN_N90deg = *msg;
  mutex_SSN.unlock ();
}

void
callback_SSN_P0deg (const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr& msg)
{
  mutex_SSN.lock ();
  heartBeat = 0;
  pclTime = msg->header.stamp;
  frameId = msg->header.frame_id;
  *cloudPtr_SSN_P0deg = *msg;
  mutex_SSN.unlock ();
}

void
callback_SSN_P90deg (const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr& msg)
{
  mutex_SSN.lock ();
  heartBeat = 0;
  pclTime = msg->header.stamp;
  frameId = msg->header.frame_id;
  *cloudPtr_SSN_P90deg = *msg;
  mutex_SSN.unlock ();
}

void
callback_IMU (const sensor_msgs::Imu::ConstPtr& msg)
{
  LinearAcc[0] = (msg->linear_acceleration.x);
  LinearAcc[1] = (msg->linear_acceleration.y);
  LinearAcc[2] = (msg->linear_acceleration.z);
}

class ObjectsDetectionHINO
{
  private:

    PointCloud<PointXYZIL>::Ptr release_cloud;
    StopWatch stopWatch;
    S1Cluster S1cluster;
    CLUSTER_INFO* cur_cluster;

    int cur_cluster_num = 0;

    thread mThread;
    mutex mutex_cluster;

  public:
    ObjectsDetectionHINO ()
    {
      cur_cluster = NULL;
      cur_cluster_num = 0;
    }

    void
    initial (boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer,
             int *input_viewID)
    {
      stopWatch.reset ();
      S1cluster = S1Cluster (input_viewer, input_viewID);

    }

    void
    setInputCloud (PointCloud<PointXYZIL>::Ptr input)
    {
      release_cloud = input;
    }

    void
    update ()
    {
      stopWatch.reset ();

      mutex_cluster.lock ();

      if (cur_cluster != NULL)
        delete[] cur_cluster;

      cur_cluster = S1cluster.getClusters (ENABLE_DEBUG_MODE, release_cloud, &cur_cluster_num);

      mutex_cluster.unlock ();

      ros::Time rosTime;
      pcl_conversions::fromPCL (pclTime, rosTime);


      RosModuleHINO::Send_LidarResults (cur_cluster, cur_cluster_num, rosTime, frameId);

      if (stopWatch.getTimeSeconds () > 0.1)
      {
        RosModuleHINO::Send_LidarResults (cur_cluster, cur_cluster_num, rosTime, frameId);
        cout << "[DBSCAN]:slow " << stopWatch.getTimeSeconds () << "s" << endl << endl;
      }
    }



};

void
detection (int argc,
           char ** argv)
{
  cout.setf (std::ios_base::fixed, std::ios_base::floatfield);
  cout.precision (3);

  RosModuleHINO::initial ("output_results_by_dbscan", argc, argv);

#if ENABLE_DEBUG_MODE == true
  RosModuleHINO::RegisterCallBackLidarAll (callback_LidarAll);
#endif

  RosModuleHINO::RegisterCallBackSSN (callback_SSN_N90deg, callback_SSN_P0deg, callback_SSN_P90deg);
  RosModuleHINO::RegisterCallBackIMU (callback_IMU);

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

#if ENABLE_DEBUG_MODE == true
  viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("PCL HDL Cloud"));
  pcl::visualization::Camera cam;
  cam = CompressFunction ().CamPara (7.122, -7.882, 71.702, 0.991, 0.003, 0.135, 15.755, -0.407, 8.052, 7.489, 143.559, 0.857, 0.857, 0.000, 0.000, 332.000,
      1028.000);
  viewer->initCameraParameters ();
  viewer->addCoordinateSystem (3.0);  //red:x green:y
  viewer->setBackgroundColor (0, 0, 0);
  viewer->setCameraParameters (cam, 0);
  viewer->setShowFPS (false);
#endif

  int viewID = 0;

  ObjectsDetectionHINO OD;
  OD.initial (viewer, &viewID);

  ros::Rate loop_rate (10);

  while (ros::ok ())
  {
    pcl::PointCloud<PointXYZIL>::Ptr release_SSNcloud (new pcl::PointCloud<PointXYZIL>);

    mutex_SSN.lock ();

    *release_SSNcloud += *cloudPtr_SSN_P0deg;
    *release_SSNcloud += *cloudPtr_SSN_N90deg;
    *release_SSNcloud += *cloudPtr_SSN_P90deg;

    heartBeat++;
    if (heartBeat > 30)
    {
      cout << "[DBSCAN]:no input" << endl;
    }

    mutex_SSN.unlock ();

#if ENABLE_DEBUG_MODE == true
    pcl::PointCloud<pcl::PointXYZI>::Ptr release_lidarAll (new pcl::PointCloud<pcl::PointXYZI>);

    mutex_LidarAll.lock ();
    if (cloudPtr_LidarAll->is_dense == false && cloudPtr_LidarAll->size () > 0)
    {
      *release_lidarAll = *cloudPtr_LidarAll;
      cloudPtr_LidarAll->is_dense = true;
    }
    mutex_LidarAll.unlock ();
#endif

    if (release_SSNcloud->size () > 0 && fabs(LinearAcc[0]) < 1.7)
    {
      OD.setInputCloud (release_SSNcloud);
      OD.update ();



#if ENABLE_DEBUG_MODE == true

      mutex_LidarAll.lock ();

      //pcl::visualization::PointCloudColorHandlerGenericField<PointXYZI> handler_OD ("intensity");
      pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> handler_OD (release_lidarAll, 255, 0, 0);

      pcl::copyPointCloud (*cloudPtr_LidarAll, *release_lidarAll);
      handler_OD.setInputCloud (release_lidarAll);
      if (!viewer->updatePointCloud (release_lidarAll, handler_OD, "OD"))
      {
        viewer->addPointCloud (release_lidarAll, handler_OD, "OD");
      }

      pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> handler_SSN (release_lidarAll, 255, 255, 255);
      pcl::copyPointCloud (*release_SSNcloud, *release_lidarAll);

      if (!viewer->updatePointCloud (release_lidarAll, handler_SSN, "OD2"))
      {
        viewer->addPointCloud (release_lidarAll, handler_SSN, "OD2");
      }

      viewer->spinOnce ();

      mutex_LidarAll.unlock ();
#endif

    }
    release_SSNcloud->clear ();

    ros::spinOnce ();
    loop_rate.sleep ();
  }
  cout << "error: computational loop thread closed" << endl;
}

int
main (int argc,
      char ** argv)
{
  cout << "===================== output_results_by_dbscan startup =====================" << endl;

  detection (argc, argv);

  cout << "===================== output_results_by_dbscan end   =====================" << endl;
  return (0);
}
