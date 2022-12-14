#include "all_header.h"
#include "VoxelGrid_CUDA.h"
#include "VoxelFilter_CUDA.h"
#include "RosModuleHINO.hpp"
#include "S1Cluster.h"

#if ENABLE_DEBUG_MODE == true

#include "CompressFunction.h"

mutex mutex_LidarAll;

pcl::PointCloud<PointXYZI>::Ptr cloudPtr_LidarAll(new pcl::PointCloud<PointXYZI>);

void callback_LidarAll(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  mutex_LidarAll.lock();
  *cloudPtr_LidarAll = *msg;
  mutex_LidarAll.unlock();
}

#endif

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
int viewID = 0;

int heartBeat;
std::atomic<float> LinearAcc[3];

StopWatch stopWatch;

void callback_SSN(const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr& msg)
{
  heartBeat = 0;

  stopWatch.reset();

  if (msg->size() > 0 && fabs(LinearAcc[0]) < 1.7)
  {
    stopWatch.reset();

    int cur_cluster_num = 0;
    CLUSTER_INFO* cur_cluster = S1Cluster(viewer, &viewID).getClusters(ENABLE_DEBUG_MODE, msg, &cur_cluster_num);

    ros::Time rosTime;
    pcl_conversions::fromPCL(msg->header.stamp, rosTime);
    RosModuleHINO::Send_LidarResults(cur_cluster, cur_cluster_num, rosTime, msg->header.frame_id);
    RosModuleHINO::Send_LidarResultsRVIZ(cur_cluster, cur_cluster_num);

    delete[] cur_cluster;

    if (stopWatch.getTimeSeconds() > 0.1)
    {
      cout << "[DBSCAN]: " << stopWatch.getTimeSeconds() << "s"
           << " rosTime: " << rosTime << endl
           << endl;
    }

#if ENABLE_DEBUG_MODE == true

    pcl::PointCloud<pcl::PointXYZI>::Ptr release_lidarAll(new pcl::PointCloud<pcl::PointXYZI>);

    mutex_LidarAll.lock();
    pcl::copyPointCloud(*cloudPtr_LidarAll, *release_lidarAll);
    mutex_LidarAll.unlock();

    // pcl::visualization::PointCloudColorHandlerGenericField<PointXYZI> handler_OD ("intensity");
    pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> handler_OD(release_lidarAll, 255, 0, 0);
    handler_OD.setInputCloud(release_lidarAll);

    if (!viewer->updatePointCloud(release_lidarAll, handler_OD, "OD"))
    {
      viewer->addPointCloud(release_lidarAll, handler_OD, "OD");
    }

    pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> handler_SSN(release_lidarAll, 255, 255, 255);
    pcl::copyPointCloud(*msg, *release_lidarAll);

    if (!viewer->updatePointCloud(release_lidarAll, handler_SSN, "OD2"))
    {
      viewer->addPointCloud(release_lidarAll, handler_SSN, "OD2");
    }

    viewer->spinOnce();
#endif
  }
}

void callback_IMU(const sensor_msgs::Imu::ConstPtr& msg)
{
  LinearAcc[0] = (msg->linear_acceleration.x);
  LinearAcc[1] = (msg->linear_acceleration.y);
  LinearAcc[2] = (msg->linear_acceleration.z);
}

int main(int argc, char** argv)
{
  RosModuleHINO::initial("output_results_by_dbscan", argc, argv);
  RosModuleHINO::RegisterCallBackSSN(callback_SSN);
  RosModuleHINO::RegisterCallBackIMU(callback_IMU);
  cout << "[" << ros::this_node::getName() << "] "
       << "----------------------------startup" << endl;

  cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  cout.precision(3);

#if ENABLE_DEBUG_MODE == true
  RosModuleHINO::RegisterCallBackLidarAll(callback_LidarAll);

  viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("PCL HDL Cloud"));
  pcl::visualization::Camera cam;
  cam = CompressFunction().CamPara(7.122, -7.882, 71.702, 0.991, 0.003, 0.135, 15.755, -0.407, 8.052, 7.489, 143.559,
                                   0.857, 0.857, 0.000, 0.000, 332.000, 1028.000);
  viewer->initCameraParameters();
  viewer->addCoordinateSystem(3.0);  // red:x green:y
  viewer->setBackgroundColor(0, 0, 0);
  viewer->setCameraParameters(cam, 0);
  viewer->setShowFPS(false);
#endif

  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    heartBeat++;
    if (heartBeat > 30)
    {
      cout << "[DBSCAN]:no input" << endl;
    }
    ++viewID;
    ros::spinOnce();
    loop_rate.sleep();
  }

  cout << "[" << ros::this_node::getName() << "] "
       << "----------------------------end" << endl;
  return (0);
}
