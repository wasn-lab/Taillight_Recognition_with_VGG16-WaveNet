#include "all_header.h"
#include "VoxelGrid_CUDA.h"
#include "VoxelFilter_CUDA.h"
#include "RosModuleB1.h"
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
bool debug_output = false;
std::atomic<uint32_t> latencyTime[2];

void callback_Clock(const rosgraph_msgs::Clock& msg)
{
  latencyTime[0] = msg.clock.sec;
  latencyTime[1] = msg.clock.nsec;
}

void callback_SSN(const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr& msg)
{
  heartBeat = 0;

  if (msg->size() > 0 && fabs(LinearAcc[0]) < 1.7)
  {
    if (debug_output)
    {
      ros::Time rosTime;
      pcl_conversions::fromPCL(msg->header.stamp, rosTime);
      if ((ros::Time::now() - rosTime).toSec() < 3600)
      {
        cout << "[All->DB]: " << (ros::Time::now() - rosTime).toSec() * 1000 << "ms" << endl;
      }
      stopWatch.reset();
    }

    int cur_cluster_num = 0;
    CLUSTER_INFO* cur_cluster = S1Cluster(viewer, &viewID).getClusters(ENABLE_DEBUG_MODE, msg, &cur_cluster_num);

    ros::Time rosTime;
    pcl_conversions::fromPCL(msg->header.stamp, rosTime);

    RosModuleB1::Send_LidarResults(cur_cluster, cur_cluster_num, rosTime, msg->header.frame_id);
    RosModuleB1::Send_LidarResults_v2(cur_cluster, cur_cluster_num, rosTime, msg->header.frame_id);
    RosModuleB1::Send_LidarResultsRVIZ(cur_cluster, cur_cluster_num);
    RosModuleB1::Send_LidarResultsRVIZ_abb(cur_cluster, cur_cluster_num);
    RosModuleB1::Send_LidarResultsRVIZ_obb(cur_cluster, cur_cluster_num);
    RosModuleB1::Send_LidarResultsRVIZ_heading(cur_cluster, cur_cluster_num);
    RosModuleB1::Send_LidarResultsGrid(cur_cluster, cur_cluster_num, rosTime, msg->header.frame_id);
    // auto start_time = chrono::high_resolution_clock::now();
    RosModuleB1::Send_LidarResultsEdge(cur_cluster, cur_cluster_num, rosTime, msg->header.frame_id);
    // auto end_time = chrono::high_resolution_clock::now();
    // cout << "[EDGE]: " << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << endl;
    delete[] cur_cluster;

    if (debug_output)
    {
      cout << "[DBScan]: " << stopWatch.getTimeSeconds() << 's' << endl;
    }

    double latency = (ros::Time::now() - rosTime).toSec();
    if (latency > 0 && latency < 3)
    {
      cout << "[Latency]: real-time " << latency << 's' << endl << endl;
    }
    else
    {
      latency = (ros::Time(latencyTime[0], latencyTime[1]) - rosTime).toSec();
      if (latency > 0 && latency < 3)
      {
        cout << "[Latency]: bag " << latency << 's' << endl << endl;
      }
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
  RosModuleB1::initial("output_results_by_dbscan", argc, argv);
  RosModuleB1::RegisterCallBackSSN(callback_SSN);
  RosModuleB1::RegisterCallBackIMU(callback_IMU);
  RosModuleB1::RegisterCallBackClock(callback_Clock);
  cout << "[" << ros::this_node::getName() << "] "
       << "----------------------------startup" << endl;

  ros::param::get("/debug_output", debug_output);

  cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  cout.precision(3);

#if ENABLE_DEBUG_MODE == true
  RosModuleB1::RegisterCallBackLidarAll(callback_LidarAll);

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

  ros::AsyncSpinner spinner(2);
  spinner.start();

  while (ros::ok())
  {
    heartBeat++;
    if (heartBeat > 30)
    {
      cout << "[DBSCAN]:no input " << heartBeat << endl;
    }

    ros::Rate(1).sleep();
  }

  cout << "[" << ros::this_node::getName() << "] "
       << "----------------------------end" << endl;
  return (0);
}
