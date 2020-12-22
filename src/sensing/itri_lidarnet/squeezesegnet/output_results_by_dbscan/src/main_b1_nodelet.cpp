#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include "all_header.h"
#include "VoxelGrid_CUDA.h"
#include "VoxelFilter_CUDA.h"
#include "RosModuleB1.h"
#include "S1Cluster.h"

StopWatch stopWatch;
bool debug_output = false;
namespace output_dbscan_nodelet
{
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
int viewID = 0;

int heartBeat;
std::atomic<float> LinearAcc[3];

StopWatch stopWatch;
std::atomic<uint32_t> latencyTime[2];

class LidarsNodelet : public nodelet::Nodelet
{
public:
  virtual void onInit()
  {
    RosModuleB1::RegisterCallBackSSN(callback_SSN);
    RosModuleB1::RegisterCallBackIMU(callback_IMU);
    RosModuleB1::RegisterCallBackClock(callback_Clock);
    RosModuleB1::RegisterCallBackTimer(callback_Timer, 1);

    cout << "[" << ros::this_node::getName() << "] "
         << "----------------------------startup" << endl;
    cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    cout.precision(3);
    ros::param::get("/debug_output", debug_output);
  }

  static void callback_Clock(const rosgraph_msgs::Clock& msg)
  {
    latencyTime[0] = msg.clock.sec;
    latencyTime[1] = msg.clock.nsec;
  }

  static void callback_SSN(const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr& msg)
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
      RosModuleB1::Send_LidarResultsRVIZ(cur_cluster, cur_cluster_num);
      RosModuleB1::Send_LidarResultsGrid(cur_cluster, cur_cluster_num, rosTime, msg->header.frame_id);
      RosModuleB1::Send_LidarResultsHeartBeat();
      RosModuleB1::Send_LidarResultsGridHeartBeat();
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
    }
  }

  static void callback_IMU(const sensor_msgs::Imu::ConstPtr& msg)
  {
    LinearAcc[0] = (msg->linear_acceleration.x);
    LinearAcc[1] = (msg->linear_acceleration.y);
    LinearAcc[2] = (msg->linear_acceleration.z);
  }

  static void callback_Timer(const ros::TimerEvent&)
  {
    heartBeat++;
    if (heartBeat > 5)
    {
      cout << "[DBSCAN]:no input " << heartBeat << endl;
    }
  }
};
}  // namespace output_dbscan_nodelet

PLUGINLIB_EXPORT_CLASS(output_dbscan_nodelet::LidarsNodelet, nodelet::Nodelet)
