#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include "all_header.h"
#include "VoxelGrid_CUDA.h"
#include "VoxelFilter_CUDA.h"
#include "RosModuleB1.hpp"
#include "S1Cluster.h"

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
      virtual void
      onInit ()
      {
        RosModuleB1::RegisterCallBackSSN (callback_SSN);
        RosModuleB1::RegisterCallBackIMU (callback_IMU);
        RosModuleB1::RegisterCallBackClock (callback_Clock);
        RosModuleB1::RegisterCallBackTimer (callback_Timer, 1);

        cout << "[" << ros::this_node::getName () << "] " << "----------------------------startup" << endl;
        cout.setf (std::ios_base::fixed, std::ios_base::floatfield);
        cout.precision (3);
      }

    private:
      static void
      callback_Clock (const rosgraph_msgs::Clock &msg)
      {
        latencyTime[0] = msg.clock.sec;
        latencyTime[1] = msg.clock.nsec;
      }

      static void
      callback_SSN (const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr &msg)
      {
        heartBeat = 0;

        stopWatch.reset ();

        if (msg->size () > 0 && fabs (LinearAcc[0]) < 1.7)
        {
          stopWatch.reset ();

          int cur_cluster_num = 0;
          CLUSTER_INFO *cur_cluster = S1Cluster (viewer, &viewID).getClusters (ENABLE_DEBUG_MODE, msg, &cur_cluster_num);

          ros::Time rosTime;
          pcl_conversions::fromPCL (msg->header.stamp, rosTime);
          RosModuleB1::Send_LidarResults (cur_cluster, cur_cluster_num, rosTime, msg->header.frame_id);
          RosModuleB1::Send_LidarResultsRVIZ (cur_cluster, cur_cluster_num);
          RosModuleB1::Send_LidarResultsGrid (cur_cluster, cur_cluster_num, rosTime, msg->header.frame_id);
          delete[] cur_cluster;

          if (stopWatch.getTimeSeconds () > 0.05)
          {
            cout << "[DBSCAN]: " << stopWatch.getTimeSeconds () << "s" << endl << endl;
          }

          double latency = (ros::Time::now () - rosTime).toSec ();
          if (latency > 0 && latency < 3)
          {
            cout << "[Latency]: real-time " << latency << endl << endl;
          }
          else
          {
            latency = (ros::Time (latencyTime[0], latencyTime[1]) - rosTime).toSec ();
            if (latency > 0 && latency < 3)
            {
              cout << "[Latency]: bag " << latency << endl << endl;
            }
          }
        }
      }

      static void
      callback_IMU (const sensor_msgs::Imu::ConstPtr &msg)
      {
        LinearAcc[0] = (msg->linear_acceleration.x);
        LinearAcc[1] = (msg->linear_acceleration.y);
        LinearAcc[2] = (msg->linear_acceleration.z);
      }

      static void
      callback_Timer (const ros::TimerEvent &)
      {
        heartBeat++;
        if (heartBeat > 5)
        {
          cout << "[DBSCAN]:no input " << heartBeat << endl;
        }
      }
  };
}

PLUGINLIB_EXPORT_CLASS(output_dbscan_nodelet::LidarsNodelet, nodelet::Nodelet)
