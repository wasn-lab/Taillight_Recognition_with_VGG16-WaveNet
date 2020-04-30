#include "convex_fusion_hino.h"
#include "VoxelGrid_CUDA.h"
#include "VoxelFilter_CUDA.h"
#include "UseApproxMVBB.h"

using namespace std;
using namespace pcl;

#define CHECKTIMES 20

mutex syncLock;
pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_cur_cloud(new pcl::PointCloud<pcl::PointXYZI>);
vector<msgs::DetectedObject> Object30;
vector<msgs::DetectedObject> Object60;
vector<msgs::DetectedObject> Object120;
size_t heartBeat[4];
ros::Time frame_time;

void callback_LidarAllNonGround(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  syncLock.lock();
  *ptr_cur_cloud = *msg;
  heartBeat[0] = 0;
  syncLock.unlock();
  // cout<<"Lidar"<<endl;
}

void callback_Camera30(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  syncLock.lock();
  Object30 = msg->objects;
  heartBeat[1] = 0;
  syncLock.unlock();
  // cout<<"30"<<endl;
}

void callback_Camera60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  syncLock.lock();
  Object60 = msg->objects;
  heartBeat[2] = 0;
  syncLock.unlock();
  // cout<<"60"<<endl;
}

void callback_Camera120(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  syncLock.lock();
  Object120 = msg->objects;
  heartBeat[3] = 0;
  syncLock.unlock();
  // cout<<"120"<<endl;
}

int main(int argc, char** argv)
{
  cout << "===================== convex_fusion startup =====================" << endl;

  cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  cout.precision(3);
  ConvexFusionHino convexFusionHino;

  convexFusionHino.initial("convex_fusion", argc, argv);
  convexFusionHino.RegisterCallBackLidarAllNonGround(callback_LidarAllNonGround);
  convexFusionHino.RegisterCallBackCameraDetection(callback_Camera30, callback_Camera60, callback_Camera120);

  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    StopWatch stopWatch;

    syncLock.lock();

    //------------------------------------------------------------------------- LiDAR

    PointCloud<PointXYZI> LidarAllNonGround;
    LidarAllNonGround = *ptr_cur_cloud;

    if (heartBeat[0] > CHECKTIMES)
    {
      LidarAllNonGround.clear();
      heartBeat[0] = 0;
      cout << "[Convex Fusion]: no LidarAllNonGroud" << endl;
    }
    else
    {
      heartBeat[0]++;
    }

    //------------------------------------------------------------------------- Camera

    size_t NumberABB = Object30.size() + Object60.size() + Object120.size();

    CLUSTER_INFO CameraABB[NumberABB];

    size_t CNT = 0;
    for (size_t i = 0; i < Object30.size(); i++)
    {
      CameraABB[i + CNT].min.x = Object30[i].bPoint.p0.x;
      CameraABB[i + CNT].min.y = Object30[i].bPoint.p0.y;
      CameraABB[i + CNT].min.z = Object30[i].bPoint.p0.z;
      CameraABB[i + CNT].max.x = Object30[i].bPoint.p6.x;
      CameraABB[i + CNT].max.y = Object30[i].bPoint.p6.y;
      CameraABB[i + CNT].max.z = Object30[i].bPoint.p6.z;
      CameraABB[i + CNT].cluster_tag = Object30[i].classId;
    }
    CNT += Object30.size();

    for (size_t i = 0; i < Object60.size(); i++)
    {
      CameraABB[i + CNT].min.x = Object60[i].bPoint.p0.x;
      CameraABB[i + CNT].min.y = Object60[i].bPoint.p0.y;
      CameraABB[i + CNT].min.z = Object60[i].bPoint.p0.z;
      CameraABB[i + CNT].max.x = Object60[i].bPoint.p6.x;
      CameraABB[i + CNT].max.y = Object60[i].bPoint.p6.y;
      CameraABB[i + CNT].max.z = Object60[i].bPoint.p6.z;
      CameraABB[i + CNT].cluster_tag = Object60[i].classId;
    }
    CNT += Object60.size();

    for (size_t i = 0; i < Object120.size(); i++)
    {
      CameraABB[i + CNT].min.x = Object120[i].bPoint.p0.x;
      CameraABB[i + CNT].min.y = Object120[i].bPoint.p0.y;
      CameraABB[i + CNT].min.z = Object120[i].bPoint.p0.z;
      CameraABB[i + CNT].max.x = Object120[i].bPoint.p6.x;
      CameraABB[i + CNT].max.y = Object120[i].bPoint.p6.y;
      CameraABB[i + CNT].max.z = Object120[i].bPoint.p6.z;
      CameraABB[i + CNT].cluster_tag = Object120[i].classId;
    }

    for (size_t i = 0; i < NumberABB; i++)
    {
      float SCALE = 0;

      switch (CameraABB[i].cluster_tag)
      {
        case 0:  // Unknown
          SCALE = 0;
          break;

        case 1:  // Person
          if (CameraABB[i].min.x < 10)
          {
            SCALE = 1;
          }
          else if (CameraABB[i].min.x >= 10 && CameraABB[i].min.x <= 30)
          {
            SCALE = 1.5;
          }
          else if (CameraABB[i].min.x > 30)
          {
            SCALE = 2;
          }
          break;

        case 2:  // Bicycle
        case 3:  // Motobike
          if (CameraABB[i].min.x < 15)
          {
            SCALE = 0.8;
          }
          else if (CameraABB[i].min.x >= 15 && CameraABB[i].min.x <= 30)
          {
            SCALE = 1.2;
          }
          else if (CameraABB[i].min.x > 30)
          {
            SCALE = 1.6;
          }
          break;

        case 4:  // Car
        case 5:  // Bus
        case 6:  // Truck
          if (CameraABB[i].min.x < 15)
          {
            SCALE = 0.2;
          }
          else if (CameraABB[i].min.x >= 15 && CameraABB[i].min.x <= 30)
          {
            SCALE = 0.8;
          }
          else if (CameraABB[i].min.x > 30)
          {
            SCALE = 1.5;
          }
          break;
      }

      CameraABB[i].min.x += -SCALE;
      CameraABB[i].min.y += +SCALE;
      CameraABB[i].max.x += +SCALE;
      CameraABB[i].max.y += -SCALE;
    }

    if (heartBeat[1] > CHECKTIMES)
    {
      Object30.clear();
      heartBeat[1] = 0;
      cout << "[Convex Fusion]: no 30" << endl;
    }
    else
    {
      heartBeat[1]++;
    }

    if (heartBeat[2] > CHECKTIMES)
    {
      Object60.clear();
      heartBeat[2] = 0;
      cout << "[Convex Fusion]: no 60" << endl;
    }
    else
    {
      heartBeat[2]++;
    }

    if (heartBeat[3] > CHECKTIMES)
    {
      Object120.clear();
      heartBeat[3] = 0;
      cout << "[Convex Fusion]: no 120" << endl;
    }
    else
    {
      heartBeat[3]++;
    }

    //-------------------------------------------------------------------------

    syncLock.unlock();

    if (NumberABB > 0)
    {
      for (size_t i = 0; i < LidarAllNonGround.size(); i++)
      {
        for (size_t j = 0; j < NumberABB; j++)
        {
          if (LidarAllNonGround.points[i].x > CameraABB[j].min.x and
              LidarAllNonGround.points[i].x < CameraABB[j].max.x and
              LidarAllNonGround.points[i].y < CameraABB[j].min.y and LidarAllNonGround.points[i].y > CameraABB[j].max.y)
          {
            CameraABB[j].cloud.push_back(
                PointXYZ(LidarAllNonGround.points[i].x, LidarAllNonGround.points[i].y, LidarAllNonGround.points[i].z));
          }
        }
      }

      for (size_t i = 0; i < NumberABB; i++)
      {
        UseApproxMVBB bbox2;
        bbox2.setInputCloud(CameraABB[i].cloud);
        bbox2.Compute(CameraABB[i].obb_vertex, CameraABB[i].center, CameraABB[i].min, CameraABB[i].max,
                      CameraABB[i].convex_hull);
      }

      convexFusionHino.Send_CameraResults(CameraABB, NumberABB, frame_time, "lidar");
    }

    if (stopWatch.getTimeSeconds() > 0.05)
    {
      cout << "[Convex Fusion]: too slow " << stopWatch.getTimeSeconds() << "s" << endl << endl;
    }

    ros::spinOnce();
    loop_rate.sleep();
  }
  convexFusionHino.send_ErrorCode(0x0006);

  cout << "===================== convex_fusion end   =====================" << endl;
  return (0);
}
