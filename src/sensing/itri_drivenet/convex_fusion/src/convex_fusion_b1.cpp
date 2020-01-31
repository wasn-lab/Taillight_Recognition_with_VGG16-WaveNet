#include "convex_fusion_b1.h"
#include "VoxelGrid_CUDA.h"
#include "VoxelFilter_CUDA.h"
#include "UseApproxMVBB.h"

using namespace std;
using namespace pcl;

#define CHECKTIMES 20

mutex syncLock;
pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_cur_cloud(new pcl::PointCloud<pcl::PointXYZI>);
vector<msgs::DetectedObject> ObjectFC;
vector<msgs::DetectedObject> ObjectFT;
vector<msgs::DetectedObject> ObjectBT;
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

void callback_CameraFC(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  syncLock.lock();
  ObjectFC = msg->objects;
  frame_time = msg->header.stamp;
  heartBeat[1] = 0;
  syncLock.unlock();
  // cout<<"30"<<endl;
}

void callback_CameraFT(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  syncLock.lock();
  ObjectFT = msg->objects;
  heartBeat[2] = 0;
  syncLock.unlock();
  // cout<<"60"<<endl;
}

void callback_CameraBT(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  syncLock.lock();
  ObjectBT = msg->objects;
  heartBeat[3] = 0;
  syncLock.unlock();
  // cout<<"120"<<endl;
}

int main(int argc, char** argv)
{
  cout << "===================== convex_fusion startup =====================" << endl;

  cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  cout.precision(3);

  ConvexFusionB1::initial("convex_fusion", argc, argv);
  ConvexFusionB1::RegisterCallBackLidarAllNonGround(callback_LidarAllNonGround);
  ConvexFusionB1::RegisterCallBackCameraDetection(callback_CameraFC, callback_CameraFT, callback_CameraBT);

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

    size_t NumberABB = ObjectFC.size() + ObjectFT.size() + ObjectBT.size();

    CLUSTER_INFO CameraABB[NumberABB];
    CLUSTER_INFO CameraABB_BBox[NumberABB];

    size_t CNT = 0;
    for (size_t i = 0; i < ObjectFC.size(); i++)
    {
      CameraABB[i + CNT].min.x = ObjectFC[i].bPoint.p0.x;
      CameraABB[i + CNT].min.y = ObjectFC[i].bPoint.p0.y;
      CameraABB[i + CNT].min.z = ObjectFC[i].bPoint.p0.z;
      CameraABB[i + CNT].max.x = ObjectFC[i].bPoint.p6.x;
      CameraABB[i + CNT].max.y = ObjectFC[i].bPoint.p6.y;
      CameraABB[i + CNT].max.z = ObjectFC[i].bPoint.p6.z;
      if (ObjectFC[i].distance < 0)
        CameraABB[i + CNT].cluster_tag = 0;
      else
        CameraABB[i + CNT].cluster_tag = ObjectFC[i].classId;
      CameraABB_BBox[i + CNT] = CameraABB[i + CNT];
    }
    CNT += ObjectFC.size();

    for (size_t i = 0; i < ObjectFT.size(); i++)
    {
      CameraABB[i + CNT].min.x = ObjectFT[i].bPoint.p0.x;
      CameraABB[i + CNT].min.y = ObjectFT[i].bPoint.p0.y;
      CameraABB[i + CNT].min.z = ObjectFT[i].bPoint.p0.z;
      CameraABB[i + CNT].max.x = ObjectFT[i].bPoint.p6.x;
      CameraABB[i + CNT].max.y = ObjectFT[i].bPoint.p6.y;
      CameraABB[i + CNT].max.z = ObjectFT[i].bPoint.p6.z;
      if (ObjectFT[i].distance < 0)
        CameraABB[i + CNT].cluster_tag = 0;
      else
        CameraABB[i + CNT].cluster_tag = ObjectFT[i].classId;
      CameraABB_BBox[i + CNT] = CameraABB[i + CNT];
    }
    CNT += ObjectFT.size();

    for (size_t i = 0; i < ObjectBT.size(); i++)
    {
      CameraABB[i + CNT].min.x = ObjectBT[i].bPoint.p0.x;
      CameraABB[i + CNT].min.y = ObjectBT[i].bPoint.p0.y;
      CameraABB[i + CNT].min.z = ObjectBT[i].bPoint.p0.z;
      CameraABB[i + CNT].max.x = ObjectBT[i].bPoint.p6.x;
      CameraABB[i + CNT].max.y = ObjectBT[i].bPoint.p6.y;
      CameraABB[i + CNT].max.z = ObjectBT[i].bPoint.p6.z;
      if (ObjectBT[i].distance < 0)
        CameraABB[i + CNT].cluster_tag = 0;
      else
        CameraABB[i + CNT].cluster_tag = ObjectBT[i].classId;
      CameraABB_BBox[i + CNT] = CameraABB[i + CNT];
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
      ObjectFC.clear();
      heartBeat[1] = 0;
      cout << "[Convex Fusion]: no FC" << endl;
    }
    else
    {
      heartBeat[1]++;
    }

    if (heartBeat[2] > CHECKTIMES)
    {
      ObjectFT.clear();
      heartBeat[2] = 0;
      cout << "[Convex Fusion]: no FT" << endl;
    }
    else
    {
      heartBeat[2]++;
    }

    if (heartBeat[3] > CHECKTIMES)
    {
      ObjectBT.clear();
      heartBeat[3] = 0;
      cout << "[Convex Fusion]: no BT" << endl;
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

#pragma omp parallel for
      for (size_t i = 0; i < NumberABB; i++)
      {
        UseApproxMVBB bbox2;
        bbox2.setInputCloud(CameraABB[i].cloud);
        bbox2.Compute(CameraABB[i].obb_vertex, CameraABB[i].center, CameraABB[i].min, CameraABB[i].max,
                      CameraABB[i].convex_hull);
      }
      ConvexFusionB1::Send_CameraResults(CameraABB, CameraABB_BBox, NumberABB, frame_time, "lidar");
    }

    if (stopWatch.getTimeSeconds() > 0.05)
    {
      cout << "[Convex Fusion]: too slow " << stopWatch.getTimeSeconds() << "s" << endl << endl;
    }

    ros::spinOnce();
    loop_rate.sleep();
  }
  ConvexFusionB1::send_ErrorCode(0x0006);

  cout << "===================== convex_fusion end   =====================" << endl;
  return (0);
}
