#include "convex_fusion_b1.h"
#include "VoxelGrid_CUDA.h"
#include "VoxelFilter_CUDA.h"
#include "UseApproxMVBB.h"

using namespace std;
using namespace pcl;

#define CHECKTIMES 20

mutex g_syncLock;
pcl::PointCloud<pcl::PointXYZI>::Ptr g_ptr_cur_cloud(new pcl::PointCloud<pcl::PointXYZI>);
vector<msgs::DetectedObject> g_object_front_60;
vector<msgs::DetectedObject> g_object_top_front_120;
vector<msgs::DetectedObject> g_object_top_rear_120;
size_t g_heartBeat[4];
ros::Time g_frame_time;
std::string g_frame_id = "lidar";
int g_module_id = 2;

void callback_lidarall_nonground(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  g_syncLock.lock();
  *g_ptr_cur_cloud = *msg;
  g_heartBeat[0] = 0;
  g_syncLock.unlock();
  // cout<<"Lidar"<<endl;
}

void callback_camera_front_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  g_syncLock.lock();
  g_object_front_60 = msg->objects;
  g_frame_time = msg->header.stamp;
  g_heartBeat[1] = 0;
  g_syncLock.unlock();
  // cout<< camera::topics_obj[camera::id::front_60] <<endl;
}

void callback_camera_top_front_120(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  g_syncLock.lock();
  g_object_top_front_120 = msg->objects;
  g_heartBeat[2] = 0;
  g_syncLock.unlock();
  // cout<< camera::topics_obj[camera::id::top_front_120] <<endl;
}

void callback_camera_top_rear_120(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  g_syncLock.lock();
  g_object_top_rear_120 = msg->objects;
  g_heartBeat[3] = 0;
  g_syncLock.unlock();
  // cout<< camera::topics_obj[camera::id::top_rear_120] <<endl;
}

int main(int argc, char** argv)
{
  cout << "===================== convex_fusion startup =====================" << endl;

  cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  cout.precision(3);
  ConvexFusionB1 convexFusionB1;

  convexFusionB1.initial("convex_fusion", argc, argv);
  convexFusionB1.registerCallBackLidarAllNonGround(callback_lidarall_nonground);
  convexFusionB1.registerCallBackCameraDetection(callback_camera_front_60, callback_camera_top_front_120,
                                                 callback_camera_top_rear_120);

  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    StopWatch stopWatch;

    g_syncLock.lock();

    //------------------------------------------------------------------------- LiDAR

    PointCloud<PointXYZI> lidarall_nonground;
    lidarall_nonground = *g_ptr_cur_cloud;

    if (g_heartBeat[0] > CHECKTIMES)
    {
      lidarall_nonground.clear();
      g_heartBeat[0] = 0;
      cout << "[Convex Fusion]: no LidarAllNonGroud" << endl;
    }
    else
    {
      g_heartBeat[0]++;
    }

    //------------------------------------------------------------------------- Camera

    size_t numberABB = g_object_front_60.size() + g_object_top_front_120.size() + g_object_top_rear_120.size();

    CLUSTER_INFO* camera_ABB = new CLUSTER_INFO[numberABB];
    CLUSTER_INFO* camera_ABB_bbox = new CLUSTER_INFO[numberABB];

    size_t cnt = 0;
    for (size_t i = 0; i < g_object_front_60.size(); i++)
    {
      camera_ABB[i + cnt].min.x = g_object_front_60[i].bPoint.p0.x;
      camera_ABB[i + cnt].min.y = g_object_front_60[i].bPoint.p0.y;
      camera_ABB[i + cnt].min.z = g_object_front_60[i].bPoint.p0.z;
      camera_ABB[i + cnt].max.x = g_object_front_60[i].bPoint.p6.x;
      camera_ABB[i + cnt].max.y = g_object_front_60[i].bPoint.p6.y;
      camera_ABB[i + cnt].max.z = g_object_front_60[i].bPoint.p6.z;
      if (g_object_front_60[i].distance < 0)
      {
        camera_ABB[i + cnt].cluster_tag = static_cast<int>(DriveNet::common_type_id::other);
      }
      else
      {
        camera_ABB[i + cnt].cluster_tag = g_object_front_60[i].classId;
      }
      camera_ABB_bbox[i + cnt] = camera_ABB[i + cnt];
    }
    cnt += g_object_front_60.size();

    for (size_t i = 0; i < g_object_top_front_120.size(); i++)
    {
      camera_ABB[i + cnt].min.x = g_object_top_front_120[i].bPoint.p0.x;
      camera_ABB[i + cnt].min.y = g_object_top_front_120[i].bPoint.p0.y;
      camera_ABB[i + cnt].min.z = g_object_top_front_120[i].bPoint.p0.z;
      camera_ABB[i + cnt].max.x = g_object_top_front_120[i].bPoint.p6.x;
      camera_ABB[i + cnt].max.y = g_object_top_front_120[i].bPoint.p6.y;
      camera_ABB[i + cnt].max.z = g_object_top_front_120[i].bPoint.p6.z;
      if (g_object_top_front_120[i].distance < 0)
      {
        camera_ABB[i + cnt].cluster_tag = static_cast<int>(DriveNet::common_type_id::other);
      }
      else
      {
        camera_ABB[i + cnt].cluster_tag = g_object_top_front_120[i].classId;
      }
      camera_ABB_bbox[i + cnt] = camera_ABB[i + cnt];
    }
    cnt += g_object_top_front_120.size();

    for (size_t i = 0; i < g_object_top_rear_120.size(); i++)
    {
      camera_ABB[i + cnt].min.x = g_object_top_rear_120[i].bPoint.p0.x;
      camera_ABB[i + cnt].min.y = g_object_top_rear_120[i].bPoint.p0.y;
      camera_ABB[i + cnt].min.z = g_object_top_rear_120[i].bPoint.p0.z;
      camera_ABB[i + cnt].max.x = g_object_top_rear_120[i].bPoint.p6.x;
      camera_ABB[i + cnt].max.y = g_object_top_rear_120[i].bPoint.p6.y;
      camera_ABB[i + cnt].max.z = g_object_top_rear_120[i].bPoint.p6.z;
      if (g_object_top_rear_120[i].distance < 0)
      {
        camera_ABB[i + cnt].cluster_tag = static_cast<int>(DriveNet::common_type_id::other);
      }
      else
      {
        camera_ABB[i + cnt].cluster_tag = g_object_top_rear_120[i].classId;
      }
      camera_ABB_bbox[i + cnt] = camera_ABB[i + cnt];
    }

    for (size_t i = 0; i < numberABB; i++)
    {
      float scale = 0;

      switch (camera_ABB[i].cluster_tag)
      {
        case static_cast<int>(DriveNet::common_type_id::other):  // Unknown
          scale = 0;
          break;

        case static_cast<int>(DriveNet::common_type_id::person):  // Person
          if (camera_ABB[i].min.x < 10)
          {
            scale = 1;
          }
          else if (camera_ABB[i].min.x >= 10 && camera_ABB[i].min.x <= 30)
          {
            scale = 1.5;
          }
          else if (camera_ABB[i].min.x > 30)
          {
            scale = 2;
          }
          break;

        case static_cast<int>(DriveNet::common_type_id::bicycle):  // Bicycle
        case static_cast<int>(DriveNet::common_type_id::motorbike):  // Motobike
          if (camera_ABB[i].min.x < 15)
          {
            scale = 0.8;
          }
          else if (camera_ABB[i].min.x >= 15 && camera_ABB[i].min.x <= 30)
          {
            scale = 1.2;
          }
          else if (camera_ABB[i].min.x > 30)
          {
            scale = 1.6;
          }
          break;

        case static_cast<int>(DriveNet::common_type_id::car):  // Car
        case static_cast<int>(DriveNet::common_type_id::bus):  // Bus
        case static_cast<int>(DriveNet::common_type_id::truck):  // Truck
          if (camera_ABB[i].min.x < 15)
          {
            scale = 0.2;
          }
          else if (camera_ABB[i].min.x >= 15 && camera_ABB[i].min.x <= 30)
          {
            scale = 0.8;
          }
          else if (camera_ABB[i].min.x > 30)
          {
            scale = 1.5;
          }
          break;
      }

      camera_ABB[i].min.x += -scale;
      camera_ABB[i].min.y += +scale;
      camera_ABB[i].max.x += +scale;
      camera_ABB[i].max.y += -scale;
    }

    if (g_heartBeat[1] > CHECKTIMES)
    {
      g_object_front_60.clear();
      g_heartBeat[1] = 0;
      cout << "[Convex Fusion]: no " << camera::topics_obj[camera::id::front_60] << endl;
    }
    else
    {
      g_heartBeat[1]++;
    }

    if (g_heartBeat[2] > CHECKTIMES)
    {
      g_object_top_front_120.clear();
      g_heartBeat[2] = 0;
      cout << "[Convex Fusion]: no " << camera::topics_obj[camera::id::top_front_120] << endl;
    }
    else
    {
      g_heartBeat[2]++;
    }

    if (g_heartBeat[3] > CHECKTIMES)
    {
      g_object_top_rear_120.clear();
      g_heartBeat[3] = 0;
      cout << "[Convex Fusion]: no " << camera::topics_obj[camera::id::top_rear_120] << endl;
    }
    else
    {
      g_heartBeat[3]++;
    }

    //-------------------------------------------------------------------------

    g_syncLock.unlock();

    if (numberABB > 0)
    {
      for (size_t i = 0; i < lidarall_nonground.size(); i++)
      {
        for (size_t j = 0; j < numberABB; j++)
        {
          if (lidarall_nonground.points[i].x > camera_ABB[j].min.x and
              lidarall_nonground.points[i].x < camera_ABB[j].max.x and
              lidarall_nonground.points[i].y < camera_ABB[j].min.y and
              lidarall_nonground.points[i].y > camera_ABB[j].max.y)
          {
            camera_ABB[j].cloud.push_back(PointXYZ(lidarall_nonground.points[i].x, lidarall_nonground.points[i].y,
                                                   lidarall_nonground.points[i].z));
          }
        }
      }

#pragma omp parallel for
      for (size_t i = 0; i < numberABB; i++)
      {
        UseApproxMVBB approxMVBB;
        approxMVBB.setInputCloud(camera_ABB[i].cloud);
        approxMVBB.Compute(camera_ABB[i].obb_vertex, camera_ABB[i].center, camera_ABB[i].min, camera_ABB[i].max,
                           camera_ABB[i].convex_hull);
      }
      convexFusionB1.sendCameraResults(camera_ABB, camera_ABB_bbox, numberABB, g_frame_time, g_frame_id);
      
      delete[]camera_ABB;
      delete[]camera_ABB_bbox;
    }

    if (stopWatch.getTimeSeconds() > 0.05)
    {
      cout << "[Convex Fusion]: too slow " << stopWatch.getTimeSeconds() << "s" << endl << endl;
    }

    ros::spinOnce();
    loop_rate.sleep();
  }
  convexFusionB1.sendErrorCode(0x0006, g_frame_id, g_module_id);

  cout << "===================== convex_fusion end   =====================" << endl;
  return (0);
}
