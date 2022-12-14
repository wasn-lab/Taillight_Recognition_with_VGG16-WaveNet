#include "convex_fusion_b1_v2.h"
#include "VoxelGrid_CUDA.h"
#include "VoxelFilter_CUDA.h"
#include "UseApproxMVBB.h"
#include "module_id.h"

using namespace std;
using namespace pcl;

#define CHECKTIMES 20

mutex g_mutex_lidarall_nonground;
mutex g_mutex_front_bottom_60;
pcl::PointCloud<pcl::PointXYZI>::Ptr g_ptr_lidarall_nonground(new pcl::PointCloud<pcl::PointXYZI>);
vector<msgs::DetectedObject> g_object_front_bottom_60;
size_t g_heart_beat[2];
ros::Time g_frame_time;
const std::string g_frame_id = "lidar";

void callback_lidarall_nonground(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  g_mutex_lidarall_nonground.lock();
  *g_ptr_lidarall_nonground = *msg;
  g_heart_beat[0] = 0;
  g_mutex_lidarall_nonground.unlock();
  // cout<<"Lidar"<<endl;
}

void callback_camera_front_bottom_60(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  g_mutex_front_bottom_60.lock();
  g_object_front_bottom_60 = msg->objects;
  g_frame_time = msg->header.stamp;
  g_heart_beat[1] = 0;
  g_mutex_front_bottom_60.unlock();
  // cout<< camera::topics_obj[camera::id::front_bottom_60] <<endl;
}

int main(int argc, char** argv)
{
  cout << "===================== convex_fusion startup =====================" << endl;

  cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  cout.precision(3);
  ConvexFusionB1V2 convexFusionB1V2;

  convexFusionB1V2.initial("convex_fusion", argc, argv);
  convexFusionB1V2.registerCallBackLidarAllNonGround(callback_lidarall_nonground);
  convexFusionB1V2.registerCallBackCameraDetection(callback_camera_front_bottom_60);

  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    StopWatch stopWatch;
    //------------------------------------------------------------------------- LiDAR
    g_mutex_lidarall_nonground.lock();
    PointCloud<PointXYZI> lidarall_nonground;
    lidarall_nonground = *g_ptr_lidarall_nonground;

    if (g_heart_beat[0] > CHECKTIMES)
    {
      lidarall_nonground.clear();
      g_heart_beat[0] = 0;
      cout << "[Convex Fusion]: no LidarAllNonGroud" << endl;
    }
    else
    {
      g_heart_beat[0]++;
    }
    g_mutex_lidarall_nonground.unlock();
    //------------------------------------------------------------------------- Camera
    std::vector<msgs::DetectedObject> object_front_bottom_60;
    ros::Time frame_time;
    g_mutex_front_bottom_60.lock();
    object_front_bottom_60 = g_object_front_bottom_60;
    frame_time = g_frame_time;
    if (g_heart_beat[1] > CHECKTIMES)
    {
      g_object_front_bottom_60.clear();
      g_heart_beat[1] = 0;
      cout << "[Convex Fusion]: no " << camera::topics_obj[camera::id::front_bottom_60] << endl;
    }
    else
    {
      g_heart_beat[1]++;
    }
    g_mutex_front_bottom_60.unlock();

    //------------------------------------------------------------------------- Main
    size_t numberABB = object_front_bottom_60.size();
    if (numberABB > 0)
    {
      std::unique_ptr<CLUSTER_INFO[]> camera_ABB(new CLUSTER_INFO[numberABB]);
      std::unique_ptr<CLUSTER_INFO[]> camera_ABB_bbox(new CLUSTER_INFO[numberABB]);

      size_t cnt = 0;
      for (size_t i = 0; i < object_front_bottom_60.size(); i++)
      {
        camera_ABB[i + cnt].min.x = object_front_bottom_60[i].bPoint.p0.x;
        camera_ABB[i + cnt].min.y = object_front_bottom_60[i].bPoint.p0.y;
        camera_ABB[i + cnt].min.z = object_front_bottom_60[i].bPoint.p0.z;
        camera_ABB[i + cnt].max.x = object_front_bottom_60[i].bPoint.p6.x;
        camera_ABB[i + cnt].max.y = object_front_bottom_60[i].bPoint.p6.y;
        camera_ABB[i + cnt].max.z = object_front_bottom_60[i].bPoint.p6.z;
        if (object_front_bottom_60[i].distance < 0)
        {
          camera_ABB[i + cnt].cluster_tag = static_cast<int>(DriveNet::common_type_id::other);
        }
        else
        {
          camera_ABB[i + cnt].cluster_tag = g_object_front_bottom_60[i].classId;
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

          case static_cast<int>(DriveNet::common_type_id::bicycle):    // Bicycle
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

          case static_cast<int>(DriveNet::common_type_id::car):    // Car
          case static_cast<int>(DriveNet::common_type_id::bus):    // Bus
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

      for (size_t i = 0; i < numberABB; i++)
      {
        UseApproxMVBB approxMVBB;
        approxMVBB.setInputCloud(camera_ABB[i].cloud);
        approxMVBB.Compute(camera_ABB[i].obb_vertex, camera_ABB[i].center, camera_ABB[i].min, camera_ABB[i].max,
                           camera_ABB[i].convex_hull);
      }
      convexFusionB1V2.sendCameraResults(camera_ABB.get(), camera_ABB_bbox.get(), numberABB, frame_time, g_frame_id);
    }

    if (stopWatch.getTimeSeconds() > 0.05)
    {
      cout << "[Convex Fusion]: too slow " << stopWatch.getTimeSeconds() << "s" << endl << endl;
    }

    ros::spinOnce();
    loop_rate.sleep();
  }
  convexFusionB1V2.sendErrorCode(0x0006, g_frame_id, sensor_msgs_itri::ModuleId::DriveNet);

  cout << "===================== convex_fusion end   =====================" << endl;
  return (0);
}
