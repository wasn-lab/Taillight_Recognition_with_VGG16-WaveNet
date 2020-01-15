#ifndef CONVEX_FUSION_B1_H
#define CONVEX_FUSION_B1_H

// =============================================
//                      STD
// =============================================
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <mutex>

// =============================================
//                      CUDA
// =============================================
#include <cuda.h>
#include <cuda_runtime.h>

// =============================================
//                      PCL
// =============================================
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

// =============================================
//                      ROS
// =============================================

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/PointCloud2.h>

#include "msgs/DetectedObjectArray.h"
#include "msgs/ErrorCode.h"

#include "UserDefine.h"
#include "camera_params.h"

class ConvexFusionB1
{
  public:
    static ros::Publisher ErrorCode_pub;
    static ros::Publisher CameraDetection_pub;
    
    static void
    initial (std::string nodename,
             int argc,
             char ** argv)
    {
      ros::init (argc, argv, nodename);
      ros::NodeHandle n;

      ErrorCode_pub = n.advertise<msgs::ErrorCode> ("/ErrorCode", 1);
      CameraDetection_pub = n.advertise<msgs::DetectedObjectArray> (camera::detect_result_polygon, 1);
    }

    static void
    RegisterCallBackLidarAllNonGround (void
    (*cb1) (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&))
    {
      ros::NodeHandle n;

      static ros::Subscriber LidarAllNonGroundSub = n.subscribe ("/LidarAll/NonGround", 1, cb1);
    }
  
    static void
    RegisterCallBackCameraDetection (void
                                     (*cb1) (const msgs::DetectedObjectArray::ConstPtr&),
                                     void
                                     (*cb2) (const msgs::DetectedObjectArray::ConstPtr&),
                                     void
                                     (*cb3) (const msgs::DetectedObjectArray::ConstPtr&))
    {
      ros::NodeHandle n;

      static ros::Subscriber CameraFCDetectionSub = n.subscribe (camera::topics_obj[camera::id::front_60], 1, cb1);
      static ros::Subscriber CameraFTDetectionSub = n.subscribe (camera::topics_obj[camera::id::top_front_120], 1, cb2);
      static ros::Subscriber CameraBTDetectionSub = n.subscribe (camera::topics_obj[camera::id::top_rear_120], 1, cb3);

    }

    static void
    send_ErrorCode (unsigned int error_code)
    {
      static uint32_t seq;

      msgs::ErrorCode objMsg;
      objMsg.header.seq = seq++;
      objMsg.header.stamp = ros::Time::now ();
      objMsg.header.frame_id = "lidar";
      objMsg.module = 2;
      objMsg.event = error_code;

      ErrorCode_pub.publish (objMsg);
    }

    static void
    Send_CameraResults (CLUSTER_INFO* cluster_info,
                        int cluster_size,
                        ros::Time rostime,
                        std::string frameId)
    {
      msgs::DetectedObjectArray msgObjArr;

      for (int i = 0; i < cluster_size; i++)
      {
        msgs::DetectedObject msgObj;
        msgObj.classId = cluster_info[i].cluster_tag;

        if (cluster_info[i].convex_hull.size () > 0)
        {
          msgObj.cPoint.lowerAreaPoints.resize (cluster_info[i].convex_hull.size ());
          for (size_t j = 0; j < cluster_info[i].convex_hull.size (); j++)
          {
            msgObj.cPoint.lowerAreaPoints[j].x = cluster_info[i].convex_hull[j].x;
            msgObj.cPoint.lowerAreaPoints[j].y = cluster_info[i].convex_hull[j].y;
            msgObj.cPoint.lowerAreaPoints[j].z = cluster_info[i].convex_hull[j].z;
          }

          msgObj.cPoint.objectHigh = cluster_info[i].dz;

          msgObj.fusionSourceId = 0;

          msgObj.header.stamp = rostime;
          msgObjArr.objects.push_back (msgObj);
        }
      }
      msgObjArr.header.stamp = rostime;
      msgObjArr.header.frame_id = frameId;
      CameraDetection_pub.publish (msgObjArr);

    }
};

ros::Publisher ConvexFusionB1::ErrorCode_pub;
ros::Publisher ConvexFusionB1::CameraDetection_pub;

#endif // CONVEX_FUSION_B1_H
