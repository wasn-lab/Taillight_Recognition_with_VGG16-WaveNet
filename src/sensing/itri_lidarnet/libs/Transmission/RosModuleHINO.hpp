#ifndef ROSMODULEHINO_H
#define ROSMODULEHINO_H

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

#include "../UserDefine.h"

#include "msgs/DetectedObjectArray.h"
#include "msgs/ErrorCode.h"

class RosModuleHINO
{
  public:

    static ros::Publisher ErrorCode_pub;
    static ros::Publisher Rviz_pub;
    static ros::Publisher LidarAllNonGround_pub;
    static ros::Publisher LidarDetection_pub;
    static ros::Publisher CameraDetection_pub;

    static void
    initial (string nodename,
             int argc,
             char ** argv)
    {
      ros::init (argc, argv, nodename);
      ros::NodeHandle n;

      ErrorCode_pub = n.advertise<msgs::ErrorCode> ("/ErrorCode", 1);
      Rviz_pub = n.advertise<sensor_msgs::PointCloud2> ("/LidarAll/NonGround2", 1);
      LidarAllNonGround_pub = n.advertise<PointCloud<PointXYZI>> ("/LidarAll/NonGround", 1);
      LidarDetection_pub = n.advertise<msgs::DetectedObjectArray> ("/LidarDetection", 1);
      CameraDetection_pub = n.advertise<msgs::DetectedObjectArray> ("/CameraDetection", 1);

    }

    static void
    RegisterCallBackLidarAll (void
    (*cb1) (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&))
    {
      ros::NodeHandle n;

      static ros::Subscriber LidarAllSub = n.subscribe ("/LidarAll", 1, cb1);
    }

    static void
    RegisterCallBackLidarAllNonGround (void
    (*cb1) (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&))
    {
      ros::NodeHandle n;

      static ros::Subscriber LidarAllNonGroundSub = n.subscribe ("/LidarAll/NonGround", 1, cb1);
    }

    static void
    RegisterCallBackSimLidarAll (void
    (*cb1) (const pcl::PointCloud<pcl::PointXYZLO>::ConstPtr&))
    {
      ros::NodeHandle n;

      static ros::Subscriber LidarAllSub = n.subscribe ("/Sim_LidarAll", 1, cb1);
    }

    static void
    RegisterCallBackLidarRaw (void
                              (*cb1) (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&),
                              void
                              (*cb2) (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&),
                              void
                              (*cb3) (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&),
                              void
                              (*cb4) (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&))
    {
      ros::NodeHandle n;
      static ros::Subscriber LidarFrontSub = n.subscribe ("/LidarFront", 1, cb1);
      static ros::Subscriber LidarLeftSub = n.subscribe ("/LidarLeft", 1, cb2);
      static ros::Subscriber LidarRightSub = n.subscribe ("/LidarRight", 1, cb3);
      static ros::Subscriber LidarTopSub = n.subscribe ("/LidarTop", 1, cb4);
    }

    static void
    RegisterCallBackSSN (void
                         (*cb2) (const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr&),
                         void
                         (*cb3) (const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr&),
                         void
                         (*cb4) (const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr&))
    {
      ros::NodeHandle n;

      static ros::Subscriber result_cloud_N90deg = n.subscribe ("/squ_seg/result_cloud_N90deg", 1, cb2);
      static ros::Subscriber result_cloud_P0deg = n.subscribe ("/squ_seg/result_cloud_P0deg", 1, cb3);
      static ros::Subscriber result_cloud_P90deg = n.subscribe ("/squ_seg/result_cloud_P90deg", 1, cb4);
    }

    static void
    RegisterCallBackIMU (void
    (*cb) (const sensor_msgs::Imu::ConstPtr&))
    {
      ros::NodeHandle n;

      static ros::Subscriber ImuSub = n.subscribe ("/imu/data", 1, cb);
    }

    static void
    RegisterCallBackLidarDetection (void
    (*cb1) (const msgs::DetectedObjectArray::ConstPtr&))
    {
      ros::NodeHandle n;

      static ros::Subscriber LidarDetectionSub = n.subscribe ("/LidarDetection", 1, cb1);
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

      static ros::Subscriber Camera30DetectionSub = n.subscribe ("/DetectedObjectArray/cam30", 1, cb1);
      static ros::Subscriber Camera60DetectionSub = n.subscribe ("/DetectedObjectArray/cam60", 1, cb2);
      static ros::Subscriber Camera120DetectionSub = n.subscribe ("/DetectedObjectArray/cam120", 1, cb3);

    }

    static void
    send_ErrorCode (unsigned int error_code)
    {
      static uint32_t seq;

      msgs::ErrorCode objMsg;
      objMsg.header.seq = seq++;
      objMsg.header.stamp = ros::Time::now ();
      objMsg.header.frame_id = "lidar";
      objMsg.module = 1;
      objMsg.event = error_code;

      ErrorCode_pub.publish (objMsg);
    }

    template <typename PointT>
    static void
    send_Rviz (const PointCloud<PointT> input)
    {
      static uint32_t seq;

      sensor_msgs::PointCloud2 spR_msg;
      pcl::toROSMsg (input, spR_msg);
      spR_msg.header.seq = seq++;
      spR_msg.header.stamp = ros::Time::now ();
      spR_msg.header.frame_id = "lidar";
      Rviz_pub.publish (spR_msg);
    }

    template <typename PointT>
    static void
    send_LidarAllNonGround (PointCloud<PointT> input,
                            pcl::uint64_t pcltime,
                            string frameId)
    {
      input.header.frame_id = frameId;
      input.header.stamp = pcltime;
      LidarAllNonGround_pub.publish (input);
    }

    static void
    Send_LidarResults (CLUSTER_INFO* cluster_info,
                       int cluster_size,
                       ros::Time rostime,
                       string frameId)
    {

      msgs::DetectedObjectArray msgObjArr;
      msgObjArr.header.frame_id = "lidar";

      for (int i = 0; i < cluster_size; i++)
      {
        if (cluster_info[i].cluster_tag >= 1)
        {
          msgs::DetectedObject msgObj;

          switch (cluster_info[i].cluster_tag)
          {
            case 0:
              msgObj.classId = 0;  //unknow
              break;
            case 1:
              msgObj.classId = 1;  //person
              break;
            case 2:
              msgObj.classId = 2;  //bicycle
              break;
            case 3:
              msgObj.classId = 3;  //motobike
              break;
            case 4:
              msgObj.classId = 4;  //car
              break;
            case 5:
              msgObj.classId = 5;  //bus
              break;
            case 6:
              msgObj.classId = 6;  //truck
              break;
            case 7:
              msgObj.classId = 7;  //sign
              break;
            case 8:
              msgObj.classId = 8;  //light
              break;
            case 9:
              msgObj.classId = 9;  //park
              break;
          }

          msgObj.distance = cluster_info[i].dis_center_origin;

          // if (cluster_info[i].dis_abbc_obbc > 0.1)
          if (true)
          {

            msgObj.bPoint.p0.x = cluster_info[i].min.x;
            msgObj.bPoint.p0.y = cluster_info[i].min.y;
            msgObj.bPoint.p0.z = cluster_info[i].min.z;

            msgObj.bPoint.p1.x = cluster_info[i].min.x;
            msgObj.bPoint.p1.y = cluster_info[i].min.y;
            msgObj.bPoint.p1.z = cluster_info[i].max.z;

            msgObj.bPoint.p2.x = cluster_info[i].max.x;
            msgObj.bPoint.p2.y = cluster_info[i].min.y;
            msgObj.bPoint.p2.z = cluster_info[i].max.z;

            msgObj.bPoint.p3.x = cluster_info[i].max.x;
            msgObj.bPoint.p3.y = cluster_info[i].min.y;
            msgObj.bPoint.p3.z = cluster_info[i].min.z;

            msgObj.bPoint.p4.x = cluster_info[i].min.x;
            msgObj.bPoint.p4.y = cluster_info[i].max.y;
            msgObj.bPoint.p4.z = cluster_info[i].min.z;

            msgObj.bPoint.p5.x = cluster_info[i].min.x;
            msgObj.bPoint.p5.y = cluster_info[i].max.y;
            msgObj.bPoint.p5.z = cluster_info[i].max.z;

            msgObj.bPoint.p6.x = cluster_info[i].max.x;
            msgObj.bPoint.p6.y = cluster_info[i].max.y;
            msgObj.bPoint.p6.z = cluster_info[i].max.z;

            msgObj.bPoint.p7.x = cluster_info[i].max.x;
            msgObj.bPoint.p7.y = cluster_info[i].max.y;
            msgObj.bPoint.p7.z = cluster_info[i].min.z;

          }
          else
          {

            msgObj.bPoint.p0.x = cluster_info[i].obb_vertex.at (0).x;
            msgObj.bPoint.p0.y = cluster_info[i].obb_vertex.at (0).y;
            msgObj.bPoint.p0.z = cluster_info[i].obb_vertex.at (0).z;

            msgObj.bPoint.p1.x = cluster_info[i].obb_vertex.at (1).x;
            msgObj.bPoint.p1.y = cluster_info[i].obb_vertex.at (1).y;
            msgObj.bPoint.p1.z = cluster_info[i].obb_vertex.at (1).z;

            msgObj.bPoint.p2.x = cluster_info[i].obb_vertex.at (2).x;
            msgObj.bPoint.p2.y = cluster_info[i].obb_vertex.at (2).y;
            msgObj.bPoint.p2.z = cluster_info[i].obb_vertex.at (2).z;

            msgObj.bPoint.p3.x = cluster_info[i].obb_vertex.at (3).x;
            msgObj.bPoint.p3.y = cluster_info[i].obb_vertex.at (3).y;
            msgObj.bPoint.p3.z = cluster_info[i].obb_vertex.at (3).z;

            msgObj.bPoint.p4.x = cluster_info[i].obb_vertex.at (4).x;
            msgObj.bPoint.p4.y = cluster_info[i].obb_vertex.at (4).y;
            msgObj.bPoint.p4.z = cluster_info[i].obb_vertex.at (4).z;

            msgObj.bPoint.p5.x = cluster_info[i].obb_vertex.at (5).x;
            msgObj.bPoint.p5.y = cluster_info[i].obb_vertex.at (5).y;
            msgObj.bPoint.p5.z = cluster_info[i].obb_vertex.at (5).z;

            msgObj.bPoint.p6.x = cluster_info[i].obb_vertex.at (6).x;
            msgObj.bPoint.p6.y = cluster_info[i].obb_vertex.at (6).y;
            msgObj.bPoint.p6.z = cluster_info[i].obb_vertex.at (6).z;

            msgObj.bPoint.p7.x = cluster_info[i].obb_vertex.at (7).x;
            msgObj.bPoint.p7.y = cluster_info[i].obb_vertex.at (7).y;
            msgObj.bPoint.p7.z = cluster_info[i].obb_vertex.at (7).z;

          }

          msgObj.cPoint.lowerAreaPoints.resize (cluster_info[i].convex_hull.size ());

          for (size_t j = 0; j < cluster_info[i].convex_hull.size (); j++)
          {
            msgObj.cPoint.lowerAreaPoints[j].x = cluster_info[i].convex_hull[j].x;
            msgObj.cPoint.lowerAreaPoints[j].y = cluster_info[i].convex_hull[j].y;
            msgObj.cPoint.lowerAreaPoints[j].z = cluster_info[i].convex_hull[j].z;
          }

          msgObj.cPoint.objectHigh = cluster_info[i].dz;

#if 0
          msgs::DetectedObject msgMMSL;

          msgMMSL.bPoint.p3 = msgObj.bPoint.p0;
          msgMMSL.bPoint.p7 = msgObj.bPoint.p1;
          msgMMSL.bPoint.p4 = msgObj.bPoint.p2;
          msgMMSL.bPoint.p0 = msgObj.bPoint.p3;
          msgMMSL.bPoint.p2 = msgObj.bPoint.p4;
          msgMMSL.bPoint.p6 = msgObj.bPoint.p5;
          msgMMSL.bPoint.p5 = msgObj.bPoint.p6;
          msgMMSL.bPoint.p1 = msgObj.bPoint.p7;

          msgObj.bPoint.p0=msgMMSL.bPoint.p3;
          msgObj.bPoint.p1=msgMMSL.bPoint.p7;
          msgObj.bPoint.p2=msgMMSL.bPoint.p4;
          msgObj.bPoint.p3=msgMMSL.bPoint.p0;
          msgObj.bPoint.p4=msgMMSL.bPoint.p2;
          msgObj.bPoint.p5=msgMMSL.bPoint.p6;
          msgObj.bPoint.p6=msgMMSL.bPoint.p5;
          msgObj.bPoint.p7=msgMMSL.bPoint.p1;
#endif

          msgObj.fusionSourceId = 2;

          msgObj.header.stamp = rostime;
          msgObjArr.objects.push_back (msgObj);
        }
      }
      msgObjArr.header.stamp = rostime;
      msgObjArr.header.frame_id = frameId;
      LidarDetection_pub.publish (msgObjArr);

    }

    static void
    Send_CameraResults (CLUSTER_INFO* cluster_info,
                        int cluster_size,
                        ros::Time rostime,
                        string frameId)
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

ros::Publisher RosModuleHINO::ErrorCode_pub;
ros::Publisher RosModuleHINO::Rviz_pub;
ros::Publisher RosModuleHINO::LidarAllNonGround_pub;
ros::Publisher RosModuleHINO::LidarDetection_pub;
ros::Publisher RosModuleHINO::CameraDetection_pub;

#endif // ROSMODULE_H
