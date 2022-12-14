#ifndef ROSMODULEHINO_H
#define ROSMODULEHINO_H

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <visualization_msgs/MarkerArray.h>

#include "msgs/DetectedObjectArray.h"
#include "msgs/ErrorCode.h"

#include "../UserDefine.h"

#include <fstream>
#include <boost/filesystem.hpp>
namespace BFS = boost::filesystem;
#define ENABLE_VALIDATION_MODE false

class RosModuleHINO
{
  public:

    static void
    initial (string nodename,
             int argc,
             char ** argv)
    {
      ros::init (argc, argv, nodename);
      ros::NodeHandle n;

      ErrorCode_pub = n.advertise<msgs::ErrorCode> ("/ErrorCode", 1);
      Rviz_pub = n.advertise<sensor_msgs::PointCloud2> ("/LidarAll/NonGround2", 1);
      LidarLeft_pub = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarLeft", 1);
      LidarRight_pub = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarRight", 1);
      LidarFront_pub = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarFront", 1);
      LidarTop_pub = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarTop", 1);
      LidarAll_pub = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarAll", 1);
      LidarAllNonGround_pub = n.advertise<PointCloud<PointXYZI>> ("/LidarAll/NonGround", 1);
      LidarDetection_pub = n.advertise<msgs::DetectedObjectArray> ("/LidarDetection", 1);
      LidarDetectionRVIZ_pub = n.advertise<visualization_msgs::MarkerArray> ("/LidarDetection/polygons", 1);
      CameraDetection_pub = n.advertise<msgs::DetectedObjectArray> ("/CameraDetection", 1);
    }

    static void
    RegisterCallBackLidarRaw (void
                              (*cb1) (const boost::shared_ptr<const sensor_msgs::PointCloud2>&),
                              void
                              (*cb2) (const boost::shared_ptr<const sensor_msgs::PointCloud2>&),
                              void
                              (*cb3) (const boost::shared_ptr<const sensor_msgs::PointCloud2>&),
                              void
                              (*cb4) (const boost::shared_ptr<const sensor_msgs::PointCloud2>&))
    {
      ros::NodeHandle n;
      static ros::Subscriber LidarFrontSub = n.subscribe ("/LidarFront/Raw", 1, cb1);
      static ros::Subscriber LidarLeftSub = n.subscribe ("/LidarLeft/Raw", 1, cb2);
      static ros::Subscriber LidarRightSub = n.subscribe ("/LidarRight/Raw", 1, cb3);
      static ros::Subscriber LidarTopSub = n.subscribe ("/LidarTop/Raw", 1, cb4);
    }

    static void
    RegisterCallBackLidar (void
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
    RegisterCallBackSSN (void
    (*cb) (const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr&))
    {
      ros::NodeHandle n;
      static ros::Subscriber result_cloud_P0deg = n.subscribe ("/squ_seg/result_cloud", 1, cb);
    }

    static void
    RegisterCallBackGT (void
    (*cb1) (const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr&))
    {
      ros::NodeHandle n;

      static ros::Subscriber GT_cloud = n.subscribe ("/LidarGT", 1, cb1);
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

    static void
    send_LidarAll (PointCloud<PointXYZI> inputLeft,
                   PointCloud<PointXYZI> inputRight,
                   PointCloud<PointXYZI> inputFront,
                   PointCloud<PointXYZI> inputTop,
                   PointCloud<PointXYZI> inputAll,
                   pcl::uint64_t pcltime,
                   string frameId)
    {
      inputLeft.header.frame_id = frameId;
      inputLeft.header.stamp = pcltime;
      inputRight.header.frame_id = frameId;
      inputRight.header.stamp = pcltime;
      inputFront.header.frame_id = frameId;
      inputFront.header.stamp = pcltime;
      inputTop.header.frame_id = frameId;
      inputTop.header.stamp = pcltime;
      inputAll.header.frame_id = frameId;
      inputAll.header.stamp = pcltime;

      LidarLeft_pub.publish (inputLeft);
      LidarRight_pub.publish (inputRight);
      LidarFront_pub.publish (inputFront);
      LidarTop_pub.publish (inputTop);
      LidarAll_pub.publish (inputAll);
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

#if ENABLE_VALIDATION_MODE == true
      // open a file to output convexhull
      BFS::path dir_output("/home/itri/self_driving_lidar/output");

      if(!BFS::exists(dir_output))
      BFS::create_directories(dir_output);

      ofstream convexFile;
      convexFile.open(dir_output.string() + "/" + frameId + ".txt");

      for (size_t i = 0; i < msgObjArr.objects.size(); i++)
      {
        convexFile << msgObjArr.objects[i].bPoint.p0.z << " "
        << msgObjArr.objects[i].bPoint.p1.z << " ";
        for (size_t j = 0; j < msgObjArr.objects[i].cPoint.lowerAreaPoints.size(); j++)
        {
          convexFile << msgObjArr.objects[i].cPoint.lowerAreaPoints[j].x << " " << msgObjArr.objects[i].cPoint.lowerAreaPoints[j].y << " ";
        }
        convexFile << "\n";
      }
      convexFile.close();

      cout << "OUTPUT: " << dir_output.string() + "/" + frameId + ".txt" << endl;
#endif

    }

    static void
    Send_LidarResultsRVIZ (CLUSTER_INFO* cluster_info,
                           int cluster_size)
    {
      visualization_msgs::MarkerArray markerArray;
      markerArray.markers.resize (cluster_size);

      size_t TrackID = 0;
      for (int i = 0; i < cluster_size; ++i)
      {
        markerArray.markers[i].header.frame_id = "lidar";
        markerArray.markers[i].header.stamp = ros::Time ();
        markerArray.markers[i].id = TrackID++;
        markerArray.markers[i].action = visualization_msgs::Marker::ADD;
        markerArray.markers[i].type = visualization_msgs::Marker::LINE_STRIP;
        markerArray.markers[i].pose.orientation.w = 1.0;
        markerArray.markers[i].scale.x = 0.1;
        markerArray.markers[i].color.r = 0.0;
        markerArray.markers[i].color.g = 1.0;
        markerArray.markers[i].color.b = 0.0;
        markerArray.markers[i].color.a = 1.0;
        markerArray.markers[i].lifetime = ros::Duration (0.1);

        markerArray.markers[i].points.resize(cluster_info[i].convex_hull.size ()+1);
        for (size_t j = 0; j < cluster_info[i].convex_hull.size (); ++j)
        {
          markerArray.markers[i].points[j].x = cluster_info[i].convex_hull.points[j].x;
          markerArray.markers[i].points[j].y = cluster_info[i].convex_hull.points[j].y;
          markerArray.markers[i].points[j].z = cluster_info[i].convex_hull.points[j].z;
          if(j == (cluster_info[i].convex_hull.size ()-1)){
            markerArray.markers[i].points[j+1].x = cluster_info[i].convex_hull.points[0].x;
            markerArray.markers[i].points[j+1].y = cluster_info[i].convex_hull.points[0].y;
            markerArray.markers[i].points[j+1].z = cluster_info[i].convex_hull.points[0].z;
          }
        }

      }
      LidarDetectionRVIZ_pub.publish (markerArray);
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

  private:

    static ros::Publisher ErrorCode_pub;
    static ros::Publisher Rviz_pub;
    static ros::Publisher LidarLeft_pub;
    static ros::Publisher LidarRight_pub;
    static ros::Publisher LidarFront_pub;
    static ros::Publisher LidarTop_pub;
    static ros::Publisher LidarAll_pub;
    static ros::Publisher LidarAllNonGround_pub;
    static ros::Publisher LidarDetection_pub;
    static ros::Publisher LidarDetectionRVIZ_pub;
    static ros::Publisher CameraDetection_pub;

};

ros::Publisher RosModuleHINO::ErrorCode_pub;
ros::Publisher RosModuleHINO::Rviz_pub;
ros::Publisher RosModuleHINO::LidarLeft_pub;
ros::Publisher RosModuleHINO::LidarRight_pub;
ros::Publisher RosModuleHINO::LidarFront_pub;
ros::Publisher RosModuleHINO::LidarTop_pub;
ros::Publisher RosModuleHINO::LidarAll_pub;
ros::Publisher RosModuleHINO::LidarAllNonGround_pub;
ros::Publisher RosModuleHINO::LidarDetection_pub;
ros::Publisher RosModuleHINO::LidarDetectionRVIZ_pub;
ros::Publisher RosModuleHINO::CameraDetection_pub;

#endif // ROSMODULE_H
