#ifndef ROSMODULEB1_H
#define ROSMODULEB1_H

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <visualization_msgs/MarkerArray.h>
#include <rosgraph_msgs/Clock.h>

#include "msgs/DetectedObjectArray.h"
#include "msgs/ErrorCode.h"

#include "../ToControl/GridMapGen/points_to_costmap.h"
#include "../UserDefine.h"

class RosModuleB1
{
  public:

    static void
    initial (string nodename,
             int argc,
             char ** argv)
    {
      ros::init (argc, argv, nodename);
    }

    static void
    RegisterCallBackClock (void
    (*cb1) (const rosgraph_msgs::Clock&))
    {
      static ros::Subscriber ClockSub = ros::NodeHandle ().subscribe ("/clock", 1, cb1);
    }

    static void
    RegisterCallBackLidarAll (void
    (*cb1) (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&))
    {
      static ros::Subscriber LidarAllSub = ros::NodeHandle ().subscribe ("/LidarAll", 1, cb1);
    }

    static void
    RegisterCallBackLidar (void
                           (*cb1) (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&),
                           void
                           (*cb2) (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&),
                           void
                           (*cb3) (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&),
                           void
                           (*cb4) (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&),
                           void
                           (*cb5) (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&))
    {
      static ros::Subscriber LidarFrontRightSub = ros::NodeHandle ().subscribe ("/LidarFrontRight", 1, cb1);
      static ros::Subscriber LidarFrontLeftSub = ros::NodeHandle ().subscribe ("/LidarFrontLeft", 1, cb2);
      static ros::Subscriber LidarRearRightSub = ros::NodeHandle ().subscribe ("/LidarRearRight", 1, cb3);
      static ros::Subscriber LidarRearLeftSub = ros::NodeHandle ().subscribe ("/LidarRearLeft", 1, cb4);
      static ros::Subscriber LidarFrontTopSub = ros::NodeHandle ().subscribe ("/LidarFrontTop", 1, cb5);
    }

    static void
    RegisterCallBackSSN (void
    (*cb) (const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr&))
    {
      static ros::Subscriber result_cloud_P0deg = ros::NodeHandle ().subscribe ("/squ_seg/result_cloud", 1, cb);
    }

    static void
    RegisterCallBackIMU (void
    (*cb) (const sensor_msgs::Imu::ConstPtr&))
    {
      static ros::Subscriber ImuSub = ros::NodeHandle ().subscribe ("/imu/data", 1, cb);
    }

    static void
    RegisterCallBackTimer (void
    (*cb) (const ros::TimerEvent&), float interval)
    {
      static ros::Timer timer = ros::NodeHandle ().createTimer(ros::Duration(interval), cb);
    }

    static void
    send_ErrorCode (unsigned int error_code)
    {
      static ros::Publisher ErrorCode_pub = ros::NodeHandle ().advertise<msgs::ErrorCode> ("/ErrorCode", 1);

      static uint32_t seq;

      msgs::ErrorCode objMsg;
      objMsg.header.seq = seq++;
      objMsg.header.stamp = ros::Time::now ();
      objMsg.module = 1;
      objMsg.event = error_code;

      ErrorCode_pub.publish (objMsg);
    }

    template <typename PointT>
    static void
    send_LidarAllNonGround (PointCloud<PointT> input,
                            pcl::uint64_t pcltime,
                            string frameId)
    {
      static ros::Publisher LidarAllNonGround_pub = ros::NodeHandle ().advertise<PointCloud<PointXYZI>> ("/LidarAll/NonGround", 1);
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
      static ros::Publisher LidarDetection_pub = ros::NodeHandle ().advertise<msgs::DetectedObjectArray> ("/LidarDetection", 1);

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
    Send_LidarResultsRVIZ (CLUSTER_INFO* cluster_info,
                           int cluster_size)
    {
      static ros::Publisher LidarDetectionRVIZ_pub = ros::NodeHandle ().advertise<visualization_msgs::MarkerArray> ("/LidarDetection/polygons", 1);

      visualization_msgs::MarkerArray markerArray;
      markerArray.markers.resize (cluster_size);

      size_t TrackID = 0;
      for (int i = 0; i < cluster_size; ++i)
      {
        markerArray.markers[i].header.frame_id = "lidar";
        markerArray.markers[i].header.stamp = ros::Time ();
        markerArray.markers[i].id = TrackID++;
        markerArray.markers[i].action = visualization_msgs::Marker::ADD;
        markerArray.markers[i].type = visualization_msgs::Marker::LINE_LIST;
        markerArray.markers[i].pose.orientation.w = 1.0;
        markerArray.markers[i].scale.x = 0.05;

        switch (cluster_info[i].cluster_tag)
        {
          case 1:
            markerArray.markers[i].color.r = 0.0;
            markerArray.markers[i].color.g = 1.0;
            markerArray.markers[i].color.b = 1.0;
            markerArray.markers[i].color.a = 1.0;
            break;
          case 2:
            markerArray.markers[i].color.r = 1.0;
            markerArray.markers[i].color.g = 0.0;
            markerArray.markers[i].color.b = 1.0;
            markerArray.markers[i].color.a = 1.0;
            break;
          case 3:
            markerArray.markers[i].color.r = 0.0;
            markerArray.markers[i].color.g = 1.0;
            markerArray.markers[i].color.b = 0.0;
            markerArray.markers[i].color.a = 1.0;
            break;
            ;
          default:
            markerArray.markers[i].color.r = 1.0;
            markerArray.markers[i].color.g = 0.0;
            markerArray.markers[i].color.b = 0.0;
            markerArray.markers[i].color.a = 1.0;
        }

        markerArray.markers[i].lifetime = ros::Duration (0.1);

        // draw a 2D polygon at the botton
        for (size_t j = 0; j < cluster_info[i].convex_hull.size (); ++j)
        {
          geometry_msgs::Point p;

          p.x = cluster_info[i].convex_hull.points[j].x;
          p.y = cluster_info[i].convex_hull.points[j].y;
          p.z = cluster_info[i].min.z;
          markerArray.markers[i].points.push_back (p);

          if (j == cluster_info[i].convex_hull.size () - 1)
          {
            p.x = cluster_info[i].convex_hull.points[0].x;
            p.y = cluster_info[i].convex_hull.points[0].y;
            p.z = cluster_info[i].min.z;
            markerArray.markers[i].points.push_back (p);
          }
          else
          {
            p.x = cluster_info[i].convex_hull.points[j + 1].x;
            p.y = cluster_info[i].convex_hull.points[j + 1].y;
            p.z = cluster_info[i].min.z;
            markerArray.markers[i].points.push_back (p);
          }
        }

#if(true)
        // draw a 2D polygon at the top
        for (size_t j = 0; j < cluster_info[i].convex_hull.size (); ++j)
        {
          geometry_msgs::Point p;

          p.x = cluster_info[i].convex_hull.points[j].x;
          p.y = cluster_info[i].convex_hull.points[j].y;
          p.z = cluster_info[i].max.z;
          markerArray.markers[i].points.push_back (p);

          if (j == cluster_info[i].convex_hull.size () - 1)
          {
            p.x = cluster_info[i].convex_hull.points[0].x;
            p.y = cluster_info[i].convex_hull.points[0].y;
            p.z = cluster_info[i].max.z;
            markerArray.markers[i].points.push_back (p);
          }
          else
          {
            p.x = cluster_info[i].convex_hull.points[j + 1].x;
            p.y = cluster_info[i].convex_hull.points[j + 1].y;
            p.z = cluster_info[i].max.z;
            markerArray.markers[i].points.push_back (p);
          }
        }

        // draw vertical lines of polygon box
        for (size_t j = 0; j < cluster_info[i].convex_hull.size (); ++j)
        {
          geometry_msgs::Point p;

          p.x = cluster_info[i].convex_hull.points[j].x;
          p.y = cluster_info[i].convex_hull.points[j].y;
          p.z = cluster_info[i].min.z;
          markerArray.markers[i].points.push_back (p);

          p.x = cluster_info[i].convex_hull.points[j].x;
          p.y = cluster_info[i].convex_hull.points[j].y;
          p.z = cluster_info[i].max.z;
          markerArray.markers[i].points.push_back (p);
        }
#endif
      }
      LidarDetectionRVIZ_pub.publish (markerArray);
    }

    static void
    Send_LidarResultsGrid (CLUSTER_INFO* cluster_info,
                           int cluster_size,
                           ros::Time rostime,
                           string frameId)
    {
      static ros::Publisher gridmap_pub = ros::NodeHandle ().advertise<nav_msgs::OccupancyGrid> ("/LidarDetection/grid", 1);

      VPointCloudXYZIL::Ptr input_Cloud (new VPointCloudXYZIL);
      for (int i = 0; i < cluster_size; i++)
      {
        if (cluster_info[i].cluster_tag >= 1)
        {
          *input_Cloud += cluster_info[i].cloud_IL;
        }
      }

      nav_msgs::OccupancyGrid occupancyGrid_msg;
      grid_map::GridMapRosConverter::toOccupancyGrid (PointsToCostmap ().makeGridMap<PointXYZIL> (input_Cloud), "points_layer", 0, 1, occupancyGrid_msg);
      occupancyGrid_msg.header.stamp = rostime;
      occupancyGrid_msg.header.frame_id = frameId;
      gridmap_pub.publish (occupancyGrid_msg);
    }

};

#endif // ROSMODULE_H
