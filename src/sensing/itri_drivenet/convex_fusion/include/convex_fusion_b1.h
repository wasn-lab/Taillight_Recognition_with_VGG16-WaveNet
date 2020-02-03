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
  static ros::Publisher error_code_pub_;
  static ros::Publisher camera_detection_pub_;

  static void initial(std::string nodename, int argc, char** argv)
  {
    ros::init(argc, argv, nodename);
    ros::NodeHandle n;

    error_code_pub_ = n.advertise<msgs::ErrorCode>("/ErrorCode", 1);
    camera_detection_pub_ = n.advertise<msgs::DetectedObjectArray>(camera::detect_result_polygon, 1);
  }

  static void registerCallBackLidarAllNonGround(void (*callback0)(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&))
  {
    ros::NodeHandle n;

    static ros::Subscriber lidarall_nonground_sub = n.subscribe("/LidarAll/NonGround", 1, callback0);
  }

  static void registerCallBackCameraDetection(void (*callback0)(const msgs::DetectedObjectArray::ConstPtr&),
                                              void (*callback1)(const msgs::DetectedObjectArray::ConstPtr&),
                                              void (*callback2)(const msgs::DetectedObjectArray::ConstPtr&))
  {
    ros::NodeHandle n;

    static ros::Subscriber camera0_detection_sub = n.subscribe(camera::topics_obj[camera::id::front_60], 1, callback0);
    static ros::Subscriber camera1_detection_sub = n.subscribe(camera::topics_obj[camera::id::top_front_120], 1, callback1);
    static ros::Subscriber camera2_detection_sub = n.subscribe(camera::topics_obj[camera::id::top_rear_120], 1, callback2);
  }

  static void sendErrorCode(unsigned int error_code, std::string& frame_id, int module_id)
  {
    static uint32_t seq;

    msgs::ErrorCode objMsg;
    objMsg.header.seq = seq++;
    objMsg.header.stamp = ros::Time::now();
    objMsg.header.frame_id = frame_id;
    objMsg.module = module_id;
    objMsg.event = error_code;

    error_code_pub_.publish(objMsg);
  }

  static void sendCameraResults(CLUSTER_INFO* cluster_info, CLUSTER_INFO* cluster_info_bbox, int cluster_size,
                                 ros::Time rostime, std::string& frame_id)
  {
    msgs::DetectedObjectArray msgObjArr;
    float min_z = -3;
    float max_z = -1.5;
    for (int i = 0; i < cluster_size; i++)
    {
      msgs::DetectedObject msgObj;
      msgObj.classId = cluster_info[i].cluster_tag;
      size_t convex_hull_size = cluster_info[i].convex_hull.size();
      if (cluster_info[i].cluster_tag != 0)
      {
        if (convex_hull_size > 0)
        {
          // bottom
          for (size_t j = 0; j < convex_hull_size; j++)
          {
            msgs::PointXYZ p0;
            p0.x = cluster_info[i].convex_hull[j].x;
            p0.y = cluster_info[i].convex_hull[j].y;
            if (cluster_info[i].min.z < min_z)
              p0.z = min_z;
            else
              p0.z = cluster_info[i].min.z;
            msgObj.cPoint.lowerAreaPoints.push_back(p0);
          }
          // top
          for (size_t j = 0; j < convex_hull_size; j++)
          {
            msgs::PointXYZ p0;
            p0.x = cluster_info[i].convex_hull[j].x;
            p0.y = cluster_info[i].convex_hull[j].y;
            if (cluster_info[i].max.z > max_z)
              p0.z = max_z;
            else
              p0.z = cluster_info[i].max.z;
            msgObj.cPoint.lowerAreaPoints.push_back(p0);
          }
          // line
          for (size_t j = 0; j < convex_hull_size; j++)
          {
            msgs::PointXYZ p0;
            p0.x = cluster_info[i].convex_hull[j].x;
            p0.y = cluster_info[i].convex_hull[j].y;
            if (cluster_info[i].min.z < min_z)
              p0.z = min_z;
            else
              p0.z = cluster_info[i].min.z;
            msgObj.cPoint.lowerAreaPoints.push_back(p0);
            p0.x = cluster_info[i].convex_hull[j].x;
            p0.y = cluster_info[i].convex_hull[j].y;
            if (cluster_info[i].max.z > max_z)
              p0.z = max_z;
            else
              p0.z = cluster_info[i].max.z;
            msgObj.cPoint.lowerAreaPoints.push_back(p0);
          }
        }
        else
        {
          /// 3D cube
          ///   p5------p6
          ///   /|  2   /|
          /// p1-|----p2 |
          ///  |p4----|-p7
          ///  |/  1  | /
          /// p0-----P3

          msgs::PointXYZ p0, p1, p2, p3, p4, p5, p6, p7;

          // bottom
          p0.x = cluster_info_bbox[i].min.x;
          p0.y = cluster_info_bbox[i].min.y;
          p0.z = cluster_info_bbox[i].min.z;
          msgObj.cPoint.lowerAreaPoints.push_back(p0);
          p3.x = cluster_info_bbox[i].min.x;
          p3.y = cluster_info_bbox[i].max.y;
          p3.z = cluster_info_bbox[i].min.z;
          msgObj.cPoint.lowerAreaPoints.push_back(p3);
          p7.x = cluster_info_bbox[i].max.x;
          p7.y = cluster_info_bbox[i].max.y;
          p7.z = cluster_info_bbox[i].min.z;
          msgObj.cPoint.lowerAreaPoints.push_back(p7);
          p4.x = cluster_info_bbox[i].max.x;
          p4.y = cluster_info_bbox[i].min.y;
          p4.z = cluster_info_bbox[i].min.z;
          msgObj.cPoint.lowerAreaPoints.push_back(p4);
          msgObj.cPoint.lowerAreaPoints.push_back(p0);

          // top
          p1.x = cluster_info_bbox[i].min.x;
          p1.y = cluster_info_bbox[i].min.y;
          p1.z = cluster_info_bbox[i].max.z;
          msgObj.cPoint.lowerAreaPoints.push_back(p1);

          p2.x = cluster_info_bbox[i].min.x;
          p2.y = cluster_info_bbox[i].max.y;
          p2.z = cluster_info_bbox[i].max.z;
          msgObj.cPoint.lowerAreaPoints.push_back(p2);

          p6.x = cluster_info_bbox[i].max.x;
          p6.y = cluster_info_bbox[i].max.y;
          p6.z = cluster_info_bbox[i].max.z;
          msgObj.cPoint.lowerAreaPoints.push_back(p6);

          p5.x = cluster_info_bbox[i].max.x;
          p5.y = cluster_info_bbox[i].min.y;
          p5.z = cluster_info_bbox[i].max.z;
          msgObj.cPoint.lowerAreaPoints.push_back(p5);
          msgObj.cPoint.lowerAreaPoints.push_back(p1);

          // line
          msgObj.cPoint.lowerAreaPoints.push_back(p0);
          msgObj.cPoint.lowerAreaPoints.push_back(p2);
          msgObj.cPoint.lowerAreaPoints.push_back(p3);
          msgObj.cPoint.lowerAreaPoints.push_back(p6);
          msgObj.cPoint.lowerAreaPoints.push_back(p7);
          msgObj.cPoint.lowerAreaPoints.push_back(p6);
          msgObj.cPoint.lowerAreaPoints.push_back(p4);
          msgObj.cPoint.lowerAreaPoints.push_back(p5);
        }

        msgObj.cPoint.objectHigh = cluster_info[i].dz;

        msgObj.fusionSourceId = 0;

        msgObj.header.stamp = rostime;
        msgObjArr.objects.push_back(msgObj);
      }
    }
    msgObjArr.header.stamp = rostime;
    msgObjArr.header.frame_id = frame_id;
    camera_detection_pub_.publish(msgObjArr);
  }
};

ros::Publisher ConvexFusionB1::error_code_pub_;
ros::Publisher ConvexFusionB1::camera_detection_pub_;

#endif  // CONVEX_FUSION_B1_H
