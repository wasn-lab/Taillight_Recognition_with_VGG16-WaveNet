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

  static void registerCallBackLidarAllNonGround(void (*callback_nonground)(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&))
  {
    ros::NodeHandle n;

    static ros::Subscriber lidarall_nonground_sub = n.subscribe("/LidarAll/NonGround", 1, callback_nonground);
  }

  static void registerCallBackCameraDetection(void (*callback_front_60)(const msgs::DetectedObjectArray::ConstPtr&),
                                              void (*callback_top_front_120)(const msgs::DetectedObjectArray::ConstPtr&),
                                              void (*callback_top_rear_120)(const msgs::DetectedObjectArray::ConstPtr&))
  {
    ros::NodeHandle n;

    static ros::Subscriber camera_front_60_detection_sub = n.subscribe(camera::topics_obj[camera::id::front_60], 1, callback_front_60);
    static ros::Subscriber camera_top_front_120_detection_sub = n.subscribe(camera::topics_obj[camera::id::top_front_120], 1, callback_top_front_120);
    static ros::Subscriber camera_top_rear_120_detection_sub = n.subscribe(camera::topics_obj[camera::id::top_rear_120], 1, callback_top_rear_120);
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
            msgs::PointXYZ convex_point;
            convex_point.x = cluster_info[i].convex_hull[j].x;
            convex_point.y = cluster_info[i].convex_hull[j].y;
            convex_point.z = std::min(min_z, cluster_info[i].min.z);
            msgObj.cPoint.lowerAreaPoints.push_back(convex_point);
          }
          // top
          for (size_t j = 0; j < convex_hull_size; j++)
          {
            msgs::PointXYZ convex_point;
            convex_point.x = cluster_info[i].convex_hull[j].x;
            convex_point.y = cluster_info[i].convex_hull[j].y;
            convex_point.z = std::max(max_z, cluster_info[i].max.z);
            msgObj.cPoint.lowerAreaPoints.push_back(convex_point);
          }
          // line
          for (size_t j = 0; j < convex_hull_size; j++)
          {
            msgs::PointXYZ convex_point;
            convex_point.x = cluster_info[i].convex_hull[j].x;
            convex_point.y = cluster_info[i].convex_hull[j].y;
            convex_point.z = std::min(min_z, cluster_info[i].min.z);
            msgObj.cPoint.lowerAreaPoints.push_back(convex_point);
            convex_point.x = cluster_info[i].convex_hull[j].x;
            convex_point.y = cluster_info[i].convex_hull[j].y;
            convex_point.z = std::max(max_z, cluster_info[i].max.z);
            msgObj.cPoint.lowerAreaPoints.push_back(convex_point);
          }
        }
        else
        {
          /// Coordinate system
          ///           ^          ///             ^
          ///      ^   /           ///        ^   /
          ///    z |  /            ///      z |  /
          ///      | /  y          ///        | /  x
          ///      ----->          ///   <-----    
          ///        x             ///       y

          /// cluster_info_bbox    ///  bbox_p0
          ///   p6------p2         ///   p5------p6
          ///   /|  2   /|         ///   /|  2   /|
          /// p5-|----p1 |         /// p1-|----p2 |
          ///  |p7----|-p3   ->    ///  |p4----|-p7
          ///  |/  1  | /          ///  |/  1  | /
          /// p4-----P0            /// p0-----P3

          // Use Cartesian coordinate system. min point of bbox is p0, max point of bbox is p6; 
          msgs::PointXYZ bbox_p0, bbox_p1, bbox_p2, bbox_p3, bbox_p4, bbox_p5, bbox_p6, bbox_p7;
          // bottom
          bbox_p0.x = cluster_info_bbox[i].min.x;
          bbox_p0.y = cluster_info_bbox[i].min.y;
          bbox_p0.z = cluster_info_bbox[i].min.z;
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p0);
          bbox_p3.x = cluster_info_bbox[i].min.x;
          bbox_p3.y = cluster_info_bbox[i].max.y;
          bbox_p3.z = cluster_info_bbox[i].min.z;
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p3);
          bbox_p7.x = cluster_info_bbox[i].max.x;
          bbox_p7.y = cluster_info_bbox[i].max.y;
          bbox_p7.z = cluster_info_bbox[i].min.z;
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p7);
          bbox_p4.x = cluster_info_bbox[i].max.x;
          bbox_p4.y = cluster_info_bbox[i].min.y;
          bbox_p4.z = cluster_info_bbox[i].min.z;
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p4);
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p0);

          // top
          bbox_p1.x = cluster_info_bbox[i].min.x;
          bbox_p1.y = cluster_info_bbox[i].min.y;
          bbox_p1.z = cluster_info_bbox[i].max.z;
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p1);

          bbox_p2.x = cluster_info_bbox[i].min.x;
          bbox_p2.y = cluster_info_bbox[i].max.y;
          bbox_p2.z = cluster_info_bbox[i].max.z;
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p2);

          bbox_p6.x = cluster_info_bbox[i].max.x;
          bbox_p6.y = cluster_info_bbox[i].max.y;
          bbox_p6.z = cluster_info_bbox[i].max.z;
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p6);

          bbox_p5.x = cluster_info_bbox[i].max.x;
          bbox_p5.y = cluster_info_bbox[i].min.y;
          bbox_p5.z = cluster_info_bbox[i].max.z;
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p5);
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p1);

          // line
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p0);
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p2);
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p3);
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p6);
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p7);
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p6);
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p4);
          msgObj.cPoint.lowerAreaPoints.push_back(bbox_p5);
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
