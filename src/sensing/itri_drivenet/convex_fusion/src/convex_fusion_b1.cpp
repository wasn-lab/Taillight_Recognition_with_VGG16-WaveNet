#include "convex_fusion_b1.h"

void ConvexFusionB1::initial(std::string nodename, int argc, char** argv)
{
  ros::init(argc, argv, nodename);
  ros::NodeHandle n;

  error_code_pub_ = n.advertise<msgs::ErrorCode>("/ErrorCode", 1);
  camera_detection_pub_ = n.advertise<msgs::DetectedObjectArray>(camera::detect_result_polygon, 1);
  occupancy_grid_publisher = n.advertise<nav_msgs::OccupancyGrid>("/CameraDetection/occupancy_grid", 1, true);
}

void ConvexFusionB1::registerCallBackLidarAllNonGround(
    void (*callback_nonground)(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&))
{
  ros::NodeHandle n;

  static ros::Subscriber lidarall_nonground_sub = n.subscribe("/LidarAll/NonGround", 1, callback_nonground);
}

void ConvexFusionB1::registerCallBackCameraDetection(
    void (*callback_front_60)(const msgs::DetectedObjectArray::ConstPtr&),
    void (*callback_top_front_120)(const msgs::DetectedObjectArray::ConstPtr&),
    void (*callback_top_rear_120)(const msgs::DetectedObjectArray::ConstPtr&))
{
  ros::NodeHandle n;

  static ros::Subscriber camera_front_60_detection_sub =
      n.subscribe(camera::topics_obj[camera::id::front_60], 1, callback_front_60);
  static ros::Subscriber camera_top_front_120_detection_sub =
      n.subscribe(camera::topics_obj[camera::id::top_front_120], 1, callback_top_front_120);
  static ros::Subscriber camera_top_rear_120_detection_sub =
      n.subscribe(camera::topics_obj[camera::id::top_rear_120], 1, callback_top_rear_120);
}

void ConvexFusionB1::sendErrorCode(unsigned int error_code, std::string& frame_id, int module_id)
{
  static uint32_t seq = 0;

  msgs::ErrorCode objMsg;
  objMsg.header.seq = seq++;
  objMsg.header.stamp = ros::Time::now();
  objMsg.header.frame_id = frame_id;
  objMsg.module = module_id;
  objMsg.event = error_code;

  error_code_pub_.publish(objMsg);
}

void ConvexFusionB1::sendCameraResults(CLUSTER_INFO* cluster_info, CLUSTER_INFO* cluster_info_bbox, int cluster_size,
                                       ros::Time rostime, std::string& frame_id)
{
  if (g_use_gridmap_publish)
  {
    g_costmap_ = g_cosmapGener.initGridMap();
  }

  msgs::DetectedObjectArray msgObjArr;
  float min_z = -3;
  float max_z = -1.5;
  for (int i = 0; i < cluster_size; i++)
  {
    msgs::DetectedObject msgObj;
    msgObj.distance = -1;
    msgObj.classId = cluster_info[i].cluster_tag;
    size_t convex_hull_size = cluster_info[i].convex_hull.size();
    if (cluster_info[i].cluster_tag != 0)
    {
      if (convex_hull_size > 0)
      {
        msgObj.distance = 0;
        float bottom_z = std::min(min_z, cluster_info_bbox[i].min.z);
        float top_z = std::max(max_z, cluster_info_bbox[i].max.z);
        msgObj.cPoint.objectHigh = top_z - bottom_z;

        // bottom
        for (size_t j = 0; j < convex_hull_size; j++)
        {
          msgs::PointXYZ convex_point;
          convex_point.x = cluster_info[i].convex_hull[j].x;
          convex_point.y = cluster_info[i].convex_hull[j].y;
          convex_point.z = bottom_z;
          msgObj.cPoint.lowerAreaPoints.push_back(convex_point);
        }

        if (g_use_gridmap_publish)
        {
          // object To grid map
          g_costmap_[g_cosmapGener.layer_name_] =
              g_cosmapGener.makeCostmapFromSingleObject(g_costmap_, g_cosmapGener.layer_name_, 8, msgObj, true);
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

        msgs::PointXYZ bbox_p0, bbox_p1, bbox_p2, bbox_p3, bbox_p4, bbox_p5, bbox_p6, bbox_p7;
        msgObj.distance = 0;
        float bottom_z = std::min(min_z, cluster_info_bbox[i].min.z);
        float top_z = std::max(max_z, cluster_info_bbox[i].max.z);
        msgObj.cPoint.objectHigh = top_z - bottom_z;

        // bottom
        bbox_p0.x = cluster_info_bbox[i].min.x;
        bbox_p0.y = cluster_info_bbox[i].min.y;
        bbox_p0.z = bottom_z;
        msgObj.cPoint.lowerAreaPoints.push_back(bbox_p0);
        msgObj.bPoint.p0 = bbox_p0;
        bbox_p3.x = cluster_info_bbox[i].min.x;
        bbox_p3.y = cluster_info_bbox[i].max.y;
        bbox_p3.z = bottom_z;
        msgObj.cPoint.lowerAreaPoints.push_back(bbox_p3);
        msgObj.bPoint.p3 = bbox_p3;
        bbox_p7.x = cluster_info_bbox[i].max.x;
        bbox_p7.y = cluster_info_bbox[i].max.y;
        bbox_p7.z = bottom_z;
        msgObj.cPoint.lowerAreaPoints.push_back(bbox_p7);
        msgObj.bPoint.p7 = bbox_p7;
        bbox_p4.x = cluster_info_bbox[i].max.x;
        bbox_p4.y = cluster_info_bbox[i].min.y;
        bbox_p4.z = bottom_z;
        msgObj.cPoint.lowerAreaPoints.push_back(bbox_p4);
        msgObj.bPoint.p4 = bbox_p4;

        if (g_use_gridmap_publish)
        {
          // object To grid map
          g_costmap_[g_cosmapGener.layer_name_] =
              g_cosmapGener.makeCostmapFromSingleObject(g_costmap_, g_cosmapGener.layer_name_, 8, msgObj, false);
        }
      }

      msgObj.fusionSourceId = 0;

      msgObj.header.stamp = rostime;
      msgObjArr.objects.push_back(msgObj);
    }
  }
  msgObjArr.header.stamp = rostime;
  msgObjArr.header.frame_id = frame_id;

  if (g_use_gridmap_publish)
  {
    // grid map To Occpancy publisher
    g_cosmapGener.OccupancyMsgPublisher(g_costmap_, occupancy_grid_publisher, msgObjArr.header);
  }
  camera_detection_pub_.publish(msgObjArr);
}