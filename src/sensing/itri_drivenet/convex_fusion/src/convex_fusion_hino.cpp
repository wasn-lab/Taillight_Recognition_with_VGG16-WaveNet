#include "convex_fusion_hino.h"

void ConvexFusionHino::initial(std::string nodename, int argc, char** argv)
{
  ros::init(argc, argv, nodename);
  ros::NodeHandle n;

  ErrorCode_pub = n.advertise<msgs::ErrorCode>("/ErrorCode", 1);
  CameraDetection_pub = n.advertise<msgs::DetectedObjectArray>("/CameraDetection", 1);
}

void ConvexFusionHino::RegisterCallBackLidarAllNonGround(void (*cb1)(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&))
{
  ros::NodeHandle n;

  ros::Subscriber LidarAllNonGroundSub = n.subscribe("/LidarAll/NonGround", 1, cb1);
}

void ConvexFusionHino::RegisterCallBackCameraDetection(void (*cb1)(const msgs::DetectedObjectArray::ConstPtr&),
                                            void (*cb2)(const msgs::DetectedObjectArray::ConstPtr&),
                                            void (*cb3)(const msgs::DetectedObjectArray::ConstPtr&))
{
  ros::NodeHandle n;

  ros::Subscriber Camera30DetectionSub = n.subscribe("/DetectedObjectArray/cam30", 1, cb1);
  ros::Subscriber Camera60DetectionSub = n.subscribe("/DetectedObjectArray/cam60", 1, cb2);
  ros::Subscriber Camera120DetectionSub = n.subscribe("/DetectedObjectArray/cam120", 1, cb3);
}

void ConvexFusionHino::send_ErrorCode(unsigned int error_code)
{
  uint32_t seq;

  msgs::ErrorCode objMsg;
  objMsg.header.seq = seq++;
  objMsg.header.stamp = ros::Time::now();
  objMsg.header.frame_id = "lidar";
  objMsg.module = 1;
  objMsg.event = error_code;

  ErrorCode_pub.publish(objMsg);
}

void ConvexFusionHino::Send_CameraResults(CLUSTER_INFO* cluster_info, int cluster_size, ros::Time rostime, std::string frameId)
{
  msgs::DetectedObjectArray msgObjArr;

  for (int i = 0; i < cluster_size; i++)
  {
    msgs::DetectedObject msgObj;
    msgObj.classId = cluster_info[i].cluster_tag;

    if (cluster_info[i].convex_hull.size() > 0)
    {
      msgObj.cPoint.lowerAreaPoints.resize(cluster_info[i].convex_hull.size());
      for (size_t j = 0; j < cluster_info[i].convex_hull.size(); j++)
      {
        msgObj.cPoint.lowerAreaPoints[j].x = cluster_info[i].convex_hull[j].x;
        msgObj.cPoint.lowerAreaPoints[j].y = cluster_info[i].convex_hull[j].y;
        msgObj.cPoint.lowerAreaPoints[j].z = cluster_info[i].convex_hull[j].z;
      }

      msgObj.cPoint.objectHigh = cluster_info[i].dz;

      msgObj.fusionSourceId = 0;

      msgObj.header.stamp = rostime;
      msgObjArr.objects.push_back(msgObj);
    }
  }
  msgObjArr.header.stamp = rostime;
  msgObjArr.header.frame_id = frameId;
  CameraDetection_pub.publish(msgObjArr);
}