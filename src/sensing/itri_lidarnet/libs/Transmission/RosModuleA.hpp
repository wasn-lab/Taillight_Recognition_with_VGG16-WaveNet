#ifndef ROSMODULEA_H
#define ROSMODULEA_H

#include <ros/ros.h>
/*
class RosModuleA
{
  public:

    static void
    initial (int argc,
             char ** argv)
    {
      ros::init (argc, argv, "display");

    }

    static void
    RegisterCallBackCoords (void
    (*cb1) (const msgs::TransfObj::ConstPtr& ))
    {
      ros::NodeHandle n;
      static ros::Subscriber Coords = n.subscribe ("/coords_transf", 1, cb1);

    }

    static void
    RegisterCallBackMap (void
    (*cb1) (const sensor_msgs::PointCloud2ConstPtr&))
    {
      ros::NodeHandle n;
      static ros::Subscriber PointsMap = n.subscribe ("/points_map", 1, cb1);

    }

    static void
    RegisterCallBackMapPose (void
    (*cb1) (const geometry_msgs::PoseStamped&))
    {
      ros::NodeHandle n;
      static ros::Subscriber CurrentPose = n.subscribe ("/current_pose", 1, cb1);
    }

    static void
    RegisterCallBackLidar (void
                           (*cb1) (const msgs::PointCloud::ConstPtr&),
                           void
                           (*cb2) (const msgs::PointCloud::ConstPtr&),
                           void
                           (*cb3) (const msgs::PointCloud::ConstPtr&),
                           void
                           (*cb4) (const msgs::PointCloud::ConstPtr&),
                           void
                           (*cb5) (const msgs::PointCloud::ConstPtr&))
    {
      ros::NodeHandle n;
      static ros::Subscriber LidFrontTopSub = n.subscribe ("LidFrontTop", 1, cb1);
      static ros::Subscriber LidFrontLeftSub = n.subscribe ("LidFrontRight", 1, cb2);
      static ros::Subscriber LidFrontRightSub = n.subscribe ("LidFrontLeft", 1, cb3);
      static ros::Subscriber LidRearLeftSub = n.subscribe ("LidRearLeft", 1, cb4);
      static ros::Subscriber LidRearRightSub = n.subscribe ("LidRearRight", 1, cb5);
      static ros::Subscriber LidFrontTopSub = n.subscribe ("LidFront", 1, cb1);
      static ros::Subscriber LidFrontLeftSub = n.subscribe ("LidLeft", 1, cb2);
      static ros::Subscriber LidFrontRightSub = n.subscribe ("LidRight", 1, cb3);
      static ros::Subscriber LidRearLeftSub = n.subscribe ("LidTop", 1, cb4);
    }

    static void
    RegisterCallBackResults (void
                             (*cb1) (const msgs::LidRoi::ConstPtr&),
                             void
                             (*cb2) (const msgs::CamRoi::ConstPtr&))
    {
      ros::NodeHandle n;
      static ros::Subscriber LidRoiSub = n.subscribe ("LidRoi", 1, cb1);
      static ros::Subscriber CamRoiSub = n.subscribe ("CamRoi", 1, cb2);

    }

};
*/

/*
class RosModuleA
{
  public:

    static ros::Publisher LidObj_pub;
    static ros::Publisher LidRoi_pub;
    static ros::Publisher LidFus_pub;

    static ros::Publisher LidStitch_pub;

    static ros::Publisher ErrorCode_pub;

    static void
    initial (int argc,
             char ** argv)
    {
      ros::init (argc, argv, "Lid");
      ros::NodeHandle n;
      //LidObj_pub = n.advertise<msgs::LidObj> ("/LidObj", 1);
      //LidRoi_pub = n.advertise<msgs::LidRoi> ("/LidRoi", 1);
      //LidFus_pub = n.advertise<msgs::LidFus> ("/LidFus", 1);

      //LidStitch_pub = n.advertise<msgs::PointCloud> ("/LidFrontLeft_sync", 1);

      //ErrorCode_pub = n.advertise<msgs::ErrorCode> ("/ErrorCode", 1);

    }


    static void
    send_LidObj (CLUSTER_INFO* cluster_info,
                 int cluster_size)
    {
      static uint32_t seq;

      msgs::LidObj objMsg;
      objMsg.lidHeader.seq = seq++;
      objMsg.lidHeader.stamp = GlobalVariable::ROS_TIMESTAMP;

      for (int i = 0; i < cluster_size; i++)
      {
        if (cluster_info[i].cluster_tag >= 1)
        {
          msgs::LidBox obj;
          obj.p0.x = cluster_info[i].obb_vertex.at (0).x;
          obj.p0.y = cluster_info[i].obb_vertex.at (0).y;
          obj.p0.z = cluster_info[i].obb_vertex.at (0).z;
          obj.p1.x = cluster_info[i].obb_vertex.at (1).x;
          obj.p1.y = cluster_info[i].obb_vertex.at (1).y;
          obj.p1.z = cluster_info[i].obb_vertex.at (1).z;
          obj.p2.x = cluster_info[i].obb_vertex.at (2).x;
          obj.p2.y = cluster_info[i].obb_vertex.at (2).y;
          obj.p2.z = cluster_info[i].obb_vertex.at (2).z;
          obj.p3.x = cluster_info[i].obb_vertex.at (3).x;
          obj.p3.y = cluster_info[i].obb_vertex.at (3).y;
          obj.p3.z = cluster_info[i].obb_vertex.at (3).z;
          obj.p4.x = cluster_info[i].obb_vertex.at (4).x;
          obj.p4.y = cluster_info[i].obb_vertex.at (4).y;
          obj.p4.z = cluster_info[i].obb_vertex.at (4).z;
          obj.p5.x = cluster_info[i].obb_vertex.at (5).x;
          obj.p5.y = cluster_info[i].obb_vertex.at (5).y;
          obj.p5.z = cluster_info[i].obb_vertex.at (5).z;
          obj.p6.x = cluster_info[i].obb_vertex.at (6).x;
          obj.p6.y = cluster_info[i].obb_vertex.at (6).y;
          obj.p6.z = cluster_info[i].obb_vertex.at (6).z;
          obj.p7.x = cluster_info[i].obb_vertex.at (7).x;
          obj.p7.y = cluster_info[i].obb_vertex.at (7).y;
          obj.p7.z = cluster_info[i].obb_vertex.at (7).z;

          switch (cluster_info[i].cluster_tag)
          {
            case 1:
              obj.cls = -1;
              break;
            case 2:
              obj.cls = 0;
              break;
            case 3:
              obj.cls = 3;
              break;
            case 4:
              obj.cls = 2;
              break;
            case 5:
              obj.cls = 5;
              break;
          }

          obj.prob = (float) cluster_info[i].confidence / 100;
          obj.speed = geometry::distance (cluster_info[i].velocity, PointXYZ (0, 0, 0));

          objMsg.lidObj.push_back (obj);
        }
      }

      LidObj_pub.publish (objMsg);
    }

    static void
    send_LidRoi (CLUSTER_INFO* cluster_info,
                 int cluster_size)
    {
      static uint32_t seq;

      msgs::LidRoi objMsg;
      objMsg.lidHeader.seq = seq++;
      objMsg.lidHeader.stamp = GlobalVariable::ROS_TIMESTAMP;

      for (int i = 0; i < cluster_size; i++)
      {
        if (cluster_info[i].cluster_tag >= 1)
        {
          msgs::LidRoiBox obj;
                    obj.p0.x = cluster_info[i].obb_vertex.at (0).x;
           obj.p0.y = cluster_info[i].obb_vertex.at (0).y;
           obj.p0.z = cluster_info[i].obb_vertex.at (0).z;
           obj.p1.x = cluster_info[i].obb_vertex.at (1).x;
           obj.p1.y = cluster_info[i].obb_vertex.at (1).y;
           obj.p1.z = cluster_info[i].obb_vertex.at (1).z;
           obj.p2.x = cluster_info[i].obb_vertex.at (2).x;
           obj.p2.y = cluster_info[i].obb_vertex.at (2).y;
           obj.p2.z = cluster_info[i].obb_vertex.at (2).z;
           obj.p3.x = cluster_info[i].obb_vertex.at (3).x;
           obj.p3.y = cluster_info[i].obb_vertex.at (3).y;
           obj.p3.z = cluster_info[i].obb_vertex.at (3).z;
           obj.p4.x = cluster_info[i].obb_vertex.at (4).x;
           obj.p4.y = cluster_info[i].obb_vertex.at (4).y;
           obj.p4.z = cluster_info[i].obb_vertex.at (4).z;
           obj.p5.x = cluster_info[i].obb_vertex.at (5).x;
           obj.p5.y = cluster_info[i].obb_vertex.at (5).y;
           obj.p5.z = cluster_info[i].obb_vertex.at (5).z;
           obj.p6.x = cluster_info[i].obb_vertex.at (6).x;
           obj.p6.y = cluster_info[i].obb_vertex.at (6).y;
           obj.p6.z = cluster_info[i].obb_vertex.at (6).z;
           obj.p7.x = cluster_info[i].obb_vertex.at (7).x;
           obj.p7.y = cluster_info[i].obb_vertex.at (7).y;
           obj.p7.z = cluster_info[i].obb_vertex.at (7).z;

          obj.p0.x = cluster_info[i].min.x;
          obj.p0.y = cluster_info[i].min.y;
          obj.p0.z = cluster_info[i].min.z;
          obj.p1.x = cluster_info[i].min.x;
          obj.p1.y = cluster_info[i].min.y;
          obj.p1.z = cluster_info[i].max.z;
          obj.p2.x = cluster_info[i].max.x;
          obj.p2.y = cluster_info[i].min.y;
          obj.p2.z = cluster_info[i].max.z;
          obj.p3.x = cluster_info[i].max.x;
          obj.p3.y = cluster_info[i].min.y;
          obj.p3.z = cluster_info[i].min.z;
          obj.p4.x = cluster_info[i].min.x;
          obj.p4.y = cluster_info[i].max.y;
          obj.p4.z = cluster_info[i].min.z;
          obj.p5.x = cluster_info[i].min.x;
          obj.p5.y = cluster_info[i].max.y;
          obj.p5.z = cluster_info[i].max.z;
          obj.p6.x = cluster_info[i].max.x;
          obj.p6.y = cluster_info[i].max.y;
          obj.p6.z = cluster_info[i].max.z;
          obj.p7.x = cluster_info[i].max.x;
          obj.p7.y = cluster_info[i].max.y;
          obj.p7.z = cluster_info[i].min.z;

          objMsg.lidRoiBox.push_back (obj);
        }
      }

      LidRoi_pub.publish (objMsg);
    }

    static void
    send_LidFus (CLUSTER_INFO* cluster_info,
                 int cluster_size)
    {
      static uint32_t seq;

      msgs::LidFus objMsg;
      objMsg.lidHeader.seq = seq++;
      objMsg.lidHeader.stamp = GlobalVariable::ROS_TIMESTAMP;

      for (int i = 0; i < cluster_size; i++)
      {
        if (cluster_info[i].cluster_tag >= 1)
        {
          msgs::LidFusObj obj;
          obj.pixel_x = cluster_info[i].to_2d_PointWL[0].x;
          obj.pixel_y = cluster_info[i].to_2d_PointWL[0].y;
          obj.pixel_width = cluster_info[i].to_2d_PointWL[1].x;
          obj.pixel_height = cluster_info[i].to_2d_PointWL[1].y;
          obj.spatial_center.x = cluster_info[i].center.x;
          obj.spatial_center.y = cluster_info[i].center.y;
          obj.spatial_center.z = cluster_info[i].center.z;

          msgs::PointXYZ pts;
          for (size_t j = 0; j < cluster_info[i].cloud.size (); ++j)
          {
            pts.x = cluster_info[i].cloud.points[j].x;
            pts.y = cluster_info[i].cloud.points[j].y;
            pts.z = cluster_info[i].cloud.points[j].z;
            obj.spatial_cloud.push_back (pts);
          }

          objMsg.lidFusObj.push_back (obj);
        }
      }

      LidFus_pub.publish (objMsg);
    }

    static void
    send_StitchCloud (const PointCloud<PointXYZI>::ConstPtr CloudStitch)
    {
      static uint32_t seq;
      msgs::PointCloud pclMsg;
      pclMsg.lidHeader.stamp = GlobalVariable::ROS_TIMESTAMP;
      pclMsg.lidHeader.seq = seq++;

      for(size_t i=0; i < CloudStitch->size();i++){
           msgs::PointXYZI pclPoint;
           pclPoint.x = CloudStitch->at(i).x;
           pclPoint.y = CloudStitch->at(i).y;
           pclPoint.z = CloudStitch->at(i).z;
           pclPoint.intensity = CloudStitch->at(i).intensity;
           pclMsg.pointCloud.push_back(pclPoint);
         }
      LidStitch_pub.publish(pclMsg);

    }

    static void
    send_ErrorCode (unsigned int error_code)
    {
      static uint32_t seq;

      msgs::ErrorCode objMsg;
      objMsg.header.seq = seq++;
      objMsg.header.stamp = GlobalVariable::ROS_TIMESTAMP;
      objMsg.module = 1;
      objMsg.event = error_code;

      ErrorCode_pub.publish (objMsg);
    }

    static void
    RegisterCallBackLidar (void
                           (*cb1) (const msgs::PointCloud::ConstPtr&),
                           void
                           (*cb2) (const msgs::PointCloud::ConstPtr&),
                           void
                           (*cb3) (const msgs::PointCloud::ConstPtr&),
                           void
                           (*cb4) (const msgs::PointCloud::ConstPtr&),
                           void
                           (*cb5) (const msgs::PointCloud::ConstPtr&))
    {
      ros::NodeHandle n;
      static ros::Subscriber LidFrontTopSub = n.subscribe ("/LidFrontTop", 1, cb1);
      static ros::Subscriber LidFrontLeftSub = n.subscribe ("/LidFrontRight", 1, cb2);
      static ros::Subscriber LidFrontRightSub = n.subscribe ("/LidFrontLeft", 1, cb3);
      static ros::Subscriber LidRearLeftSub = n.subscribe ("/LidRearLeft", 1, cb4);
      static ros::Subscriber LidRearRightSub = n.subscribe ("/LidRearRight", 1, cb5);
    }

};


ros::Publisher RosModuleA::LidObj_pub;
ros::Publisher RosModuleA::LidRoi_pub;
ros::Publisher RosModuleA::LidFus_pub;

ros::Publisher RosModuleA::LidStitch_pub;
ros::Publisher RosModuleA::ErrorCode_pub;
*/


#endif // ROSMODULEA_H
