#ifndef ROSMODULE_H
#define ROSMODULE_H


//--------------------------------------------------------------------------------------------------------------Traffic
#include "ros/ros.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/LidLLA.h"
#include "msgs/TaichungVehInfo.h"
#include "std_msgs/String.h"

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/tf.h>

class RosModuleTraffic
{
  public:

    static void
    Initial (int argc,
             char ** argv)
    {
      ros::init (argc, argv, "adv_to_server");

    }

    static void
    RegisterCallBack (void
                      (*cb1) (const msgs::DetectedObjectArray&),
                      void
                      (*cb2) (const msgs::LidLLA&),
                      void
                      (*cb3) (const msgs::TaichungVehInfo&),
                      void
                      (*cb4) (const geometry_msgs::PoseStamped::ConstPtr&),
                      void
                      (*cb5) (const std_msgs::String::ConstPtr&))
    {
      ros::NodeHandle n;
      static ros::Subscriber detObj = n.subscribe ("LidarDetection", 1, cb1);
      static ros::Subscriber gps = n.subscribe ("lidar_lla", 1, cb2);
      static ros::Subscriber vehInfo = n.subscribe ("taichung_veh_info", 1, cb3);
      static ros::Subscriber gnss2local_sub = n.subscribe("gnss2local_data", 1, cb4);
      static ros::Subscriber fps = n.subscribe("/GUI/topic_fps_out", 1, cb5);
    }

};

#endif // ROSMODULE_H
