#ifndef ROSMODULE_H
#define ROSMODULE_H


//--------------------------------------------------------------------------------------------------------------Traffic
#include "ros/ros.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/LidLLA.h"
#include "msgs/VehInfo.h"
#include "std_msgs/String.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Int32.h"
#include "msgs/Flag_Info.h"
#include "msgs/StopInfoArray.h"
#include "msgs/StopInfo.h"
#include "msgs/RouteInfo.h"
#include "msgs/BackendInfo.h"
#include "sensor_msgs/Imu.h"
#include "msgs/Spat.h"

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/tf.h>
#include <chrono>
#include <thread>



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
                      (*cb3) (const msgs::VehInfo&),
                      void
                      (*cb4) (const geometry_msgs::PoseStamped::ConstPtr&),
                      void
                      (*cb5) (const std_msgs::String::ConstPtr&),
                      void
                      (*cb6) (const msgs::Flag_Info::ConstPtr&),
                      void
                      (*cb7) (const std_msgs::String::ConstPtr&),
                      void
                      (*cb8) (const msgs::Flag_Info::ConstPtr&),
                      void
                      (*cb9) (const std_msgs::Int32::ConstPtr&),
                      void
                      (*cb10) (const sensor_msgs::Imu::ConstPtr&),
                      void
                      (*cb11) (const std_msgs::String::ConstPtr&),
                      void
                      (*cb12) (const msgs::BackendInfo::ConstPtr&))
    {
      ros::NodeHandle n;
      static ros::Subscriber detObj = n.subscribe ("LidarDetection", 1, cb1);
      static ros::Subscriber gps = n.subscribe ("lidar_lla", 1, cb2);
      static ros::Subscriber vehInfo = n.subscribe ("veh_info", 1, cb3);
      static ros::Subscriber gnss2local_sub = n.subscribe("gnss2local_data", 1, cb4);
      static ros::Subscriber fps = n.subscribe("/GUI/topic_fps_out", 1, cb5);
      static ros::Subscriber busStopInfo = n.subscribe("/BusStop/Info", 1, cb6);
      static ros::Subscriber reverse = n.subscribe("/mileage/relative_mileage", 1, cb7);
      static ros::Subscriber next_stop = n.subscribe("/NextStop/Info", 1, cb8);
      static ros::Subscriber round = n.subscribe("/BusStop/Round", 1, cb9);
      static ros::Subscriber imu = n.subscribe("imu_data_rad", 1, cb10);
      //checker big buffer for multi event at the same time.
      static ros::Subscriber checker = n.subscribe("/ADV_op/event_json", 1000, cb11);
      static ros::Subscriber backendInfo = n.subscribe("Backend/Info", 1, cb12);
    }

    static void
    publishTraffic(std::string topic, msgs::Spat input)
    {
      std::cout << "publishTraffic topic " << topic <<  std::endl;
      ros::NodeHandle n;
      static ros::Publisher traffic_pub = n.advertise<msgs::Spat>(topic, 1000);
      traffic_pub.publish(input);
    }

    static void
    publishServerStatus(std::string topic, bool input)
    {
      std::cout << "publishServerStatus topic " << topic << " , input " << input << std::endl;
      ros::NodeHandle n;
      static ros::Publisher server_status_pub = n.advertise<std_msgs::Bool>(topic, 1000);
      std_msgs::Bool msg;
      msg.data = input;
      server_status_pub.publish(msg);
    }

    static void
    publishReserve(std::string topic, msgs::StopInfoArray msg)
    {
      std::cout << "publishReserve topic " << topic  << std::endl;
      ros::NodeHandle n;
      static ros::Publisher reserve_status_pub = n.advertise<msgs::StopInfoArray>(topic, 1000);
      short count = 0;
      while (count < 30)
      { 
        count++;
        int numOfSub = reserve_status_pub.getNumSubscribers() ;
        //std::cout << "numOfSub = " << numOfSub << std::endl;
        if(numOfSub > 0) 
        {
          std::chrono::duration<int, std::milli> timespan(100);
          std::this_thread::sleep_for(timespan);
          reserve_status_pub.publish(msg);
          return;
        }
      } 
    }

    static void
    publishRoute(std::string topic, msgs::RouteInfo msg)
    {
      std::cout << "publishReserve topic " << topic  << std::endl;
      ros::NodeHandle n;
      static ros::Publisher route_pub = n.advertise<msgs::RouteInfo>(topic, 1000);
      short count = 0;
      while (count < 30)
      { 
        count++;
        int numOfSub = route_pub.getNumSubscribers() ;
        //std::cout << "numOfSub = " << numOfSub << std::endl;
        if(numOfSub > 0) 
        {
          std::chrono::duration<int, std::milli> timespan(100);
          std::this_thread::sleep_for(timespan);
          route_pub.publish(msg);
          return;
        }
      } 
    }
};

#endif // ROSMODULE_H
