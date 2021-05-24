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


static ros::Publisher traffic_pub;
static ros::Publisher backend_pub;
static ros::Publisher occ_pub;

class RosModuleTraffic
{
  public:
    
    static void
    Initial (int argc,
             char ** argv)
    {
      ros::init (argc, argv, "adv_to_server");
      ros::NodeHandle n;
      traffic_pub = n.advertise<msgs::Spat>("/traffic", 1000);
      backend_pub = n.advertise<std_msgs::Bool>("/backend_sender/status", 1000);
      occ_pub = n.advertise<std_msgs::Bool>("/occ_sender/status", 1000);
    }

    static std::string getPlate(){
        ros::NodeHandle n;
        std::string plate;
        if(n.getParam("/south_bridge/license_plate_number", plate)){
            return plate;
        }else{
            return "DEFAULT-ITRI-ADV";
        }
    }

    static std::string getVid(){
        ros::NodeHandle n;
        std::string vid;
        if(n.getParam("/south_bridge/vid", vid)){
            return vid;
        }else{
            return "Default-vid";
        }
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
                      (*cb11) (const std_msgs::Bool::ConstPtr&),
                      void
                      (*cb12) (const msgs::BackendInfo::ConstPtr&),
                      void
                      (*cb13) (const std_msgs::String::ConstPtr&),
                      void
                      (*cb14) (const msgs::DetectedObjectArray&),
                      void
                      (*cb15) (const std_msgs::String::ConstPtr&),
                      void
                      (*cb16) (const msgs::Flag_Info::ConstPtr&),
                      void
                      (*cb17) (const msgs::Flag_Info::ConstPtr&),
                      bool isNewMap)
    {
      ros::NodeHandle n;
      static ros::Subscriber detObj = n.subscribe ("LidarDetection", 1, cb1);
     
      static ros::Subscriber vehInfo = n.subscribe ("veh_info", 1, cb3);

      if(isNewMap){
        std::cout << "===============================subscribe for new map" << std::endl;
        static ros::Subscriber gps = n.subscribe ("lidar_lla_wgs84", 1, cb2);
        static ros::Subscriber gnss2local_sub = n.subscribe("gnss_data", 1, cb4);
      }else{
        std::cout << "===============================subscribe for old map" << std::endl;
        static ros::Subscriber gps = n.subscribe ("lidar_lla", 1, cb2);
        static ros::Subscriber gnss2local_sub = n.subscribe("gnss2local_data", 1, cb4);
      }
      
      static ros::Subscriber fps = n.subscribe("/GUI/topic_fps_out", 1, cb5);
      static ros::Subscriber busStopInfo = n.subscribe("/BusStop/Info", 1, cb6);
      static ros::Subscriber reverse = n.subscribe("/mileage/relative_mileage", 1, cb7);
      static ros::Subscriber next_stop = n.subscribe("/NextStop/Info", 1, cb8);
      static ros::Subscriber round = n.subscribe("/BusStop/Round", 1, cb9);
      static ros::Subscriber imu = n.subscribe("imu_data_rad", 1, cb10);
      //checker big buffer for multi event at the same time.
      //get event from fail_safe
      static ros::Subscriber checker = n.subscribe("/ADV_op/sys_ready", 1000, cb11);
      static ros::Subscriber backendInfo = n.subscribe("Backend/Info", 1, cb12);
      static ros::Subscriber sensor_status = n.subscribe("/vehicle/report/itri/sensor_status", 1, cb13);
      static ros::Subscriber tracking = n.subscribe("/Tracking3D/xyz2lla", 100, cb14);
      static ros::Subscriber fail_safe = n.subscribe("/vehicle/report/itri/fail_safe_status", 1, cb15);
      static ros::Subscriber flag04 = n.subscribe("/Flag_Info04", 1, cb16);
      static ros::Subscriber flag02 = n.subscribe("/Flag_Info02", 1, cb17);
    }

    static void
    publishTraffic(std::string topic, msgs::Spat input)
    {
      std::cout << "publishTraffic topic " << topic <<  std::endl;
      traffic_pub.publish(input);
    }

    static void pubBackendState(bool input)
    {
        std_msgs::Bool result;
        result.data = input;
        backend_pub.publish(result);
    }

    static void pubOCCState(bool input)
    {
        std_msgs::Bool result;
        result.data = input;
        occ_pub.publish(result);
    }

    static void
    publishServerStatus(std::string topic, bool input)
    {
      //std::cout << "publishServerStatus topic " << topic << " , input " << input << std::endl;
      ros::NodeHandle n;
      static ros::Publisher server_status_pub = n.advertise<std_msgs::Bool>(topic, 1000);
      std_msgs::Bool msg;
      msg.data = input;
      server_status_pub.publish(msg);
    }

    static void
    publishReserve(std::string topic, msgs::StopInfoArray msg)
    {
      //std::cout << "publishReserve topic " << topic  << std::endl;
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
      //std::cout << "publishReserve topic " << topic  << std::endl;
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
