// ros
#include <ros/ros.h>
#include <ros/package.h>

//std
#include <string>
#include <iostream>
#include <vector>
#include <mutex>

// msg
#include "msgs/DetectedObjectArray.h"

//---------------------------- Publisher
static ros::Publisher g_pub_LidarDetection;

//--------------------------- Global Variables

std::mutex g_Car_Lock;
std::mutex g_Ped_Cyc_Lock;


// pcl::StopWatch g_stopWatch_L;
// pcl::StopWatch g_stopWatch_R;


//------------------------------ Callbacks
void cb_LidarDetection_Car(const msgs::DetectedObjectArray msgArr)
{
  g_Car_Lock.lock();

  std::cout << "Car Arr Size: " << msgArr.objects.size() << std::endl;

  g_Car_Lock.unlock();
}

void cb_LidarDetection_Ped_Cyc(const msgs::DetectedObjectArray msgArr)
{

  g_Ped_Cyc_Lock.lock();

  std::cout << "Ped_Cyc Arr Size: " << msgArr.objects.size() << std::endl;

  g_Ped_Cyc_Lock.unlock();

}

void LidarDetection_Publisher(int argc, char** argv)
{
  ros::Rate loop_rate(20);  
  while (ros::ok())
  {

    /*
      code for add two msg
    */

    loop_rate.sleep();
  }
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "lidar_point_pillars_integrator");
  ros::NodeHandle n;

  // subscriber
  ros::Subscriber sub_LidarDetection_Car =
      n.subscribe<msgs::DetectedObjectArray>("/LidarDetection/Car", 1, cb_LidarDetection_Car);
  ros::Subscriber sub_LidarDetection_Ped_Cyc =
      n.subscribe<msgs::DetectedObjectArray>("/LidarDetection/Ped_Cyc", 1, cb_LidarDetection_Ped_Cyc);

  // publisher
  g_pub_LidarDetection = n.advertise<msgs::DetectedObjectArray>("/LidarDetection", 1);


  ros::AsyncSpinner spinner(3);
  spinner.start();

  ros::waitForShutdown();
  return 0;
}
