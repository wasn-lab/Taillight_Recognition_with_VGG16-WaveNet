// ros
#include <ros/ros.h>
#include <ros/package.h>

//std
#include <string>
#include <iostream>
#include <vector>
#include <mutex>
#include <thread>

// msg
#include "msgs/DetectedObjectArray.h"

// for StopWatch
#include <pcl/common/time.h>

//---------------------------- Publisher
static ros::Publisher g_pub_lidar_detection;

//--------------------------- Global Variables
std::mutex g_car_lock;
std::mutex g_ped_cyc_lock;

ros::Time g_car_msg_rostime;
ros::Time g_ped_cyc_msg_rostime;

msgs::DetectedObjectArray g_msgArr;

pcl::StopWatch g_integrator_stopWatch;

//------------------------------ Callbacks
void cb_LidarDetection_Car(const boost::shared_ptr<const msgs::DetectedObjectArray>& msgArr)
{
  g_car_lock.lock();
  g_integrator_stopWatch.reset();

  g_car_msg_rostime = msgArr->header.stamp;
  for (const auto & object : msgArr->objects)
  {
    g_msgArr.objects.push_back(object);
  }

  g_car_lock.unlock();
}


void cb_LidarDetection_Ped_Cyc(const boost::shared_ptr<const msgs::DetectedObjectArray>& msgArr)
{
  g_ped_cyc_lock.lock();

  g_ped_cyc_msg_rostime = msgArr->header.stamp;

  for (const auto & object : msgArr->objects)
  {
    g_msgArr.objects.push_back(object);
  }

  g_ped_cyc_lock.unlock();
}

void LidarDetection_Publisher(int argc, char** argv)
{
  ros::Rate loop_rate(20);  
  while (ros::ok())
  {
    uint64_t time_diff_ms = 0;
    time_diff_ms = g_car_msg_rostime.toSec()*1000 - g_ped_cyc_msg_rostime.toSec()*1000;
    
    if(time_diff_ms > 50)
    {
      std::cout << "WARNING: Car & Ped_Cyc is Out of Sync! " << time_diff_ms << "ms" << std::endl;
    }
    
    if(!g_msgArr.objects.empty())
    {
      g_msgArr.header.frame_id = "lidar";
      g_msgArr.header.stamp = g_car_msg_rostime;

      g_pub_lidar_detection.publish(g_msgArr);
      g_msgArr.objects = {};


      std::cout << "[Integrator]: " << g_integrator_stopWatch.getTimeSeconds() << 's' << std::endl;

      uint64_t top_to_now_time = (ros::Time::now().toSec() - g_msgArr.header.stamp.toSec()) * 1000;
      if (top_to_now_time < 3600)
      {
        std::cout << "[Latency]: " << top_to_now_time << "ms" << std::endl;
      }
      std::cout << "" << std::endl;
    }
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
  g_pub_lidar_detection = n.advertise<msgs::DetectedObjectArray>("/LidarDetection", 1);

  std::thread Thread_PointPillars_Pub(LidarDetection_Publisher, argc, argv);

  ros::AsyncSpinner spinner(3);
  spinner.start();

  Thread_PointPillars_Pub.join();

  ros::waitForShutdown();
  return 0;
}
