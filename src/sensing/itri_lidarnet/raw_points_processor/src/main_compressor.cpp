// ros
#include <ros/ros.h>
#include <ros/package.h>

//std
#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include <mutex>

// lib
#include "pc2_compressor.h"
#include <pcl/common/time.h>
//#include "car_model.h"

// msg
#include "msgs/CompressedPointCloud.h"
#include <std_msgs/Empty.h>


//---------------------------- Publisher
static ros::Publisher g_pub_LidarFrontLeft_Compress;
static ros::Publisher g_pub_LidarFrontRight_Compress;
static ros::Publisher g_pub_LidarFrontTop_Compress;

//--------------------------- Global Variables
bool g_debug_output = true;

std::mutex g_L_Lock;
std::mutex g_R_Lock;
std::mutex g_T_Lock;

pcl::StopWatch g_stopWatch_L;
pcl::StopWatch g_stopWatch_R;
pcl::StopWatch g_stopWatch_T;
pcl::StopWatch g_stopWatch_Compressor;

//------------------------------ Callbacks
void cb_LidarFrontLeft(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  g_L_Lock.lock();

  // -------------------Raw/heartbeat publisher
  // check heartbeat by subcriber data receiver
  // std_msgs::Empty empty_msg;
  // g_pub_LidarFrontLeft_Raw_HeartBeat.publish(empty_msg);

  if (input_cloud->width * input_cloud->height > 100)
  {
    g_stopWatch_L.reset();

    // check data from hardware
    if (g_debug_output && (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) < 3600)
    {
      uint64_t diff_time = (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) * 1000;
      std::cout << "[Left->Cmpr]: " << diff_time << "ms" << std::endl;
    }

    //-------------------------- compress 
    msgs::CompressedPointCloud compressed_pointcloud;
    compressed_pointcloud.data = pc2_compressor::compress(input_cloud);

    // publish
    compressed_pointcloud.header = input_cloud->header;
    g_pub_LidarFrontLeft_Compress.publish(compressed_pointcloud);
  }
  g_L_Lock.unlock();
}

void cb_LidarFrontRight(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  g_R_Lock.lock();

  if (input_cloud->width * input_cloud->height > 100)
  {
    g_stopWatch_R.reset();

    // check data from hardware
    if (g_debug_output && (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) < 3600)
    {
      uint64_t diff_time = (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) * 1000;
      std::cout << "[Right->Cmpr]: " << diff_time << "ms" << std::endl;
    }

    //-------------------------- compress 
    msgs::CompressedPointCloud compressed_pointcloud;
    compressed_pointcloud.data = pc2_compressor::compress(input_cloud);

    // publish
    compressed_pointcloud.header = input_cloud->header;
    g_pub_LidarFrontRight_Compress.publish(compressed_pointcloud);

  }
  g_R_Lock.unlock();
}

void cb_LidarFrontTop(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  g_T_Lock.lock();

  if (input_cloud->width * input_cloud->height > 100)
  {
    g_stopWatch_T.reset();

    // check data from hardware
    if (g_debug_output && (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) < 3600)
    {
      uint64_t diff_time = (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) * 1000;
      std::cout << "[Top->Cmpr]: " << diff_time << "ms" << std::endl;
    }

    //-------------------------- compress 
    msgs::CompressedPointCloud compressed_pointcloud;
    compressed_pointcloud.data = pc2_compressor::compress(input_cloud);

    // publish
    compressed_pointcloud.header = input_cloud->header;
    g_pub_LidarFrontTop_Compress.publish(compressed_pointcloud);

    std::cout << "[T-Cmpr]: " << g_stopWatch_T.getTimeSeconds() << 's' << std::endl;
  }
  g_T_Lock.unlock();
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "raw_compressor");
  ros::NodeHandle n;

  // subscriber
  ros::Subscriber sub_LidarFrontLeft =
      n.subscribe<sensor_msgs::PointCloud2>("/LidarFrontLeft/Raw", 1, cb_LidarFrontLeft);
  ros::Subscriber sub_LidarFrontRight =
      n.subscribe<sensor_msgs::PointCloud2>("/LidarFrontRight/Raw", 1, cb_LidarFrontRight);
  ros::Subscriber sub_LidarFrontTop =
      n.subscribe<sensor_msgs::PointCloud2>("/LidarFrontTop/Raw", 1, cb_LidarFrontTop);

  // publisher - compressed
  g_pub_LidarFrontLeft_Compress = n.advertise<msgs::CompressedPointCloud>("/LidarFrontLeft/Compressed", 1);
  g_pub_LidarFrontRight_Compress = n.advertise<msgs::CompressedPointCloud>("/LidarFrontRight/Compressed", 1);
  g_pub_LidarFrontTop_Compress = n.advertise<msgs::CompressedPointCloud>("/LidarFrontTop/Compressed", 1);

  ros::AsyncSpinner spinner(3);
  spinner.start();

  ros::waitForShutdown();
  return 0;
}
