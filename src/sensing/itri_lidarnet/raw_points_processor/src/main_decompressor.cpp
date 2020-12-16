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
ros::Publisher g_pub_LidarFrontLeft_DeCompress;
ros::Publisher g_pub_LidarFrontRight_DeCompress;
ros::Publisher g_pub_LidarFrontTop_DeCompress;

//--------------------------- Global Variables
bool g_debug_output = true;
bool pub_decompress = false;

std::mutex L_Lock;
std::mutex R_Lock;
std::mutex T_Lock;

pcl::StopWatch stopWatch_L;
pcl::StopWatch stopWatch_R;
pcl::StopWatch stopWatch_T;

//------------------------------ Callbacks
void cb_LidarFrontLeft(const msgs::CompressedPointCloud& msg)
{
  L_Lock.lock();
  stopWatch_L.reset();

  // -------------------Raw/heartbeat publisher
  // check heartbeat by subcriber data receiver
  // std_msgs::Empty empty_msg;
  // g_pub_LidarFrontLeft_Raw_HeartBeat.publish(empty_msg);

  auto sensor_pc2_ptr = pc2_compressor::decompress(msg.data);

  // publish
  sensor_pc2_ptr->header = msg.header;
  g_pub_LidarFrontLeft_DeCompress.publish(*sensor_pc2_ptr);
  //std::cout << "[L-DeCmpr]: " << stopWatch_L.getTimeSeconds() << 's' << std::endl;
  L_Lock.unlock();
}

void cb_LidarFrontRight(const msgs::CompressedPointCloud& msg)
{
  R_Lock.lock();
  stopWatch_R.reset();

  // -------------------Raw/heartbeat publisher
  // check heartbeat by subcriber data receiver
  // std_msgs::Empty empty_msg;
  // g_pub_LidarFrontLeft_Raw_HeartBeat.publish(empty_msg);

  auto sensor_pc2_ptr = pc2_compressor::decompress(msg.data);

  // publish
  sensor_pc2_ptr->header = msg.header;
  g_pub_LidarFrontRight_DeCompress.publish(*sensor_pc2_ptr);
  //std::cout << "[R-DeCmpr]: " << stopWatch_R.getTimeSeconds() << 's' << std::endl;
  R_Lock.unlock();
}

void cb_LidarFrontTop(const msgs::CompressedPointCloud& msg)
{
  T_Lock.lock();
  stopWatch_T.reset();

  // -------------------Raw/heartbeat publisher
  // check heartbeat by subcriber data receiver
  // std_msgs::Empty empty_msg;
  // g_pub_LidarFrontLeft_Raw_HeartBeat.publish(empty_msg);

  auto sensor_pc2_ptr = pc2_compressor::decompress(msg.data);
  
  // publish
  sensor_pc2_ptr->header = msg.header;
  g_pub_LidarFrontTop_DeCompress.publish(*sensor_pc2_ptr);
  std::cout << "[De-Cmpr]: " << stopWatch_T.getTimeSeconds() << 's' << std::endl;
  T_Lock.unlock();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "raw_decompressor");
  ros::NodeHandle n;

  ros::param::get("/pub_decompress", pub_decompress);

  // subscriber
  ros::Subscriber sub_LidarFrontLeft =
      n.subscribe<const msgs::CompressedPointCloud&>("/LidarFrontLeft/Compressed", 1, cb_LidarFrontLeft);
  ros::Subscriber sub_LidarFrontRight =
      n.subscribe<const msgs::CompressedPointCloud&>("/LidarFrontRight/Compressed", 1, cb_LidarFrontRight);
  ros::Subscriber sub_LidarFrontTop =
      n.subscribe<const msgs::CompressedPointCloud&>("/LidarFrontTop/Compressed", 1, cb_LidarFrontTop);

  if (pub_decompress)
  {
    // publisher
    g_pub_LidarFrontLeft_DeCompress = n.advertise<const sensor_msgs::PointCloud2>("/LidarFrontLeft/Decompressed", 1);
    g_pub_LidarFrontRight_DeCompress = n.advertise<const sensor_msgs::PointCloud2>("/LidarFrontRight/Decompressed", 1);
    g_pub_LidarFrontTop_DeCompress = n.advertise<const sensor_msgs::PointCloud2>("/LidarFrontTop/Decompressed", 1);
  }
  else
  {
    // publisher
    g_pub_LidarFrontLeft_DeCompress = n.advertise<const sensor_msgs::PointCloud2>("/LidarFrontLeft/Raw", 1);
    g_pub_LidarFrontRight_DeCompress = n.advertise<const sensor_msgs::PointCloud2>("/LidarFrontRight/Raw", 1);
    g_pub_LidarFrontTop_DeCompress = n.advertise<const sensor_msgs::PointCloud2>("/LidarFrontTop/Raw", 1);
  }

  ros::AsyncSpinner spinner(3);
  spinner.start();

  ros::waitForShutdown();
  return 0;
}
