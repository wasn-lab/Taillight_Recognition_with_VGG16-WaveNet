#pragma once
#include <ros/ros.h>
#include "msgs/CompressedPointCloud.h"
#include "msgs/CompressedPointCloud2.h"
#include "pc2_compressor.h"

namespace pc2_compressor
{
class PC2DecompressorNode
{
private:
  // member variables
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  ros::NodeHandle node_handle_;

  // functions
  void callback_v1(const msgs::CompressedPointCloudConstPtr& msg);
  void callback_v2(const msgs::CompressedPointCloud2ConstPtr& msg);
  int set_subscriber();
  int set_publisher();

public:
  PC2DecompressorNode();
  ~PC2DecompressorNode();
  void run();
};
};  // namespace pc2_compressor
