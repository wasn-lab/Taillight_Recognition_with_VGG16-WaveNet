#pragma once
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

namespace pcd_to_pc2
{
class PCDToPc2Node
{
private:
  ros::NodeHandle node_handle_;
  sensor_msgs::PointCloud2 pc2_;

public:
  PCDToPc2Node();
  ~PCDToPc2Node();
  void load_pcd();
  void run();
};
}; // namespace pcd_to_pc2
