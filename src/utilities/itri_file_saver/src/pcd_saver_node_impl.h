#ifndef __IMAGE_SAVER_NODE_IMPL_H__
#define __IMAGE_SAVER_NODE_IMPL_H__

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

class PCDSaverNodeImpl
{
private:
  // member variables
  ros::Subscriber subscriber_;
  ros::NodeHandle node_handle_;

  // functions
  void pcd_callback(const sensor_msgs::PointCloud2ConstPtr& in_pcd_message);
  void subscribe();
  void save(const sensor_msgs::PointCloud2ConstPtr& in_pcd_message, int sec, int nsec);

public:
  PCDSaverNodeImpl();
  ~PCDSaverNodeImpl();
  void run();
};

#endif  // __IMAGE_SAVER_NODE_IMPL_H__
