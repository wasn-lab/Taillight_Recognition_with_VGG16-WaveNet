#ifndef ALIGNMENT_OFFLINE_H_
#define ALIGNMENT_OFFLINE_H_

// ROS message
#include "ros/ros.h"
#include <std_msgs/String.h>
#include <std_msgs/Header.h>
#include "sensor_msgs/PointCloud2.h"

#include "camera_params.h"  // include camera topic name
#include <msgs/BoxPoint.h>
#include <msgs/PointXYZ.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "projection/projector2.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>


class AlignmentOff
{
private:
  int carId = 1;
  Projector2 pj;

public:
  void init(int carId);
  vector<int> run();  
  vector<int> out;
  
};

#endif /*ALIGNMENT_OFFLINE_H_*/
