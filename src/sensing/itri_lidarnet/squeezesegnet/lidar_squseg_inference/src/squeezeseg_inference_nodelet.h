#ifndef SQSG_NODELET_H_
#define SQSG_NODELET_H_

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <iostream>
#include <cmath>
#include <mutex>
#include <omp.h>
#include <thread>

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <eigen3/Eigen/Dense>

#include "ssn_config.h"
#include "CuboidFilter.h"



namespace ssn_nodelet
{
  extern bool debug_output;

  extern ros::Publisher nn_pub;
  extern ros::Subscriber LidarAllSub;

  extern string data_set;
  extern char ViewType;
  extern int pub_type;
  extern bool hybrid_detect;

  extern string GET_data_set;
  extern string GET_ViewType;
  extern string GET_pub_type;
  extern string GET_hybrid_detect;

  extern vector<TF_inference> SSN_all;

  class LidarsNodelet : public nodelet::Nodelet
  {
    public:
      virtual void onInit ();
      static void callback_LidarAll (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &msg);

    private:
      bool to_bool (std::string const &s);
      
  };

}

#endif /* SQSG_NODELET_H_ */

