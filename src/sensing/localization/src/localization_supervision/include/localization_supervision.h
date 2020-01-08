/*
 *   File: localization_supervision.h
 *   Created on: Aug , 2019
 *   Author: Xu, Bo Chun
 *	 Institute: ITRI ICL U300
 */

#ifndef LOCALIZATION_SUPERVISION_H
#define LOCALIZATION_SUPERVISION_H

#define POSE_INTERVAL_MS 500
#define LIDAR_POINTCLOUD_INTERVAL_MS 500
#define GNSS_INTERVAL_MS 1000

#define POSE_X_RATE_THRESHOLD_KMPERHR 30
#define POSE_Y_RATE_THRESHOLD_KMPERHR 30
#define POSE_Z_RATE_THRESHOLD_KMPERHR 30

#define POSE_ROLL_THRESHOLD_RADPERS 0.0175*90
#define POSE_PITCH_THRESHOLD_RADPERS 0.0175*90
#define POSE_YAW_THRESHOLD_RADPERS 0.0175*90

#include <iostream>
#include <vector>
#include <queue>
#include <boost/thread/recursive_mutex.hpp>
#include <chrono>

#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <visualization_msgs/MarkerArray.h>
#include <localization_supervision/pose.h>
struct pose
{
        double x;
        double y;
        double z;
        double roll;
        double pitch;
        double yaw;
};

static pose current_pose, pre_pose, pose_rate;
static geometry_msgs::PoseStamped current_pose_quat, pre_pose_quat;
geometry_msgs::PoseWithCovarianceStamped initialpose_;
geometry_msgs::PoseWithCovarianceStamped prev_initialpose_;
geometry_msgs::PoseWithCovarianceStamped gnss_pose_;
geometry_msgs::PoseWithCovarianceStamped prev_gnss_pose_;

static ros::Publisher pose_rate_pub;
static ros::Publisher state_pub;

boost::recursive_mutex ms_checker;


#endif //LOCALIZATION_STATE_SUPERVISION_H
