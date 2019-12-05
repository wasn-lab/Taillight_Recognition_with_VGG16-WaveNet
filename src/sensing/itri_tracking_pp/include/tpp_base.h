#ifndef __TPP_BASE_H__
#define __TPP_BASE_H__

#include <stdio.h>   // puts
#include <iostream>  // std::cout
#include <vector>
#include <iomanip>
#include <chrono>     // std::chrono
#include <thread>     // this_thread
#include <cmath>      // std::tan2
#include <math.h>     // round
#include <stdexcept>  // std::runtime_error

#include <mutex>
#include <condition_variable>
#include <signal.h>

#include <Eigen/Eigenvalues>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <ros/callback_queue.h>

// ros msgs
#include <msgs/PointXYZV.h>
#include <std_msgs/Header.h>
#include <msgs/TrackState.h>
#include <msgs/PathPrediction.h>
#include <msgs/TrackInfo.h>
#include <msgs/DetectedObject.h>
#include <msgs/DetectedObjectArray.h>
#include <msgs/VehInfo.h>
#include <msgs/LocalizationToVeh.h>

#define TTC_TEST 0
#if TTC_TEST
#include <tf/tf.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float64.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#endif

#include <std_msgs/ColorRGBA.h>
#include <sensor_msgs/Imu.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <opencv2/highgui/highgui.hpp>

#define DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete

#define DISALLOW_ASSIGN(TypeName) TypeName& operator=(const TypeName&) = delete

#define DISALLOW_COPY_AND_ASSIGN(TypeName)                                                                             \
  DISALLOW_COPY(TypeName);                                                                                             \
  DISALLOW_ASSIGN(TypeName)

#define Point32 msgs::PointXYZ
#define Vector3_32 msgs::PointXYZ

#define FPS 0
#define ENABLE_PROFILING_MODE 0
#define FPS_EXTRAPOLATION 0

// virtual input test
#define VIRTUAL_INPUT 0
#define SAME_OBJ_MARKER_HEADER 0
#define SAVE_OUTPUT_TXT 0

#define SPEEDUP_KALMAN_VEL_EST 1  // speed up kalman velocity estimation

// debug
#define DEBUG 0
#define DEBUG_CALLBACK 0
#define DEBUG_COMPACT 0
#define DEBUG_DATA_IN 0
#define DEBUG_VELOCITY 0
#define DEBUG_HUNGARIAN_DIST 0
#define DEBUG_PP 0
#define DEBUG_CONF_E 0
#define DEBUG_TRACKTIME 0
#define DELAY_TIME 1

#define FILL_CONVEX_HULL 1

#define USE_RADAR_REL_SPEED 0  // use radar's relative speed w.r.t. ego-vehicle
#if USE_RADAR_REL_SPEED
#define USE_RADAR_ABS_SPEED 0  // compute absolute speed from ege speed, ego heading, and radar's relative speed
#endif

#define REMOVE_IMPULSE_NOISE 0
#define NOT_OUTPUT_SHORT_TERM_TRACK_LOST_BBOX 0
// when a tracked bbox shrinks severely in a sudden, replace it with the previous (larger) bbox
#define PREVENT_SHRINK_BBOX 0

#define O_FIX std::setiosflags(std::ios::fixed)
#define O_P std::setprecision(8)

#define USE_GLOG 0
#if USE_GLOG
#include "glog/logging.h"
#define LOG_INFO LOG(INFO)
#define LOG_WARNING LOG(WARNING)
#define LOG_ERROR LOG(ERROR)
#define LOG_FATAL LOG(FATAL)
#else
#define LOG_INFO std::cout
#define LOG_WARNING std::cout
#define LOG_ERROR std::cout
#define LOG_FATAL std::cout
#endif

namespace tpp
{
struct PoseRPY32
{
  float x;
  float y;
  float z;

  float roll;
  float pitch;
  float yaw;
};

struct MarkerConfig
{
  ros::Publisher pub_bbox;
  ros::Publisher pub_polygon;
  ros::Publisher pub_pp;
  ros::Publisher pub_vel;

  ros::Publisher pub_id;
  ros::Publisher pub_speed;
  ros::Publisher pub_delay;

  double lifetime_sec = 0.1;
  double module_pubtime_sec = 0.;

  bool show_classid = 0;
  bool show_tracktime = 0;
  bool show_source = 0;
  bool show_distance = 0;
  bool show_absspeed = 0;  // km/h
  unsigned int show_pp = 0;

  std_msgs::ColorRGBA color;
  std_msgs::ColorRGBA color_lidar_tpp;
  std_msgs::ColorRGBA color_radar_tpp;
  std_msgs::ColorRGBA color_camera_tpp;
  std_msgs::ColorRGBA color_fusion_tpp;
};
}

#endif  // __TPP_BASE_H__
