#ifndef __TPP_BASE_H__
#define __TPP_BASE_H__

#include <cstdio>    // puts
#include <iostream>  // std::cout
#include <vector>
#include <iomanip>
#include <chrono>     // std::chrono
#include <thread>     // this_thread
#include <cmath>      // std::tan2
#include <cmath>      // round
#include <stdexcept>  // std::runtime_error

#include <mutex>
#include <condition_variable>
#include <csignal>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <ros/callback_queue.h>
#include <tf/tf.h>

// ros msgs
#include <msgs/PointXYZV.h>
#include <std_msgs/Header.h>
#include <msgs/TrackState.h>
#include <msgs/PathPrediction.h>
#include <msgs/TrackInfo.h>
#include <msgs/DetectedObject.h>
#include <msgs/DetectedObjectArray.h>
#include <msgs/VehInfo.h>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/TransformStamped.h>

#include <std_msgs/ColorRGBA.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <opencv2/highgui/highgui.hpp>

#define DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete

#define DISALLOW_ASSIGN(TypeName) TypeName& operator=(const TypeName&) = delete

#define DISALLOW_COPY_AND_ASSIGN(TypeName)                                                                             \
  DISALLOW_COPY(TypeName);                                                                                             \
  DISALLOW_ASSIGN(TypeName)

#define MyPoint32 msgs::PointXYZ
#define Vector3_32 msgs::PointXYZ

#define ENABLE_PROFILING_MODE 0

// virtual input test
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
#define DEBUG_PP_TRAJ 0
#define DEBUG_CONF_E 0
#define DEBUG_TRACKTIME 0
#define DELAY_TIME 1

#define INPUT_ALL_CLASS 1

#define EIGEN3_ROTATION 1

#define PP_VERTICES_VIA_SPEED 1

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
}

#endif  // __TPP_BASE_H__
