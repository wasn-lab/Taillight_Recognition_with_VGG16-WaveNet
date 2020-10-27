#ifndef __TRACK2D_BASE_H__
#define __TRACK2D_BASE_H__

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

#include <ros/ros.h>
#include <ros/spinner.h>
#include <ros/callback_queue.h>

#include <msgs/PointXYZV.h>
#include <std_msgs/Header.h>
#include <msgs/TrackState.h>
#include <msgs/TrackInfo.h>
#include <msgs/DetectedObject.h>
#include <msgs/DetectedObjectArray.h>

#include <opencv2/highgui/highgui.hpp>

#define DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete

#define DISALLOW_ASSIGN(TypeName) TypeName& operator=(const TypeName&) = delete

#define DISALLOW_COPY_AND_ASSIGN(TypeName)                                                                             \
  DISALLOW_COPY(TypeName);                                                                                             \
  DISALLOW_ASSIGN(TypeName)

#define ENABLE_PROFILING_MODE 0
#define DEBUG 0
#define DEBUG_TIME 0
#define DEBUG_TRACKTIME 0
#define DEBUG_HUNGARIAN 0

#define INPUT_ALL_CLASS 1
#define SPEEDUP_KALMAN_VEL_EST 1  // speed up kalman velocity estimation

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

#endif  // __TRACK2D_BASE_H__
