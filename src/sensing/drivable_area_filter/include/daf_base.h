#ifndef __DAF_BASE_H__
#define __DAF_BASE_H__

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
#include <tf/tf.h>

// ros msgs
#include <std_msgs/Header.h>
#include <msgs/DetectedObject.h>
#include <msgs/DetectedObjectArray.h>

#include "detected_object_class_id.h"

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <opencv2/highgui/highgui.hpp>

#define ENABLE_PROFILING_MODE 0

#define USE_GLOG 0
#if USE_GLOG == 1
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

#endif  // __DAF_BASE_H__
