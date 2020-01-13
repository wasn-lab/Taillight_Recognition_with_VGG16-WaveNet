#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "std_msgs/Header.h"
#include <time.h>
#include <pthread.h>
#include <thread>

// ROS msgs
#include <msgs/FreeSpaceResult.h>

/// Opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <nppdefs.h>

ros::Publisher OpenRoadNet_pub;

std::thread mThread;

cv::Mat mat60, mat60_clone;

std::string pkg_path_pb, pkg_path_json;

msgs::FreeSpaceResult OpenRoadNet_output_pub_main;
// sensor_msgs::Image msg_img;
ros::Publisher pub;

int count = 0;
bool display_flag = false;
