#ifndef CAMGRABBER_HPP
#define CAMGRABBER_HPP

#include <iostream>
#include "ros/ros.h"
#include "std_msgs/Header.h"
#include <string>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <mutex>
#include <math.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sensor_msgs/image_encodings.h>
#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

extern void sync_callbackThreads();

const static int IMG_WIDTH = 608;
const static int IMG_HEIGHT = 384;
const static int IMG_SIZE = IMG_WIDTH*IMG_HEIGHT*3;

extern cv::Mat cam10_inputImg;
extern cv::Mat cam10_cloneImg;
extern cv::Mat cam11_inputImg;
extern cv::Mat cam11_cloneImg;
extern cv::Mat cam12_inputImg;
extern cv::Mat cam12_cloneImg;


void camera1_0(const sensor_msgs::ImageConstPtr& msg);
void camera1_1(const sensor_msgs::ImageConstPtr& msg);
void camera1_2(const sensor_msgs::ImageConstPtr& msg);


#endif
