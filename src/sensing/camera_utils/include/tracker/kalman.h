#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <stdio.h>
class Kalman
{
private:
  // 8 state variables (x, y, width, height, vx, vy, vw, vh), 4 measurements (x, y, width, height)
  int stateNum=8;                                  
  int measureNum=4;                  
  float delta_t=1;  
  cv::KalmanFilter KF;
public: 
  int tracking_count = 0;
  int id;
  Kalman(int id);
  cv::Rect last_detection;
  bool initialized = false;
  bool isUpdated = false;
  void init(cv::Rect rect);
  void predict();
  cv::Rect update(cv::Rect rect, bool hasDetected);
};
