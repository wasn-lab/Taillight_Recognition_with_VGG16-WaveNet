#include <cmath>
#include <iostream>
#include <vector>
#include <msgs/CamInfo.h>
#include <msgs/DetectedObject.h>
#include <msgs/DetectedObjectArray.h>
#include <opencv2/opencv.hpp>
#include "kalman.h"

class Tracker
{
private:
  const int max_tracking_num = 99;
  const int max_tracking_frames = 2;
  std::vector<Kalman> kalman_vector;
  int tracking_id = 0;
public:
  msgs::DetectedObjectArray tracking(msgs::DetectedObjectArray camera_objects);
};
