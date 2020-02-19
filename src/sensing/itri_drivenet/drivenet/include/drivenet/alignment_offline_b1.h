#ifndef ALIGNMENT_OFFLINE_H_
#define ALIGNMENT_OFFLINE_H_

// ROS message
#include "camera_params.h"  // include camera topic name
#include <msgs/BoxPoint.h>
#include <msgs/PointXYZ.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


class AlignmentOff
{
private:
  int carId = 1;

public:
  void init(int carId);

};

#endif /*ALIGNMENT_OFFLINE_H_*/
