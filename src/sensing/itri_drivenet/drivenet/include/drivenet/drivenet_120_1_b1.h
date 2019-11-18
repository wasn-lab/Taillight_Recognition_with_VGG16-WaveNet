#ifndef DRIVENET_120_1_B1_H_
#define DRIVENET_120_1_B1_H_

#include "camera_params.h" // include camera topic name
#include "drivenet/trt_yolo_interface.h"
#include "drivenet/distance_estimation_b1.h"
#include "drivenet/boundary_util.h"
#include "drivenet/object_label_util.h"
#include "drivenet/image_preprocessing.h"

#if CAR_MODEL_IS_B1
  // TODO: fill in the correct camera id.
  const std::vector<int> cam_ids_{ 
    camera::id::top_right_front_120,
    camera::id::top_right_rear_120,
    camera::id::top_left_front_120,
    camera::id::top_left_rear_120,
  };
#else
#error "car model is not well defined"
#endif

/// library
DistanceEstimation distEst;
Yolo_app yoloApp;

/// launch param
int car_id = 1;
bool standard_FPS = 0;
bool display_flag = 0;
bool input_resize = 1; //grabber input mode 0: 1920x1208, 1:608x384 yolo format
bool imgResult_publish = 0; 

#endif /*DRIVENET_120_1_B1_H_*/