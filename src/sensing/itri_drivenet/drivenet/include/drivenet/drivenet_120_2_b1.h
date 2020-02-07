#ifndef DRIVENET_120_2_B1_H_
#define DRIVENET_120_2_B1_H_

#include "camera_params.h"  // include camera topic name
#include "drivenet/trt_yolo_interface.h"
#include "drivenet/distance_estimation_b1.h"
#include "drivenet/boundary_util.h"
#include "drivenet/object_label_util.h"
#include "drivenet/image_preprocessing.h"
#include "drivenet/image_exception_handling.h"
#include "drivenet/math_util.h"
#include "drivenet/type_conversion.h"
// #include "costmap_generator.h"

#if CAR_MODEL_IS_B1
const std::vector<int> cam_ids_{ camera::id::top_front_120, camera::id::top_rear_120 };

#else
#error "car model is not well defined"
#endif

/// library
DistanceEstimation distEst;
Yolo_app yoloApp;
// CosmapGenerator cosmapGener;

/// launch param
int car_id = 1;
bool standard_FPS = 0;
bool display_flag = 0;
bool input_resize = 1;  // grabber input mode 0: 1920x1208, 1:608x384 yolo format
bool imgResult_publish = 1;

#endif /*DRIVENET_120_2_B1_H_*/
