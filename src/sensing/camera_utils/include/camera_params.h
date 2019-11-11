/*
   CREATER: ICL U300
   DATE: August, 2019
 */

#ifndef __CAMERA_PARAMS_H__
#define __CAMERA_PARAMS_H__
#include "car_model.h"

#if CAR_MODEL_IS_B1
#include "camera_params_b1.h"
#elif CAR_MODEL_IS_HINO
#include "camera_params_hino.h"
#elif CAR_MODEL_IS_C
#error "Camera parameters for car C is not defined yet."
#elif CAR_MODEL_IS_A
#error "Camera parameters for car A is not defined yet."
#else
#error "Car model is not defined"
#endif

#endif  // __CAMERA_PARAMS_H__
