/*
   CREATER: ICL U300
   DATE: August, 2019
 */

#ifndef __CAMERA_PARAMS_H__
#define __CAMERA_PARAMS_H__
#include "car_model.h"

#if CAR_MODEL_IS_B1_V2
#include "camera_params_b1_v2.h"
#elif CAR_MODEL_IS_B1_V3
#include "camera_params_b1_v3.h"
#elif CAR_MODEL_IS_C1
#include "camera_params_c1.h"  // TODO: generate and use C1 car param.
#else
#error "Car model is not defined"
#endif

#endif  // __CAMERA_PARAMS_H__
