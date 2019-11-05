//
// File: Localization_UKF_data.cpp
//
// Code generated for Simulink model 'Localization_UKF'.
//
// Model version                  : 1.1
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Fri Oct 25 18:26:59 2019
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
// Code generation objectives:
//    1. Execution efficiency
//    2. RAM efficiency
// Validation result: Not run
//
#include "Localization_UKF.h"

// Constant parameters (default storage)
const ConstP rtConstP = {
  // Expression: [eye(5)*0.001]
  //  Referenced by: '<S1>/Unit Delay33'

  { 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001 },

  // Expression: diag([0.00025,0.00025,0.0000001,1,0.0001])
  //  Referenced by: '<S1>/Unit Delay37'

  { 0.00025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00025, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0E-7,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001 },

  // Expression: diag([0.1,0.1,0.1,0.0001,0.0001])
  //  Referenced by: '<S1>/Unit Delay36'

  { 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001 }
};

//
// File trailer for generated code.
//
// [EOF]
//
