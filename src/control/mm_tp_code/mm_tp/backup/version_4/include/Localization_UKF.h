//
// File: Localization_UKF.h
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
#ifndef RTW_HEADER_Localization_UKF_h_
#define RTW_HEADER_Localization_UKF_h_
#include <cmath>
#include <string.h>
#ifndef Localization_UKF_COMMON_INCLUDES_
# define Localization_UKF_COMMON_INCLUDES_
#include "rtwtypes.h"
#endif                                 // Localization_UKF_COMMON_INCLUDES_

// Macros for accessing real-time model data structure
#ifndef rtmGetErrorStatus
# define rtmGetErrorStatus(rtm)        ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
# define rtmSetErrorStatus(rtm, val)   ((rtm)->errorStatus = (val))
#endif

// Forward declaration for rtModel
typedef struct tag_RTM RT_MODEL;

// Block signals and states (default storage) for system '<Root>'
typedef struct {
  real_T UnitDelay34_DSTATE[5];        // '<S1>/Unit Delay34'
  real_T UnitDelay33_DSTATE[25];       // '<S1>/Unit Delay33'
  real_T UnitDelay1_DSTATE[5];         // '<S1>/Unit Delay1'
  real_T UnitDelay35_DSTATE[4];        // '<S1>/Unit Delay35'
  real_T UnitDelay37_DSTATE[25];       // '<S1>/Unit Delay37'
  real_T UnitDelay36_DSTATE[25];       // '<S1>/Unit Delay36'
  real_T UnitDelay38_DSTATE;           // '<S1>/Unit Delay38'
} DW;

// Constant parameters (default storage)
typedef struct {
  // Expression: [eye(5)*0.001]
  //  Referenced by: '<S1>/Unit Delay33'

  real_T UnitDelay33_InitialCondition[25];

  // Expression: diag([0.00025,0.00025,0.0000001,1,0.0001])
  //  Referenced by: '<S1>/Unit Delay37'

  real_T UnitDelay37_InitialCondition[25];

  // Expression: diag([0.1,0.1,0.1,0.0001,0.0001])
  //  Referenced by: '<S1>/Unit Delay36'

  real_T UnitDelay36_InitialCondition[25];
} ConstP;

// External inputs (root inport signals with default storage)
typedef struct {
  real_T angular_vz;                   // '<Root>/angular_vz'
  real_T Speed_mps;                    // '<Root>/Speed_mps'
  real_T SLAM_x;                       // '<Root>/SLAM_x'
  real_T SLAM_y;                       // '<Root>/SLAM_y'
  real_T SLAM_heading;                 // '<Root>/SLAM_heading'
  real_T SLAM_counter;                 // '<Root>/SLAM_counter'
  real_T SLAM_fs;                      // '<Root>/SLAM_fs'
  real_T SLAM_fault;                   // '<Root>/SLAM_fault'
  real_T dt;                           // '<Root>/dt'
} ExtU;

// External outputs (root outports fed by signals with default storage)
typedef struct {
  real_T X_UKF_SLAM[5];                // '<Root>/X_UKF_SLAM'
} ExtY;

// Real-time Model Data Structure
struct tag_RTM {
  const char_T * volatile errorStatus;
};

// Constant parameters (default storage)
extern const ConstP rtConstP;

// Class declaration for model Localization_UKF
class Localization_UKFModelClass {
  // public data and function members
 public:
  // External inputs
  ExtU rtU;

  // External outputs
  ExtY rtY;

  // model initialize function
  void initialize();

  // model step function
  void step();

  // Constructor
  Localization_UKFModelClass();

  // Destructor
  ~Localization_UKFModelClass();

  // Real-Time Model get method
  RT_MODEL * getRTM();

  // private data and function members
 private:
  // Block signals and states
  DW rtDW;

  // Real-Time Model
  RT_MODEL rtM;

  // private member function(s) for subsystem '<Root>'
  real_T sum(const real_T x[10]);
  void invNxN(const real_T x[25], real_T y[25]);
};

//-
//  These blocks were eliminated from the model due to optimizations:
//
//  Block '<S1>/Constant3' : Unused code path elimination


//-
//  The generated code includes comments that allow you to trace directly
//  back to the appropriate location in the model.  The basic format
//  is <system>/block_name, where system is the system number (uniquely
//  assigned by Simulink) and block_name is the name of the block.
//
//  Use the MATLAB hilite_system command to trace the generated code back
//  to the model.  For example,
//
//  hilite_system('<S3>')    - opens system 3
//  hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
//
//  Here is the system hierarchy for this model
//
//  '<Root>' : 'Localization_UKF'
//  '<S1>'   : 'Localization_UKF/Localization_UKF'
//  '<S2>'   : 'Localization_UKF/Localization_UKF/SLAM_Check'
//  '<S3>'   : 'Localization_UKF/Localization_UKF/SLAM_Generate_sigma_pt_UKF'
//  '<S4>'   : 'Localization_UKF/Localization_UKF/SLAM_UKF'
//  '<S5>'   : 'Localization_UKF/Localization_UKF/UKF_para'

#endif                                 // RTW_HEADER_Localization_UKF_h_

//
// File trailer for generated code.
//
// [EOF]
//
