//
// File: Geofence.h
//
// Code generated for Simulink model 'Geofence'.
//
// Model version                  : 1.9
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Wed Oct  2 13:29:17 2019
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
// Code generation objectives:
//    1. Execution efficiency
//    2. RAM efficiency
// Validation result: Not run
//
#ifndef RTW_HEADER_Geofence_h_
#define RTW_HEADER_Geofence_h_
#include <stddef.h>
#include <cmath>
#include <math.h>
#include <string.h>
#ifndef Geofence_COMMON_INCLUDES_
#define Geofence_COMMON_INCLUDES_
#include "rtwtypes.h"
#endif  // Geofence_COMMON_INCLUDES_

// Macros for accessing real-time model data structure
#ifndef rtmGetErrorStatus
#define rtmGetErrorStatus(rtm) ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
#define rtmSetErrorStatus(rtm, val) ((rtm)->errorStatus = (val))
#endif

// Forward declaration for rtModel
typedef struct tag_RTM RT_MODEL;

// Block signals and states (default storage) for system '<Root>'
typedef struct
{
  real_T X_points_one[10201];
  real_T Y_points_one[10201];
  real_T X_points_two[10201];
  real_T Y_points_two[10201];
  real_T X_points[20402];
  real_T Y_points[20402];
  real_T Line_length_one[10201];
  real_T Line_length_two[10201];
  real_T Line_length[20402];
} DW;

// External inputs (root inport signals with default storage)
typedef struct
{
  real_T X_Poly[12];         // '<Root>/X_Poly'
  real_T Y_Poly[12];         // '<Root>/Y_Poly'
  real_T BoundingBox[1000];  // '<Root>/BoundingBox'
  real_T OB_num;             // '<Root>/OB_num'
} ExtU;

// External outputs (root outports fed by signals with default storage)
typedef struct
{
  real_T Trigger;    // '<Root>/Trigger'
  real_T Range;      // '<Root>/Range'
  real_T Obj_Speed;  // '<Root>/Obj_Speed'
} ExtY;

// Real-time Model Data Structure
struct tag_RTM
{
  const char_T* volatile errorStatus;
};

// Class declaration for model Geofence
class untitled1ModelClass
{
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
  untitled1ModelClass();

  // Destructor
  ~untitled1ModelClass();

  // Real-Time Model get method
  RT_MODEL* getRTM();

  // private data and function members
private:
  // Block signals and states
  DW rtDW;

  // Real-Time Model
  RT_MODEL rtM;
};

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
//  '<Root>' : 'Geofence'
//  '<S1>'   : 'Geofence/Calculator'

#endif  // RTW_HEADER_Geofence_h_

//
// File trailer for generated code.
//
// [EOF]
//
