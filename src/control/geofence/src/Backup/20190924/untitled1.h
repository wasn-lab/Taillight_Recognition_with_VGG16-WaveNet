//
// File: untitled1.h
//
// Code generated for Simulink model 'untitled1'.
//
// Model version                  : 1.5
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Tue Sep 10 16:32:53 2019
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
// Code generation objectives:
//    1. Execution efficiency
//    2. RAM efficiency
// Validation result: Not run
//
#ifndef RTW_HEADER_untitled1_h_
#define RTW_HEADER_untitled1_h_
#include <stddef.h>
#include <cmath>
#include <math.h>
#include <string.h>
#ifndef untitled1_COMMON_INCLUDES_
#define untitled1_COMMON_INCLUDES_
#include "rtwtypes.h"
#endif  // untitled1_COMMON_INCLUDES_

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
  real_T Input[12];    // '<Root>/Input'
  real_T Input1[12];   // '<Root>/Input1'
  real_T Input2[400];  // '<Root>/Input2'
  real_T Input3;       // '<Root>/Input3'
} ExtU;

// External outputs (root outports fed by signals with default storage)
typedef struct
{
  real_T Output;   // '<Root>/Output'
  real_T Output1;  // '<Root>/Output1'
} ExtY;

// Real-time Model Data Structure
struct tag_RTM
{
  const char_T* volatile errorStatus;
};

// Class declaration for model untitled1
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
//  '<Root>' : 'untitled1'
//  '<S1>'   : 'untitled1/Boundary'

#endif  // RTW_HEADER_untitled1_h_

//
// File trailer for generated code.
//
// [EOF]
//
