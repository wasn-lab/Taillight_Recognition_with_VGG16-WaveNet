//
// File: Map_data_read.h
//
// Code generated for Simulink model 'Map_data_read'.
//
// Model version                  : 1.2
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Fri Oct 25 18:29:44 2019
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
// Code generation objectives:
//    1. Execution efficiency
//    2. RAM efficiency
// Validation result: Not run
//
#ifndef RTW_HEADER_Map_data_read_h_
#define RTW_HEADER_Map_data_read_h_
#include <string.h>
#ifndef Map_data_read_COMMON_INCLUDES_
# define Map_data_read_COMMON_INCLUDES_
#include "rtwtypes.h"
#endif                                 // Map_data_read_COMMON_INCLUDES_

// Macros for accessing real-time model data structure
#ifndef rtmGetErrorStatus
# define rtmGetErrorStatus(rtm)        ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
# define rtmSetErrorStatus(rtm, val)   ((rtm)->errorStatus = (val))
#endif

// Forward declaration for rtModel
typedef struct tag_RTM RT_MODEL;

// Constant parameters (default storage)
typedef struct {
  // Expression: Map_data
  //  Referenced by: '<S1>/Constant4'

  real_T Constant4_Value[2599];

  // Expression: nodes
  //  Referenced by: '<S1>/Constant3'

  real_T Constant3_Value[339];

  // Expression: segments
  //  Referenced by: '<S1>/Constant5'

  real_T Constant5_Value[339];
} ConstP;

// External outputs (root outports fed by signals with default storage)
typedef struct {
  real_T Map_data[2599];               // '<Root>/Map_data'
  real_T nodes[339];                   // '<Root>/nodes'
  real_T segments[339];                // '<Root>/segments'
  real_T start_node_id;                // '<Root>/start_node_id'
  real_T finish_node_id;               // '<Root>/finish_node_id'
  real_T Map_data_length1;             // '<Root>/Map_data_length1'
  real_T Map_data_length2;             // '<Root>/Map_data_length2'
  real_T nodes_length1;                // '<Root>/nodes_length1'
  real_T nodes_length2;                // '<Root>/nodes_length2'
  real_T segments_length1;             // '<Root>/segments_length1'
  real_T segments_length2;             // '<Root>/segments_length2'
} ExtY;

// Real-time Model Data Structure
struct tag_RTM {
  const char_T * volatile errorStatus;
};

// Constant parameters (default storage)
extern const ConstP rtConstP;

// Class declaration for model Map_data_read
class Map_data_readModelClass {
  // public data and function members
 public:
  // External outputs
  ExtY rtY;

  // model initialize function
  void initialize();

  // model step function
  void step();

  // Constructor
  Map_data_readModelClass();

  // Destructor
  ~Map_data_readModelClass();

  // Real-Time Model get method
  RT_MODEL * getRTM();

  // private data and function members
 private:
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
//  '<Root>' : 'Map_data_read'
//  '<S1>'   : 'Map_data_read/Map_data_read'
//  '<S2>'   : 'Map_data_read/Map_data_read/MATLAB Function'

#endif                                 // RTW_HEADER_Map_data_read_h_

//
// File trailer for generated code.
//
// [EOF]
//
