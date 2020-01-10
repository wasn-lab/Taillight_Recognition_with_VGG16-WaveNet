//
// File: Map_data_read.cpp
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
#include "Map_data_read.h"

// Model step function
void Map_data_readModelClass::step()
{
  // Outport: '<Root>/Map_data_length1' incorporates:
  //   MATLAB Function: '<S1>/MATLAB Function'

  rtY.Map_data_length1 = 113.0;

  // Outport: '<Root>/Map_data_length2' incorporates:
  //   MATLAB Function: '<S1>/MATLAB Function'

  rtY.Map_data_length2 = 23.0;

  // Outport: '<Root>/nodes_length1' incorporates:
  //   MATLAB Function: '<S1>/MATLAB Function'

  rtY.nodes_length1 = 113.0;

  // Outport: '<Root>/nodes_length2' incorporates:
  //   MATLAB Function: '<S1>/MATLAB Function'

  rtY.nodes_length2 = 3.0;

  // Outport: '<Root>/segments_length1' incorporates:
  //   MATLAB Function: '<S1>/MATLAB Function'

  rtY.segments_length1 = 113.0;

  // Outport: '<Root>/segments_length2' incorporates:
  //   MATLAB Function: '<S1>/MATLAB Function'

  rtY.segments_length2 = 3.0;
}

// Model initialize function
void Map_data_readModelClass::initialize()
{
  // ConstCode for Outport: '<Root>/Map_data' incorporates:
  //   Constant: '<S1>/Constant4'

  memcpy(&rtY.Map_data[0], &rtConstP.Constant4_Value[0], 2599U * sizeof(real_T));

  // ConstCode for Outport: '<Root>/nodes' incorporates:
  //   Constant: '<S1>/Constant3'

  memcpy(&rtY.nodes[0], &rtConstP.Constant3_Value[0], 339U * sizeof(real_T));

  // ConstCode for Outport: '<Root>/segments' incorporates:
  //   Constant: '<S1>/Constant5'

  memcpy(&rtY.segments[0], &rtConstP.Constant5_Value[0], 339U * sizeof(real_T));

  // ConstCode for Outport: '<Root>/start_node_id' incorporates:
  //   Constant: '<S1>/Constant7'

  rtY.start_node_id = 1.0;

  // ConstCode for Outport: '<Root>/finish_node_id' incorporates:
  //   Constant: '<S1>/Constant8'

  rtY.finish_node_id = 113.0;
}

// Constructor
Map_data_readModelClass::Map_data_readModelClass()
{
}

// Destructor
Map_data_readModelClass::~Map_data_readModelClass()
{
  // Currently there is no destructor body generated.
}

// Real-Time Model get method
RT_MODEL * Map_data_readModelClass::getRTM()
{
  return (&rtM);
}

//
// File trailer for generated code.
//
// [EOF]
//
