//
// File: MM.h
//
// Code generated for Simulink model 'MM'.
//
// Model version                  : 1.1
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Fri Oct 25 18:32:04 2019
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
// Code generation objectives:
//    1. Execution efficiency
//    2. RAM efficiency
// Validation result: Not run
//
#ifndef RTW_HEADER_MM_h_
#define RTW_HEADER_MM_h_
#include <stddef.h>
#include "rtwtypes.h"
#include <cmath>
#include <string.h>
#ifndef MM_COMMON_INCLUDES_
# define MM_COMMON_INCLUDES_
#include "rtwtypes.h"
#endif                                 // MM_COMMON_INCLUDES_

// Macros for accessing real-time model data structure
#ifndef rtmGetErrorStatus
# define rtmGetErrorStatus(rtm)        ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
# define rtmSetErrorStatus(rtm, val)   ((rtm)->errorStatus = (val))
#endif

// Forward declaration for rtModel
typedef struct tag_RTM RT_MODEL;

// Custom Type definition for MATLAB Function: '<S1>/MM'
#ifndef struct_tag_spGKsvEVm7uA89hv31XX4LH
#define struct_tag_spGKsvEVm7uA89hv31XX4LH

struct tag_spGKsvEVm7uA89hv31XX4LH
{
  uint32_T MissingPlacement;
  uint32_T ComparisonMethod;
};

#endif                                 //struct_tag_spGKsvEVm7uA89hv31XX4LH

#ifndef typedef_spGKsvEVm7uA89hv31XX4LH
#define typedef_spGKsvEVm7uA89hv31XX4LH

typedef struct tag_spGKsvEVm7uA89hv31XX4LH spGKsvEVm7uA89hv31XX4LH;

#endif                                 //typedef_spGKsvEVm7uA89hv31XX4LH

#ifndef struct_tag_sJCxfmxS8gBOONUZjbjUd9E
#define struct_tag_sJCxfmxS8gBOONUZjbjUd9E

struct tag_sJCxfmxS8gBOONUZjbjUd9E
{
  boolean_T CaseSensitivity;
  boolean_T StructExpand;
  char_T PartialMatching[6];
  boolean_T IgnoreNulls;
};

#endif                                 //struct_tag_sJCxfmxS8gBOONUZjbjUd9E

#ifndef typedef_sJCxfmxS8gBOONUZjbjUd9E
#define typedef_sJCxfmxS8gBOONUZjbjUd9E

typedef struct tag_sJCxfmxS8gBOONUZjbjUd9E sJCxfmxS8gBOONUZjbjUd9E;

#endif                                 //typedef_sJCxfmxS8gBOONUZjbjUd9E

// Custom Type definition for MATLAB Function: '<S1>/Final_Static_Path'
#ifndef struct_emxArray_real_T_1000
#define struct_emxArray_real_T_1000

struct emxArray_real_T_1000
{
  real_T data[1000];
  int32_T size;
};

#endif                                 //struct_emxArray_real_T_1000

#ifndef typedef_emxArray_real_T_1000
#define typedef_emxArray_real_T_1000

typedef struct emxArray_real_T_1000 emxArray_real_T_1000;

#endif                                 //typedef_emxArray_real_T_1000

// Block signals and states (default storage) for system '<Root>'
typedef struct {
  emxArray_real_T_1000 path_out1;      // '<S1>/Final_Static_Path'
  real_T path_2[1000];                 // '<S2>/Dijkstra'
  real_T oi_xy_data[2000];
  real_T nodes_data[3000];
  real_T segments_data[3000];
  real_T node_ids_data[1000];
  real_T table_data[2000];
  real_T shortest_distance_data[1000];
  real_T path_data[1000000];
  real_T tmp_path_data[1000];
  real_T Static_Path_0[1000000];       // '<S1>/Final_Static_Path'
  real_T xy_ini_data[2000];
  real_T xy_end_data[2000];
  real_T seg_id_data[1000];
  real_T ind_temp_data[1000];
  real_T SEG_GPS_HEAD_data[2000];
  real_T dist_ini_data[1000];
  real_T dist_end_data[1000];
  real_T pt_xy_data[1000];
  real_T tmp_data[1000];
  real_T tmp_data_m[1000];
  real_T z1_data[1000];
  real_T z1_data_c[1000];
  real_T vwork_data[1000];
  real_T xwork_data[1000];
  real_T c_x_data[1000];
  real_T dist;                         // '<S2>/Dijkstra'
  real_T UnitDelay_DSTATE;             // '<S1>/Unit Delay'
  real_T Memory1_PreviousInput;        // '<S1>/Memory1'
  real_T Memory_PreviousInput;         // '<S1>/Memory'
  int32_T SFunction_DIMS2[2];          // '<S1>/Final_Static_Path'
  int32_T SFunction_DIMS3[2];          // '<S1>/Final_Static_Path'
  int32_T SFunction_DIMS4[2];          // '<S1>/Final_Static_Path'
  int32_T SFunction_DIMS8[2];          // '<S1>/Final_Static_Path'
  int32_T SFunction_DIMS3_l[2];        // '<S2>/Dijkstra'
  int32_T b_index_data[2000];
  boolean_T path_out1_not_empty;       // '<S1>/Final_Static_Path'
} DW;

// External inputs (root inport signals with default storage)
typedef struct {
  real_T Map_data_i[1000000];          // '<Root>/Map_data_i'
  real_T X_UKF_SLAM_i[5];              // '<Root>/X_UKF_SLAM_i'
  real_T nodes_i[3000];                // '<Root>/nodes_i'
  real_T segments_i[3000];             // '<Root>/segments_i'
  real_T start_node_id_i;              // '<Root>/start_node_id_i'
  real_T finish_node_id_i;             // '<Root>/finish_node_id_i'
  real_T Map_data_length1_i;           // '<Root>/Map_data_length1_i'
  real_T Map_data_length2_i;           // '<Root>/Map_data_length2_i'
  real_T nodes_length1_i;              // '<Root>/nodes_length1_i'
  real_T nodes_length2_i;              // '<Root>/nodes_length2_i'
  real_T segments_length1_i;           // '<Root>/segments_length1_i'
  real_T segments_length2_i;           // '<Root>/segments_length2_i'
} ExtU;

// External outputs (root outports fed by signals with default storage)
typedef struct {
  real_T seg_id_near;                  // '<Root>/seg_id_near'
  real_T Oi_near[2];                   // '<Root>/Oi_near'
  real_T seg_Curvature;                // '<Root>/seg_Curvature'
  real_T Static_Path_0[1000000];       // '<Root>/Static_Path_0'
  real_T Static_Path_0_length1;        // '<Root>/Static_Path_0_length1'
  real_T Static_Path_0_length2;        // '<Root>/Static_Path_0_length2'
} ExtY;

// External output sizes (for root outports fed by signals with variable sizes)
typedef struct {
  int32_T SFunction_DIMS4[2];          // '<Root>/Static_Path_0'
} ExtYSize;

// Real-time Model Data Structure
struct tag_RTM {
  const char_T * volatile errorStatus;
};

// Class declaration for model MM
class MMModelClass {
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
  MMModelClass();

  // Destructor
  ~MMModelClass();

  // Real-Time Model get method
  RT_MODEL * getRTM();

  // private data and function members
 private:
  // Block signals and states
  DW rtDW;

  // Real-Time Model
  RT_MODEL rtM;

  // private member function(s) for subsystem '<Root>'
  int32_T nonSingletonDim(const int32_T *x_size);
  void merge(int32_T idx_data[], real_T x_data[], int32_T offset, int32_T np,
             int32_T nq, int32_T iwork_data[], real_T xwork_data[]);
  void merge_block(int32_T idx_data[], real_T x_data[], int32_T offset, int32_T
                   n, int32_T preSortLevel, int32_T iwork_data[], real_T
                   xwork_data[]);
  void merge_pow2_block(int32_T idx_data[], real_T x_data[], int32_T offset);
  void sortIdx(real_T x_data[], int32_T *x_size, int32_T idx_data[], int32_T
               *idx_size);
  void sort(real_T x_data[], int32_T *x_size, int32_T idx_data[], int32_T
            *idx_size);
  void power_n(const real_T a_data[], const int32_T *a_size, real_T y_data[],
               int32_T *y_size);
  void power_n5(const real_T a_data[], const int32_T *a_size, real_T y_data[],
                int32_T *y_size);
  void rel_dist_xy(const real_T ref_xy[2], const real_T pt_xy_data[], const
                   int32_T pt_xy_size[2], real_T dist_data[], int32_T *dist_size);
  real_T rel_dist_xy_f(const real_T ref_xy[2], const real_T pt_xy[2]);
  void MM_g(real_T heading, const real_T X_pos[2], const real_T oi_xy_data[],
            const int32_T oi_xy_size[2], const real_T dist_op_data[], const
            int32_T *dist_op_size, const real_T Map_data_data[], const int32_T
            Map_data_size[2], real_T *seg_id_near, real_T *op_distance, real_T
            oi_near[2], real_T *note, real_T *seg_direction, real_T *head_err,
            real_T num_lane_direction[4], real_T *seg_heading);
  void power(const real_T a_data[], const int32_T a_size[2], real_T y_data[],
             int32_T y_size[2]);
  real_T sum(const real_T x_data[], const int32_T x_size[2]);
};

//-
//  These blocks were eliminated from the model due to optimizations:
//
//  Block '<S1>/Constant' : Unused code path elimination
//  Block '<S1>/To Workspace1' : Unused code path elimination
//  Block '<S1>/To Workspace2' : Unused code path elimination
//  Block '<S1>/To Workspace3' : Unused code path elimination
//  Block '<S1>/To Workspace4' : Unused code path elimination
//  Block '<S1>/To Workspace5' : Unused code path elimination


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
//  '<Root>' : 'MM'
//  '<S1>'   : 'MM/Map_Matching'
//  '<S2>'   : 'MM/Map_Matching/Enabled Subsystem'
//  '<S3>'   : 'MM/Map_Matching/Final_Static_Path'
//  '<S4>'   : 'MM/Map_Matching/MATLAB Function'
//  '<S5>'   : 'MM/Map_Matching/MM'
//  '<S6>'   : 'MM/Map_Matching/SLAM_UKF_MM_Check'
//  '<S7>'   : 'MM/Map_Matching/Enabled Subsystem/Dijkstra'

#endif                                 // RTW_HEADER_MM_h_

//
// File trailer for generated code.
//
// [EOF]
//
