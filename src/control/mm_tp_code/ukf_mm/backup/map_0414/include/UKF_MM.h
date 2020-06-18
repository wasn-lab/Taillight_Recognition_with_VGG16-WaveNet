//
// File: UKF_MM.h
//
// Code generated for Simulink model 'UKF_MM'.
//
// Model version                  : 1.8
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Thu Apr 16 09:34:12 2020
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
// Code generation objectives:
//    1. Execution efficiency
//    2. RAM efficiency
// Validation result: Not run
//
#ifndef RTW_HEADER_UKF_MM_h_
#define RTW_HEADER_UKF_MM_h_
#include <stddef.h>
#include "rtwtypes.h"
#include <cmath>
#include <string.h>
#ifndef UKF_MM_COMMON_INCLUDES_
# define UKF_MM_COMMON_INCLUDES_
#include "rtwtypes.h"
#endif                                 // UKF_MM_COMMON_INCLUDES_

// Macros for accessing real-time model data structure
#ifndef rtmGetErrorStatus
# define rtmGetErrorStatus(rtm)        ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
# define rtmSetErrorStatus(rtm, val)   ((rtm)->errorStatus = (val))
#endif

// Forward declaration for rtModel
typedef struct tag_RTM RT_MODEL;

// Custom Type definition for MATLAB Function: '<S2>/MM'
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

// Custom Type definition for MATLAB Function: '<S2>/Final_Static_Path'
#ifndef struct_emxArray_real_T_144
#define struct_emxArray_real_T_144

struct emxArray_real_T_144
{
  real_T data[144];
  int32_T size;
};

#endif                                 //struct_emxArray_real_T_144

#ifndef typedef_emxArray_real_T_144
#define typedef_emxArray_real_T_144

typedef struct emxArray_real_T_144 emxArray_real_T_144;

#endif                                 //typedef_emxArray_real_T_144

// Block signals and states (default storage) for system '<Root>'
typedef struct {
  emxArray_real_T_144 path_out1;       // '<S2>/Final_Static_Path'
  real_T path_2[144];                  // '<S9>/Dijkstra'
  real_T UnitDelay1_DSTATE[5];         // '<S3>/Unit Delay1'
  real_T UnitDelay35_DSTATE[4];        // '<S3>/Unit Delay35'
  real_T UnitDelay37_DSTATE[25];       // '<S3>/Unit Delay37'
  real_T UnitDelay36_DSTATE[25];       // '<S3>/Unit Delay36'
  real_T UnitDelay34_DSTATE[5];        // '<S3>/Unit Delay34'
  real_T UnitDelay33_DSTATE[25];       // '<S3>/Unit Delay33'
  real_T path[20736];
  real_T Static_Path_0[3312];          // '<S2>/Final_Static_Path'
  real_T dist;                         // '<S9>/Dijkstra'
  real_T UnitDelay_DSTATE;             // '<S2>/Unit Delay'
  real_T UnitDelay38_DSTATE;           // '<S3>/Unit Delay38'
  real_T Memory1_PreviousInput;        // '<S2>/Memory1'
  real_T Memory_PreviousInput;         // '<S2>/Memory'
  int32_T SFunction_DIMS6[2];          // '<S2>/target_seg_id_search'
  int32_T SFunction_DIMS4_f[2];        // '<S2>/Final_Static_Path'
  int32_T SFunction_DIMS2;             // '<S2>/target_seg_id_search'
  int32_T SFunction_DIMS3;             // '<S2>/target_seg_id_search'
  int32_T SFunction_DIMS4;             // '<S2>/target_seg_id_search'
  int32_T SFunction_DIMS2_h;           // '<S2>/Final_Static_Path'
  int32_T SFunction_DIMS3_f;           // '<S2>/Final_Static_Path'
  int32_T SFunction_DIMS6_c;           // '<S2>/Final_Static_Path'
  int32_T SFunction_DIMS3_g;           // '<S9>/Dijkstra'
  boolean_T path_out1_not_empty;       // '<S2>/Final_Static_Path'
} DW;

// Constant parameters (default storage)
typedef struct {
  // Pooled Parameter (Expression: Map_data)
  //  Referenced by:
  //    '<S2>/Constant6'
  //    '<S3>/Constant4'

  real_T pooled2[3312];

  // Expression: nodes
  //  Referenced by: '<S2>/Constant3'

  real_T Constant3_Value[432];

  // Expression: segments
  //  Referenced by: '<S2>/Constant5'

  real_T Constant5_Value[432];

  // Expression: diag([0.00025,0.00025,0.0000001,1,0.0001])
  //  Referenced by: '<S3>/Unit Delay37'

  real_T UnitDelay37_InitialCondition[25];

  // Expression: diag([0.1,0.1,0.1,0.0001,0.0001])
  //  Referenced by: '<S3>/Unit Delay36'

  real_T UnitDelay36_InitialCondition[25];

  // Expression: [eye(5)*0.001]
  //  Referenced by: '<S3>/Unit Delay33'

  real_T UnitDelay33_InitialCondition[25];
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
  real_T U_turn;                       // '<Root>/U_turn'
  real_T Look_ahead_time_straight;     // '<Root>/Look_ahead_time_straight'
  real_T Look_ahead_time_turn;         // '<Root>/Look_ahead_time_turn'
  real_T ID_turn[8];                   // '<Root>/ID_turn'
} ExtU;

// External outputs (root outports fed by signals with default storage)
typedef struct {
  real_T X_UKF_SLAM[5];                // '<Root>/X_UKF_SLAM'
  real_T seg_id_near;                  // '<Root>/seg_id_near'
  real_T Target_seg_id;                // '<Root>/Target_seg_id'
  real_T Look_ahead_time;              // '<Root>/Look_ahead_time'
} ExtY;

// Real-time Model Data Structure
struct tag_RTM {
  const char_T * volatile errorStatus;
};

// Constant parameters (default storage)
extern const ConstP rtConstP;

// Class declaration for model UKF_MM
class UKF_MMModelClass {
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
  UKF_MMModelClass();

  // Destructor
  ~UKF_MMModelClass();

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
  void merge(int32_T idx[144], real_T x[144], int32_T offset, int32_T np,
             int32_T nq, int32_T iwork[144], real_T xwork[144]);
  void sort(real_T x[144], int32_T idx[144]);
  void power(const real_T a[144], real_T y[144]);
  void rel_dist_xy(const real_T ref_xy[2], const real_T pt_xy[288], real_T dist
                   [144]);
  real_T rel_dist_xy_g(const real_T ref_xy[2], const real_T pt_xy[2]);
  void MM(real_T heading, const real_T X_pos[2], const real_T oi_xy[288], const
          real_T dist_op[144], const real_T Map_data[3312], real_T *seg_id_near,
          real_T *op_distance, real_T oi_near[2], real_T *note, real_T
          *seg_direction, real_T *head_err, real_T num_lane_direction[4], real_T
          *seg_heading);
  void merge_o(int32_T idx_data[], real_T x_data[], int32_T offset, int32_T np,
               int32_T nq, int32_T iwork_data[], real_T xwork_data[]);
  void merge_block(int32_T idx_data[], real_T x_data[], int32_T n, int32_T
                   iwork_data[], real_T xwork_data[]);
  void sortIdx(real_T x_data[], int32_T *x_size, int32_T idx_data[], int32_T
               *idx_size);
  void sort_m(real_T x_data[], int32_T *x_size, int32_T idx_data[], int32_T
              *idx_size);
  void power_j(const real_T a_data[], const int32_T *a_size, real_T y_data[],
               int32_T *y_size);
  void power_jm(const real_T a_data[], const int32_T *a_size, real_T y_data[],
                int32_T *y_size);
  void rel_dist_xy_o(const real_T ref_xy[2], const real_T pt_xy_data[], const
                     int32_T pt_xy_size[2], real_T dist_data[], int32_T
                     *dist_size);
  void MM_o(real_T heading, const real_T X_pos[2], const real_T oi_xy_data[],
            const int32_T oi_xy_size[2], const real_T dist_op_data[], const
            int32_T *dist_op_size, const real_T Map_data_data[], const int32_T
            Map_data_size[2], real_T *seg_id_near, real_T *op_distance, real_T
            oi_near[2], real_T *note, real_T *seg_direction, real_T *head_err,
            real_T num_lane_direction[4], real_T *seg_heading);
  void power_d(const real_T a[2], real_T y[2]);
  real_T sum_k(const real_T x[2]);
};

//-
//  These blocks were eliminated from the model due to optimizations:
//
//  Block '<S3>/Constant' : Unused code path elimination
//  Block '<S3>/Constant3' : Unused code path elimination
//  Block '<S2>/To Workspace1' : Unused code path elimination
//  Block '<S2>/To Workspace2' : Unused code path elimination
//  Block '<S2>/To Workspace3' : Unused code path elimination
//  Block '<S2>/To Workspace4' : Unused code path elimination
//  Block '<S2>/To Workspace5' : Unused code path elimination


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
//  '<Root>' : 'UKF_MM'
//  '<S1>'   : 'UKF_MM/MM'
//  '<S2>'   : 'UKF_MM/Target_planner'
//  '<S3>'   : 'UKF_MM/MM/UKF_Only'
//  '<S4>'   : 'UKF_MM/MM/UKF_Only/SLAM_Check'
//  '<S5>'   : 'UKF_MM/MM/UKF_Only/SLAM_Generate_sigma_pt_UKF'
//  '<S6>'   : 'UKF_MM/MM/UKF_Only/SLAM_UKF'
//  '<S7>'   : 'UKF_MM/MM/UKF_Only/SLAM_UKF_MM'
//  '<S8>'   : 'UKF_MM/MM/UKF_Only/SLAM_UKF_MM_Check'
//  '<S9>'   : 'UKF_MM/Target_planner/Enabled Subsystem'
//  '<S10>'  : 'UKF_MM/Target_planner/Final_Static_Path'
//  '<S11>'  : 'UKF_MM/Target_planner/MATLAB Function2'
//  '<S12>'  : 'UKF_MM/Target_planner/MM'
//  '<S13>'  : 'UKF_MM/Target_planner/Steer_gain_scheduling'
//  '<S14>'  : 'UKF_MM/Target_planner/target_seg_id_search'
//  '<S15>'  : 'UKF_MM/Target_planner/Enabled Subsystem/Dijkstra'

#endif                                 // RTW_HEADER_UKF_MM_h_

//
// File trailer for generated code.
//
// [EOF]
//
