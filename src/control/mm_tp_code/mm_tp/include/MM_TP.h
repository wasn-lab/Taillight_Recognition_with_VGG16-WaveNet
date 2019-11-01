//
// File: MM_TP.h
//
// Code generated for Simulink model 'MM_TP'.
//
// Model version                  : 1.392
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Tue Oct 29 16:53:02 2019
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
// Code generation objectives:
//    1. Execution efficiency
//    2. RAM efficiency
// Validation result: Not run
//
#ifndef RTW_HEADER_MM_TP_h_
#define RTW_HEADER_MM_TP_h_
#include <stddef.h>
#include "rtwtypes.h"
#include <cmath>
#include <math.h>
#include <string.h>
#ifndef MM_TP_COMMON_INCLUDES_
# define MM_TP_COMMON_INCLUDES_
#include "rtwtypes.h"
#endif                                 // MM_TP_COMMON_INCLUDES_

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

// Custom Type definition for MATLAB Function: '<S2>/DynamicPathPlanning1'
#ifndef struct_tag_skA4KFEZ4HPkJJBOYCrevdH
#define struct_tag_skA4KFEZ4HPkJJBOYCrevdH

struct tag_skA4KFEZ4HPkJJBOYCrevdH
{
  uint32_T SafeEq;
  uint32_T Absolute;
  uint32_T NaNBias;
  uint32_T NaNWithFinite;
  uint32_T FiniteWithNaN;
  uint32_T NaNWithNaN;
};

#endif                                 //struct_tag_skA4KFEZ4HPkJJBOYCrevdH

#ifndef typedef_skA4KFEZ4HPkJJBOYCrevdH
#define typedef_skA4KFEZ4HPkJJBOYCrevdH

typedef struct tag_skA4KFEZ4HPkJJBOYCrevdH skA4KFEZ4HPkJJBOYCrevdH;

#endif                                 //typedef_skA4KFEZ4HPkJJBOYCrevdH

// Custom Type definition for MATLAB Function: '<S2>/MM'
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
#ifndef struct_emxArray_real_T_113
#define struct_emxArray_real_T_113

struct emxArray_real_T_113
{
  real_T data[113];
  int32_T size;
};

#endif                                 //struct_emxArray_real_T_113

#ifndef typedef_emxArray_real_T_113
#define typedef_emxArray_real_T_113

typedef struct emxArray_real_T_113 emxArray_real_T_113;

#endif                                 //typedef_emxArray_real_T_113

// Block signals and states (default storage) for system '<Root>'
typedef struct {
  emxArray_real_T_113 path_out1;       // '<S2>/Final_Static_Path'
  real_T path_2[113];                  // '<S13>/Dijkstra'
  real_T UnitDelay1_DSTATE[5];         // '<S3>/Unit Delay1'
  real_T UnitDelay35_DSTATE[4];        // '<S3>/Unit Delay35'
  real_T UnitDelay37_DSTATE[25];       // '<S3>/Unit Delay37'
  real_T UnitDelay36_DSTATE[25];       // '<S3>/Unit Delay36'
  real_T UnitDelay34_DSTATE[5];        // '<S3>/Unit Delay34'
  real_T UnitDelay33_DSTATE[25];       // '<S3>/Unit Delay33'
  real_T UnitDelay6_DSTATE[22];        // '<S2>/Unit Delay6'
  real_T UnitDelay19_DSTATE[4];        // '<S2>/Unit Delay19'
  real_T UnitDelay15_DSTATE[4];        // '<S2>/Unit Delay15'
  real_T UnitDelay5_DSTATE[22];        // '<S2>/Unit Delay5'
  real_T Static_Path_ends_POS_data[904];
  real_T Path_RES_0_data[40000];
  real_T Path_RES_0_1[40000];
  real_T Path_RES_1_data[40000];
  real_T ob_distance_data[20000];
  real_T Path_RES_data[80000];
  real_T Forward_Static_Path_0_data[2599];
  real_T path[12769];
  real_T Static_Path_0[2599];          // '<S2>/Final_Static_Path'
  real_T rtb_X_data[20000];
  real_T tmp_data[20000];
  real_T tmp_data_m[20000];
  real_T Path_RES_0_data_c[40000];
  real_T z1_data[20000];
  real_T z1_data_k[20000];
  real_T z1_data_c[20000];
  real_T dist;                         // '<S13>/Dijkstra'
  real_T UnitDelay_DSTATE;             // '<S2>/Unit Delay'
  real_T UnitDelay38_DSTATE;           // '<S3>/Unit Delay38'
  real_T UnitDelay14_DSTATE;           // '<S2>/Unit Delay14'
  real_T UnitDelay16_DSTATE;           // '<S2>/Unit Delay16'
  real_T UnitDelay18_DSTATE;           // '<S2>/Unit Delay18'
  real_T UnitDelay17_DSTATE;           // '<S2>/Unit Delay17'
  real_T UnitDelay7_DSTATE;            // '<S2>/Unit Delay7'
  real_T UnitDelay11_DSTATE;           // '<S2>/Unit Delay11'
  real_T UnitDelay13_DSTATE;           // '<S2>/Unit Delay13'
  real_T Memory1_PreviousInput;        // '<S2>/Memory1'
  real_T Memory_PreviousInput;         // '<S2>/Memory'
  int32_T SFunction_DIMS6[2];          // '<S2>/target_seg_id_search'
  int32_T SFunction_DIMS6_a[2];        // '<S2>/Forward_Seg'
  int32_T SFunction_DIMS4_h[2];        // '<S2>/Final_Static_Path'
  int32_T SFunction_DIMS2;             // '<S2>/target_seg_id_search'
  int32_T SFunction_DIMS3;             // '<S2>/target_seg_id_search'
  int32_T SFunction_DIMS4;             // '<S2>/target_seg_id_search'
  int32_T SFunction_DIMS2_a;           // '<S2>/Forward_Seg1'
  int32_T SFunction_DIMS3_a;           // '<S2>/Forward_Seg1'
  int32_T SFunction_DIMS4_l;           // '<S2>/Forward_Seg1'
  int32_T SFunction_DIMS2_h;           // '<S2>/Forward_Seg'
  int32_T SFunction_DIMS3_k;           // '<S2>/Forward_Seg'
  int32_T SFunction_DIMS4_f;           // '<S2>/Forward_Seg'
  int32_T SFunction_DIMS2_m;           // '<S2>/Final_Static_Path'
  int32_T SFunction_DIMS3_l;           // '<S2>/Final_Static_Path'
  int32_T SFunction_DIMS6_c;           // '<S2>/Final_Static_Path'
  int32_T SFunction_DIMS3_c;           // '<S13>/Dijkstra'
  boolean_T path_out1_not_empty;       // '<S2>/Final_Static_Path'
} DW;

// Constant parameters (default storage)
typedef struct {
  // Pooled Parameter (Expression: Map_data)
  //  Referenced by:
  //    '<S2>/Constant6'
  //    '<S3>/Constant4'

  real_T pooled2[2599];

  // Expression: nodes
  //  Referenced by: '<S2>/Constant3'

  real_T Constant3_Value[339];

  // Expression: segments
  //  Referenced by: '<S2>/Constant5'

  real_T Constant5_Value[339];

  // Expression: diag([0.00025,0.00025,0.0000001,1,0.0001])
  //  Referenced by: '<S3>/Unit Delay37'

  real_T UnitDelay37_InitialCondition[25];

  // Expression: diag([0.1,0.1,0.1,0.0001,0.0001])
  //  Referenced by: '<S3>/Unit Delay36'

  real_T UnitDelay36_InitialCondition[25];

  // Expression: [eye(5)*0.001]
  //  Referenced by: '<S3>/Unit Delay33'

  real_T UnitDelay33_InitialCondition[25];

  // Pooled Parameter (Expression: Veh_CG)
  //  Referenced by:
  //    '<S2>/Constant12'
  //    '<S2>/Constant4'

  real_T pooled7[2];

  // Pooled Parameter (Expression: Veh_size)
  //  Referenced by:
  //    '<S2>/Constant13'
  //    '<S2>/Constant14'
  //    '<S2>/Constant18'

  real_T pooled8[2];
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
  real_T BB_num;                       // '<Root>/BB_num'
  real_T BB_all_XY[400];               // '<Root>/BB_all_XY'
  real_T U_turn;                       // '<Root>/U_turn'
  real_T Look_ahead_time_straight;     // '<Root>/Look_ahead_time_straight'
  real_T Path_flag;                    // '<Root>/Path_flag'
  real_T Look_ahead_time_turn;         // '<Root>/Look_ahead_time_turn'
  real_T ID_turn[8];                   // '<Root>/ID_turn'
  real_T safe_range;                   // '<Root>/safe_range'
  real_T W_1[6];                       // '<Root>/W_1'
  real_T W_2[3];                       // '<Root>/W_2'
  real_T w_fs;                         // '<Root>/w_fs'
  real_T Freespace[37500];             // '<Root>/Freespace'
  real_T Freespace_mode;               // '<Root>/Freespace_mode'
  real_T takeover_mag;                 // '<Root>/takeover_mag'
  real_T forward_length_2;             // '<Root>/forward_length_2'
  real_T J_minvalue_diff_min;          // '<Root>/J_minvalue_diff_min'
  real_T J_minvalue_index;             // '<Root>/J_minvalue_index'
  real_T VirBB_mode;                   // '<Root>/VirBB_mode'
  real_T w_off_avoid;                  // '<Root>/w_off_avoid'
  real_T w_off_;                       // '<Root>/w_off_'
  real_T OB_enlarge;                   // '<Root>/OB_enlarge'
} ExtU;

// External outputs (root outports fed by signals with default storage)
typedef struct {
  real_T Vehicle_Target_x;             // '<Root>/Vehicle_Target_x'
  real_T Vehicle_Target_y;             // '<Root>/Vehicle_Target_y'
  real_T XP_final[6];                  // '<Root>/XP_final'
  real_T YP_final[6];                  // '<Root>/YP_final'
  real_T XP_final_1[6];                // '<Root>/XP_final_1'
  real_T YP_final_1[6];                // '<Root>/YP_final_1'
  real_T X_UKF_SLAM[5];                // '<Root>/X_UKF_SLAM'
  real_T J_minind;                     // '<Root>/J_minind'
  real_T J_finalind;                   // '<Root>/J_finalind'
  real_T End_x;                        // '<Root>/End_x'
  real_T End_y;                        // '<Root>/End_y'
  real_T forward_length;               // '<Root>/forward_length'
  real_T seg_id_near;                  // '<Root>/seg_id_near'
  real_T Target_seg_id;                // '<Root>/Target_seg_id'
  real_T Look_ahead_time;              // '<Root>/Look_ahead_time'
  real_T J_fsc[13];                    // '<Root>/J_fsc'
  real_T forward_length_free;          // '<Root>/forward_length_free'
  real_T U_c[13];                      // '<Root>/U_c'
  real_T U_c_1[13];                    // '<Root>/U_c_1'
  real_T safety_level_all[13];         // '<Root>/safety_level_all'
  real_T safety_level_all_1[13];       // '<Root>/safety_level_all_1'
  real_T takeover_length;              // '<Root>/takeover_length'
  real_T J[13];                        // '<Root>/J'
  real_T avoidance_mode;               // '<Root>/avoidance_mode'
  real_T takeoverlength_ind;           // '<Root>/takeoverlength_ind'
} ExtY;

// Real-time Model Data Structure
struct tag_RTM {
  const char_T * volatile errorStatus;
};

// Constant parameters (default storage)
extern const ConstP rtConstP;

// Class declaration for model MM_TP
class MM_DPP_1ModelClass {
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
  MM_DPP_1ModelClass();

  // Destructor
  ~MM_DPP_1ModelClass();

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
  void merge(int32_T idx[113], real_T x[113], int32_T offset, int32_T np,
             int32_T nq, int32_T iwork[113], real_T xwork[113]);
  void sort(real_T x[113], int32_T idx[113]);
  void rel_dist_xy(const real_T ref_xy[2], const real_T pt_xy[226], real_T dist
                   [113]);
  real_T rel_dist_xy_c(const real_T ref_xy[2], const real_T pt_xy[2]);
  void MM(real_T heading, const real_T X_pos[2], const real_T oi_xy[226], const
          real_T dist_op[113], const real_T Map_data[2599], real_T *seg_id_near,
          real_T *op_distance, real_T oi_near[2], real_T *note, real_T
          *seg_direction, real_T *head_err, real_T num_lane_direction[4], real_T
          *seg_heading);
  void merge_e(int32_T idx_data[], real_T x_data[], int32_T offset, int32_T np,
               int32_T nq, int32_T iwork_data[], real_T xwork_data[]);
  void merge_block(int32_T idx_data[], real_T x_data[], int32_T n, int32_T
                   iwork_data[], real_T xwork_data[]);
  void sortIdx(real_T x_data[], int32_T *x_size, int32_T idx_data[], int32_T
               *idx_size);
  void sort_g(real_T x_data[], int32_T *x_size, int32_T idx_data[], int32_T
              *idx_size);
  void power_l(const real_T a_data[], const int32_T *a_size, real_T y_data[],
               int32_T *y_size);
  void power_lz(const real_T a_data[], const int32_T *a_size, real_T y_data[],
                int32_T *y_size);
  void rel_dist_xy_d(const real_T ref_xy[2], const real_T pt_xy_data[], const
                     int32_T pt_xy_size[2], real_T dist_data[], int32_T
                     *dist_size);
  void MM_f(real_T heading, const real_T X_pos[2], const real_T oi_xy_data[],
            const int32_T oi_xy_size[2], const real_T dist_op_data[], const
            int32_T *dist_op_size, const real_T Map_data_data[], const int32_T
            Map_data_size[2], real_T *seg_id_near, real_T *op_distance, real_T
            oi_near[2], real_T *note, real_T *seg_direction, real_T *head_err,
            real_T num_lane_direction[4], real_T *seg_heading);
  void power_ec(const real_T a_data[], const int32_T *a_size, real_T y_data[],
                int32_T *y_size);
  void power_j(const real_T a_data[], const int32_T *a_size, real_T y_data[],
               int32_T *y_size);
  void power_k(const real_T a_data[], const int32_T *a_size, real_T y_data[],
               int32_T *y_size);
  void abs_i(const real_T x[4], real_T y[4]);
  void power(const real_T a[4], real_T y[4]);
  void G2splines(real_T xa, real_T ya, real_T thetaa, real_T ka, real_T xb,
                 real_T yb, real_T thetab, real_T kb, real_T path_length, real_T
                 X_0[11], real_T Y[11], real_T XP[6], real_T YP[6], real_T K[11],
                 real_T K_1[11], real_T *L_path);
  void power_dw(const real_T a[143], real_T y[143]);
  void power_dw3(const real_T a[143], real_T y[143]);
  real_T std(const real_T x[13]);
  void power_dw3x(const real_T a[13], real_T y[13]);
  void exp_n(real_T x[13]);
  real_T sum_a(const real_T x[13]);
  void power_d(const real_T a[11], real_T y[11]);
  void sqrt_l(real_T x[11]);
  void power_dw3xd(const real_T a_data[], const int32_T a_size[2], real_T
                   y_data[], int32_T y_size[2]);
  void sum_ae(const real_T x_data[], const int32_T x_size[2], real_T y_data[],
              int32_T y_size[2]);
  void sqrt_l5(real_T x_data[], int32_T x_size[2]);
  real_T mod(real_T x);
  void point2safetylevel(const real_T X_data[], const int32_T X_size[2], const
    real_T Y_data[], const int32_T Y_size[2], const real_T Freespace[37500],
    real_T X_grid_data[], int32_T X_grid_size[2], real_T Y_grid_data[], int32_T
    Y_grid_size[2], real_T *safety_level);
  void FreespaceDetectCollision(const real_T Freespace[37500], const real_T XP[6],
    const real_T YP[6], const real_T Vehicle_state[3], real_T forward_length,
    real_T safe_range, const real_T Veh_size[2], const real_T Veh_CG[2], real_T *
    U_c, real_T *safety_level_all, real_T *forward_length_free);
  void abs_a(const real_T x[143], real_T y[143]);
  void G2splines_e(real_T xa, real_T ya, real_T thetaa, real_T ka, real_T xb,
                   real_T yb, real_T thetab, real_T kb, real_T path_length,
                   real_T X_1[11], real_T Y[11], real_T XP[6], real_T YP[6],
                   real_T K[11], real_T K_1[11], real_T *L_path);
  void power_egqso(const real_T a_data[], const int32_T a_size[2], real_T
                   y_data[], int32_T y_size[2]);
  void sum_hx(const real_T x_data[], const int32_T x_size[2], real_T y_data[],
              int32_T y_size[2]);
  void point2safetylevel_b(const real_T X_data[], const int32_T X_size[2], const
    real_T Y_data[], const int32_T Y_size[2], const real_T Freespace[37500],
    real_T X_grid_data[], int32_T X_grid_size[2], real_T Y_grid_data[], int32_T
    Y_grid_size[2], real_T *safety_level);
  void FreespaceDetectCollision_b(const real_T Freespace[37500], const real_T
    XP[6], const real_T YP[6], const real_T Vehicle_state[3], real_T
    forward_length, real_T safe_range, const real_T Veh_size[2], const real_T
    Veh_CG[2], real_T *U_c, real_T *safety_level_all, real_T
    *forward_length_free);
  void power_n(const real_T a[2], real_T y[2]);
  real_T sum_e(const real_T x[2]);
};

//-
//  These blocks were eliminated from the model due to optimizations:
//
//  Block '<S3>/Constant' : Unused code path elimination
//  Block '<S3>/Constant3' : Unused code path elimination
//  Block '<S2>/Constant1' : Unused code path elimination
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
//  '<Root>' : 'MM_TP'
//  '<S1>'   : 'MM_TP/MM'
//  '<S2>'   : 'MM_TP/Target_planner'
//  '<S3>'   : 'MM_TP/MM/UKF_Only'
//  '<S4>'   : 'MM_TP/MM/UKF_Only/SLAM_Check'
//  '<S5>'   : 'MM_TP/MM/UKF_Only/SLAM_Generate_sigma_pt_UKF'
//  '<S6>'   : 'MM_TP/MM/UKF_Only/SLAM_UKF'
//  '<S7>'   : 'MM_TP/MM/UKF_Only/SLAM_UKF_MM'
//  '<S8>'   : 'MM_TP/MM/UKF_Only/SLAM_UKF_MM_Check'
//  '<S9>'   : 'MM_TP/Target_planner/Boundingbox_trans'
//  '<S10>'  : 'MM_TP/Target_planner/DangerousArea1'
//  '<S11>'  : 'MM_TP/Target_planner/DynamicPathPlanning'
//  '<S12>'  : 'MM_TP/Target_planner/DynamicPathPlanning1'
//  '<S13>'  : 'MM_TP/Target_planner/Enabled Subsystem'
//  '<S14>'  : 'MM_TP/Target_planner/EndPointDecision'
//  '<S15>'  : 'MM_TP/Target_planner/EndPointDecision1'
//  '<S16>'  : 'MM_TP/Target_planner/EndPointDecision2'
//  '<S17>'  : 'MM_TP/Target_planner/Fianl_Path_Decision'
//  '<S18>'  : 'MM_TP/Target_planner/Final_Static_Path'
//  '<S19>'  : 'MM_TP/Target_planner/Forward_Length_Decision1'
//  '<S20>'  : 'MM_TP/Target_planner/Forward_Seg'
//  '<S21>'  : 'MM_TP/Target_planner/Forward_Seg1'
//  '<S22>'  : 'MM_TP/Target_planner/J_fsc_design'
//  '<S23>'  : 'MM_TP/Target_planner/MATLAB Function2'
//  '<S24>'  : 'MM_TP/Target_planner/MM'
//  '<S25>'  : 'MM_TP/Target_planner/Steer_gain_scheduling'
//  '<S26>'  : 'MM_TP/Target_planner/Target_Point_Decision'
//  '<S27>'  : 'MM_TP/Target_planner/target_seg_id_search'
//  '<S28>'  : 'MM_TP/Target_planner/Enabled Subsystem/Dijkstra'

#endif                                 // RTW_HEADER_MM_TP_h_

//
// File trailer for generated code.
//
// [EOF]
//
