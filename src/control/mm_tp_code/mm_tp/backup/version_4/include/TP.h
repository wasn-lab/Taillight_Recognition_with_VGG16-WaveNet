//
// File: TP.h
//
// Code generated for Simulink model 'TP'.
//
// Model version                  : 1.2
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Fri Oct 25 18:35:50 2019
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
// Code generation objectives:
//    1. Execution efficiency
//    2. RAM efficiency
// Validation result: Not run
//
#ifndef RTW_HEADER_TP_h_
#define RTW_HEADER_TP_h_
#include <stddef.h>
#include "rtwtypes.h"
#include <cmath>
#include <math.h>
#include <string.h>
#ifndef TP_COMMON_INCLUDES_
# define TP_COMMON_INCLUDES_
#include "rtwtypes.h"
#endif                                 // TP_COMMON_INCLUDES_

// Macros for accessing real-time model data structure
#ifndef rtmGetErrorStatus
# define rtmGetErrorStatus(rtm)        ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
# define rtmSetErrorStatus(rtm, val)   ((rtm)->errorStatus = (val))
#endif

// Forward declaration for rtModel
typedef struct tag_RTM RT_MODEL;

// Custom Type definition for MATLAB Function: '<S1>/DynamicPathPlanning1'
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

// Block signals and states (default storage) for system '<Root>'
typedef struct {
  real_T UnitDelay19_DSTATE[4];        // '<S1>/Unit Delay19'
  real_T UnitDelay15_DSTATE[4];        // '<S1>/Unit Delay15'
  real_T UnitDelay5_DSTATE[22];        // '<S1>/Unit Delay5'
  real_T UnitDelay6_DSTATE[22];        // '<S1>/Unit Delay6'
  real_T xy_ends_POS_data[4000];
  real_T seg_id_data[1000];
  real_T Forward_Static_Path_id_0_data[1000];
  real_T Static_Path_ends_POS_data[8000];
  real_T Forward_Static_Path_data[4002];
  real_T Path_RES_0_data[40000];
  real_T Path_RES_0_1[40000];
  real_T Path_RES_1_data[40000];
  real_T ob_distance_data[20000];
  real_T Path_RES_data[80000];
  real_T Forward_Static_Path_data_m[4002];
  real_T Forward_Static_Path_0_data[1000000];
  real_T end_heading_0_data[1000];
  real_T Static_Path_0[1000000];       // '<S1>/MATLAB Function1'
  real_T Forward_Static_Path_x_p[1000];// '<S1>/Forward_Seg'
  real_T Forward_Static_Path_y_gb[1000];// '<S1>/Forward_Seg'
  real_T Forward_Static_Path_id_h[1000];// '<S1>/Forward_Seg'
  real_T Forward_Static_Path_id_g[1000];// '<S1>/Forward_Seg1'
  real_T tmp_data[20000];
  real_T tmp_data_c[20000];
  real_T tmp_data_k[20000];
  real_T seg_id_data_c[2000];
  real_T Forward_Static_Path_data_b[2001];
  real_T Path_RES_0_data_p[40000];
  real_T z1_data[20000];
  real_T z1_data_c[20000];
  real_T z1_data_f[20000];
  real_T UnitDelay14_DSTATE;           // '<S1>/Unit Delay14'
  real_T UnitDelay16_DSTATE;           // '<S1>/Unit Delay16'
  real_T UnitDelay18_DSTATE;           // '<S1>/Unit Delay18'
  real_T UnitDelay17_DSTATE;           // '<S1>/Unit Delay17'
  real_T UnitDelay7_DSTATE;            // '<S1>/Unit Delay7'
  real_T UnitDelay11_DSTATE;           // '<S1>/Unit Delay11'
  real_T UnitDelay13_DSTATE;           // '<S1>/Unit Delay13'
  int32_T SFunction_DIMS6[2];          // '<S1>/target_seg_id_search'
  int32_T SFunction_DIMS2_g[2];        // '<S1>/MATLAB Function1'
  int32_T SFunction_DIMS2_i[2];        // '<S1>/Forward_Seg1'
  int32_T SFunction_DIMS3_a[2];        // '<S1>/Forward_Seg1'
  int32_T SFunction_DIMS4_l[2];        // '<S1>/Forward_Seg1'
  int32_T SFunction_DIMS2_c[2];        // '<S1>/Forward_Seg'
  int32_T SFunction_DIMS3_n[2];        // '<S1>/Forward_Seg'
  int32_T SFunction_DIMS4_k[2];        // '<S1>/Forward_Seg'
  int32_T SFunction_DIMS2;             // '<S1>/target_seg_id_search'
  int32_T SFunction_DIMS3;             // '<S1>/target_seg_id_search'
  int32_T SFunction_DIMS4;             // '<S1>/target_seg_id_search'
} DW;

// Constant parameters (default storage)
typedef struct {
  // Pooled Parameter (Expression: Veh_CG)
  //  Referenced by:
  //    '<S1>/Constant12'
  //    '<S1>/Constant4'

  real_T pooled4[2];

  // Pooled Parameter (Expression: Veh_size)
  //  Referenced by:
  //    '<S1>/Constant13'
  //    '<S1>/Constant14'
  //    '<S1>/Constant18'

  real_T pooled5[2];
} ConstP;

// External inputs (root inport signals with default storage)
typedef struct {
  real_T BB_num;                       // '<Root>/BB_num'
  real_T BB_all_XY[400];               // '<Root>/BB_all_XY'
  real_T X_UKF_SLAM_i1[5];             // '<Root>/X_UKF_SLAM_i1'
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
  real_T Speed_mps1;                   // '<Root>/Speed_mps1'
  real_T Look_ahead_S0;                // '<Root>/Look_ahead_S0'
  real_T Oi_near_i[2];                 // '<Root>/Oi_near_i'
  real_T seg_id_near_i;                // '<Root>/seg_id_near_i'
  real_T seg_Curvature_i;              // '<Root>/seg_Curvature_i'
  real_T Static_Path_0_i[1000000];     // '<Root>/Static_Path_0_i'
  real_T Static_Path_0_length1_i;      // '<Root>/Static_Path_0_length1_i'
  real_T Static_Path_0_length2_i;      // '<Root>/Static_Path_0_length2_i'
} ExtU;

// External outputs (root outports fed by signals with default storage)
typedef struct {
  real_T Vehicle_Target_x;             // '<Root>/Vehicle_Target_x'
  real_T Vehicle_Target_y;             // '<Root>/Vehicle_Target_y'
  real_T XP_final[6];                  // '<Root>/XP_final'
  real_T YP_final[6];                  // '<Root>/YP_final'
  real_T XP_final_1[6];                // '<Root>/XP_final_1'
  real_T YP_final_1[6];                // '<Root>/YP_final_1'
  real_T J_minind;                     // '<Root>/J_minind'
  real_T J_finalind;                   // '<Root>/J_finalind'
  real_T End_x;                        // '<Root>/End_x'
  real_T End_y;                        // '<Root>/End_y'
  real_T forward_length;               // '<Root>/forward_length'
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
  real_T OB_enlargescale;              // '<Root>/OB_enlargescale'
  real_T OB_enlargescale_frontbehind;  // '<Root>/OB_enlargescale_frontbehind'
} ExtY;

// Real-time Model Data Structure
struct tag_RTM {
  const char_T * volatile errorStatus;
};

// Constant parameters (default storage)
extern const ConstP rtConstP;

// Class declaration for model TP
class TPModelClass {
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
  TPModelClass();

  // Destructor
  ~TPModelClass();

  // Real-Time Model get method
  RT_MODEL * getRTM();

  // private data and function members
 private:
  // Block signals and states
  DW rtDW;

  // Real-Time Model
  RT_MODEL rtM;

  // private member function(s) for subsystem '<Root>'
  void power_m(const real_T a_data[], const int32_T *a_size, real_T y_data[],
               int32_T *y_size);
  void power_a(const real_T a_data[], const int32_T *a_size, real_T y_data[],
               int32_T *y_size);
  void abs_i(const real_T x[4], real_T y[4]);
  void power(const real_T a[4], real_T y[4]);
  void G2splines(real_T xa, real_T ya, real_T thetaa, real_T ka, real_T xb,
                 real_T yb, real_T thetab, real_T kb, real_T path_length, real_T
                 X_0[11], real_T Y[11], real_T XP[6], real_T YP[6], real_T K[11],
                 real_T K_1[11], real_T *L_path);
  void power_bv(const real_T a[143], real_T y[143]);
  void power_bvw(const real_T a[143], real_T y[143]);
  real_T std(const real_T x[13]);
  void power_bvwt(const real_T a[13], real_T y[13]);
  void exp_n(real_T x[13]);
  real_T sum(const real_T x[13]);
  void power_b(const real_T a[11], real_T y[11]);
  void sqrt_f(real_T x[11]);
  void power_bvwts(const real_T a_data[], const int32_T a_size[2], real_T
                   y_data[], int32_T y_size[2]);
  void sum_p(const real_T x_data[], const int32_T x_size[2], real_T y_data[],
             int32_T y_size[2]);
  void sqrt_fh(real_T x_data[], int32_T x_size[2]);
  real_T mod(real_T x);
  void point2safetylevel(const real_T X_data[], const int32_T X_size[2], const
    real_T Y_data[], const int32_T Y_size[2], const real_T Freespace[37500],
    real_T X_grid_data[], int32_T X_grid_size[2], real_T Y_grid_data[], int32_T
    Y_grid_size[2], real_T *safety_level);
  void FreespaceDetectCollision(const real_T Freespace[37500], const real_T XP[6],
    const real_T YP[6], const real_T Vehicle_state[3], real_T forward_length,
    real_T safe_range, const real_T Veh_size[2], const real_T Veh_CG[2], real_T *
    U_c, real_T *safety_level_all, real_T *forward_length_free);
  void abs_n(const real_T x[143], real_T y[143]);
  void power_j(const real_T a_data[], const int32_T *a_size, real_T y_data[],
               int32_T *y_size);
  void G2splines_k(real_T xa, real_T ya, real_T thetaa, real_T ka, real_T xb,
                   real_T yb, real_T thetab, real_T kb, real_T path_length,
                   real_T X_1[11], real_T Y[11], real_T XP[6], real_T YP[6],
                   real_T K[11], real_T K_1[11], real_T *L_path);
  void power_pcxfb(const real_T a_data[], const int32_T a_size[2], real_T
                   y_data[], int32_T y_size[2]);
  void sum_h1(const real_T x_data[], const int32_T x_size[2], real_T y_data[],
              int32_T y_size[2]);
  void point2safetylevel_k(const real_T X_data[], const int32_T X_size[2], const
    real_T Y_data[], const int32_T Y_size[2], const real_T Freespace[37500],
    real_T X_grid_data[], int32_T X_grid_size[2], real_T Y_grid_data[], int32_T
    Y_grid_size[2], real_T *safety_level);
  void FreespaceDetectCollision_m(const real_T Freespace[37500], const real_T
    XP[6], const real_T YP[6], const real_T Vehicle_state[3], real_T
    forward_length, real_T safe_range, const real_T Veh_size[2], const real_T
    Veh_CG[2], real_T *U_c, real_T *safety_level_all, real_T
    *forward_length_free);
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
//  '<Root>' : 'TP'
//  '<S1>'   : 'TP/Target_planner'
//  '<S2>'   : 'TP/Target_planner/Boundingbox_trans'
//  '<S3>'   : 'TP/Target_planner/DangerousArea1'
//  '<S4>'   : 'TP/Target_planner/DynamicPathPlanning'
//  '<S5>'   : 'TP/Target_planner/DynamicPathPlanning1'
//  '<S6>'   : 'TP/Target_planner/EndPointDecision'
//  '<S7>'   : 'TP/Target_planner/EndPointDecision1'
//  '<S8>'   : 'TP/Target_planner/EndPointDecision2'
//  '<S9>'   : 'TP/Target_planner/Fianl_Path_Decision'
//  '<S10>'  : 'TP/Target_planner/Forward_Length_Decision1'
//  '<S11>'  : 'TP/Target_planner/Forward_Seg'
//  '<S12>'  : 'TP/Target_planner/Forward_Seg1'
//  '<S13>'  : 'TP/Target_planner/J_fsc_design'
//  '<S14>'  : 'TP/Target_planner/MATLAB Function'
//  '<S15>'  : 'TP/Target_planner/MATLAB Function1'
//  '<S16>'  : 'TP/Target_planner/MATLAB Function2'
//  '<S17>'  : 'TP/Target_planner/Target_Point_Decision'
//  '<S18>'  : 'TP/Target_planner/target_seg_id_search'

#endif                                 // RTW_HEADER_TP_h_

//
// File trailer for generated code.
//
// [EOF]
//
