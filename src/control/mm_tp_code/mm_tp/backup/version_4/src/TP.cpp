//
// File: TP.cpp
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
#include "TP.h"
#define NumBitsPerChar                 8U

extern real_T rt_atan2d_snf(real_T u0, real_T u1);
extern real_T rt_roundd_snf(real_T u);
extern real_T rt_powd_snf(real_T u0, real_T u1);
static int32_T div_nde_s32_floor(int32_T numerator, int32_T denominator);
extern "C" {
  extern real_T rtGetInf(void);
  extern real32_T rtGetInfF(void);
  extern real_T rtGetMinusInf(void);
  extern real32_T rtGetMinusInfF(void);
}                                      // extern "C"
  extern "C"
{
  extern real_T rtGetNaN(void);
  extern real32_T rtGetNaNF(void);
}                                      // extern "C"

//===========*
//  Constants *
// ===========
#define RT_PI                          3.14159265358979323846
#define RT_PIF                         3.1415927F
#define RT_LN_10                       2.30258509299404568402
#define RT_LN_10F                      2.3025851F
#define RT_LOG10E                      0.43429448190325182765
#define RT_LOG10EF                     0.43429449F
#define RT_E                           2.7182818284590452354
#define RT_EF                          2.7182817F

//
//  UNUSED_PARAMETER(x)
//    Used to specify that a function parameter (argument) is required but not
//    accessed by the function body.

#ifndef UNUSED_PARAMETER
# if defined(__LCC__)
#   define UNUSED_PARAMETER(x)                                   // do nothing
# else

//
//  This is the semi-ANSI standard way of indicating that an
//  unused function parameter is required.

#   define UNUSED_PARAMETER(x)         (void) (x)
# endif
#endif

extern "C" {
  extern real_T rtInf;
  extern real_T rtMinusInf;
  extern real_T rtNaN;
  extern real32_T rtInfF;
  extern real32_T rtMinusInfF;
  extern real32_T rtNaNF;
  extern void rt_InitInfAndNaN(size_t realSize);
  extern boolean_T rtIsInf(real_T value);
  extern boolean_T rtIsInfF(real32_T value);
  extern boolean_T rtIsNaN(real_T value);
  extern boolean_T rtIsNaNF(real32_T value);
  typedef struct {
    struct {
      uint32_T wordH;
      uint32_T wordL;
    } words;
  } BigEndianIEEEDouble;

  typedef struct {
    struct {
      uint32_T wordL;
      uint32_T wordH;
    } words;
  } LittleEndianIEEEDouble;

  typedef struct {
    union {
      real32_T wordLreal;
      uint32_T wordLuint;
    } wordL;
  } IEEESingle;
}                                      // extern "C"
  extern "C"
{
  real_T rtInf;
  real_T rtMinusInf;
  real_T rtNaN;
  real32_T rtInfF;
  real32_T rtMinusInfF;
  real32_T rtNaNF;
}

extern "C" {
  //
  // Initialize rtInf needed by the generated code.
  // Inf is initialized as non-signaling. Assumes IEEE.
  //
  real_T rtGetInf(void)
  {
    size_t bitsPerReal = sizeof(real_T) * (NumBitsPerChar);
    real_T inf = 0.0;
    if (bitsPerReal == 32U) {
      inf = rtGetInfF();
    } else {
      union {
        LittleEndianIEEEDouble bitVal;
        real_T fltVal;
      } tmpVal;

      tmpVal.bitVal.words.wordH = 0x7FF00000U;
      tmpVal.bitVal.words.wordL = 0x00000000U;
      inf = tmpVal.fltVal;
    }

    return inf;
  }

  //
  // Initialize rtInfF needed by the generated code.
  // Inf is initialized as non-signaling. Assumes IEEE.
  //
  real32_T rtGetInfF(void)
  {
    IEEESingle infF;
    infF.wordL.wordLuint = 0x7F800000U;
    return infF.wordL.wordLreal;
  }

  //
  // Initialize rtMinusInf needed by the generated code.
  // Inf is initialized as non-signaling. Assumes IEEE.
  //
  real_T rtGetMinusInf(void)
  {
    size_t bitsPerReal = sizeof(real_T) * (NumBitsPerChar);
    real_T minf = 0.0;
    if (bitsPerReal == 32U) {
      minf = rtGetMinusInfF();
    } else {
      union {
        LittleEndianIEEEDouble bitVal;
        real_T fltVal;
      } tmpVal;

      tmpVal.bitVal.words.wordH = 0xFFF00000U;
      tmpVal.bitVal.words.wordL = 0x00000000U;
      minf = tmpVal.fltVal;
    }

    return minf;
  }

  //
  // Initialize rtMinusInfF needed by the generated code.
  // Inf is initialized as non-signaling. Assumes IEEE.
  //
  real32_T rtGetMinusInfF(void)
  {
    IEEESingle minfF;
    minfF.wordL.wordLuint = 0xFF800000U;
    return minfF.wordL.wordLreal;
  }
}
  extern "C"
{
  //
  // Initialize rtNaN needed by the generated code.
  // NaN is initialized as non-signaling. Assumes IEEE.
  //
  real_T rtGetNaN(void)
  {
    size_t bitsPerReal = sizeof(real_T) * (NumBitsPerChar);
    real_T nan = 0.0;
    if (bitsPerReal == 32U) {
      nan = rtGetNaNF();
    } else {
      union {
        LittleEndianIEEEDouble bitVal;
        real_T fltVal;
      } tmpVal;

      tmpVal.bitVal.words.wordH = 0xFFF80000U;
      tmpVal.bitVal.words.wordL = 0x00000000U;
      nan = tmpVal.fltVal;
    }

    return nan;
  }

  //
  // Initialize rtNaNF needed by the generated code.
  // NaN is initialized as non-signaling. Assumes IEEE.
  //
  real32_T rtGetNaNF(void)
  {
    IEEESingle nanF = { { 0 } };

    nanF.wordL.wordLuint = 0xFFC00000U;
    return nanF.wordL.wordLreal;
  }
}

extern "C" {
  //
  // Initialize the rtInf, rtMinusInf, and rtNaN needed by the
  // generated code. NaN is initialized as non-signaling. Assumes IEEE.
  //
  void rt_InitInfAndNaN(size_t realSize)
  {
    (void) (realSize);
    rtNaN = rtGetNaN();
    rtNaNF = rtGetNaNF();
    rtInf = rtGetInf();
    rtInfF = rtGetInfF();
    rtMinusInf = rtGetMinusInf();
    rtMinusInfF = rtGetMinusInfF();
  }

  // Test if value is infinite
  boolean_T rtIsInf(real_T value)
  {
    return (boolean_T)((value==rtInf || value==rtMinusInf) ? 1U : 0U);
  }

  // Test if single-precision value is infinite
  boolean_T rtIsInfF(real32_T value)
  {
    return (boolean_T)(((value)==rtInfF || (value)==rtMinusInfF) ? 1U : 0U);
  }

  // Test if value is not a number
  boolean_T rtIsNaN(real_T value)
  {
    return (boolean_T)((value!=value) ? 1U : 0U);
  }

  // Test if single-precision value is not a number
  boolean_T rtIsNaNF(real32_T value)
  {
    return (boolean_T)(((value!=value) ? 1U : 0U));
  }
}
  static int32_T div_nde_s32_floor(int32_T numerator, int32_T denominator)
{
  return (((numerator < 0) != (denominator < 0)) && (numerator % denominator !=
           0) ? -1 : 0) + numerator / denominator;
}

real_T rt_atan2d_snf(real_T u0, real_T u1)
{
  real_T y;
  int32_T u0_0;
  int32_T u1_0;
  if (rtIsNaN(u0) || rtIsNaN(u1)) {
    y = (rtNaN);
  } else if (rtIsInf(u0) && rtIsInf(u1)) {
    if (u0 > 0.0) {
      u0_0 = 1;
    } else {
      u0_0 = -1;
    }

    if (u1 > 0.0) {
      u1_0 = 1;
    } else {
      u1_0 = -1;
    }

    y = atan2((real_T)u0_0, (real_T)u1_0);
  } else if (u1 == 0.0) {
    if (u0 > 0.0) {
      y = RT_PI / 2.0;
    } else if (u0 < 0.0) {
      y = -(RT_PI / 2.0);
    } else {
      y = 0.0;
    }
  } else {
    y = atan2(u0, u1);
  }

  return y;
}

real_T rt_roundd_snf(real_T u)
{
  real_T y;
  if (std::abs(u) < 4.503599627370496E+15) {
    if (u >= 0.5) {
      y = std::floor(u + 0.5);
    } else if (u > -0.5) {
      y = u * 0.0;
    } else {
      y = std::ceil(u - 0.5);
    }
  } else {
    y = u;
  }

  return y;
}

// Function for MATLAB Function: '<S1>/EndPointDecision'
void TPModelClass::power_m(const real_T a_data[], const int32_T *a_size, real_T
  y_data[], int32_T *y_size)
{
  int32_T loop_ub;
  int16_T a_idx_0;
  a_idx_0 = (int16_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&rtDW.z1_data[0], &y_data[0], a_idx_0 * sizeof(real_T));
  }

  for (loop_ub = 0; loop_ub < a_idx_0; loop_ub++) {
    rtDW.z1_data[loop_ub] = a_data[loop_ub] * a_data[loop_ub];
  }

  *y_size = (int16_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&y_data[0], &rtDW.z1_data[0], a_idx_0 * sizeof(real_T));
  }
}

// Function for MATLAB Function: '<S1>/EndPointDecision2'
void TPModelClass::power_a(const real_T a_data[], const int32_T *a_size, real_T
  y_data[], int32_T *y_size)
{
  int32_T loop_ub;
  int16_T a_idx_0;
  a_idx_0 = (int16_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&rtDW.z1_data_f[0], &y_data[0], a_idx_0 * sizeof(real_T));
  }

  for (loop_ub = 0; loop_ub < a_idx_0; loop_ub++) {
    rtDW.z1_data_f[loop_ub] = a_data[loop_ub] * a_data[loop_ub];
  }

  *y_size = (int16_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&y_data[0], &rtDW.z1_data_f[0], a_idx_0 * sizeof(real_T));
  }
}

// Function for MATLAB Function: '<S1>/DangerousArea1'
void TPModelClass::abs_i(const real_T x[4], real_T y[4])
{
  y[0] = std::abs(x[0]);
  y[1] = std::abs(x[1]);
  y[2] = std::abs(x[2]);
  y[3] = std::abs(x[3]);
}

// Function for MATLAB Function: '<S1>/DangerousArea1'
void TPModelClass::power(const real_T a[4], real_T y[4])
{
  y[0] = a[0] * a[0];
  y[1] = a[1] * a[1];
  y[2] = a[2] * a[2];
  y[3] = a[3] * a[3];
}

real_T rt_powd_snf(real_T u0, real_T u1)
{
  real_T y;
  real_T tmp;
  real_T tmp_0;
  if (rtIsNaN(u0) || rtIsNaN(u1)) {
    y = (rtNaN);
  } else {
    tmp = std::abs(u0);
    tmp_0 = std::abs(u1);
    if (rtIsInf(u1)) {
      if (tmp == 1.0) {
        y = 1.0;
      } else if (tmp > 1.0) {
        if (u1 > 0.0) {
          y = (rtInf);
        } else {
          y = 0.0;
        }
      } else if (u1 > 0.0) {
        y = 0.0;
      } else {
        y = (rtInf);
      }
    } else if (tmp_0 == 0.0) {
      y = 1.0;
    } else if (tmp_0 == 1.0) {
      if (u1 > 0.0) {
        y = u0;
      } else {
        y = 1.0 / u0;
      }
    } else if (u1 == 2.0) {
      y = u0 * u0;
    } else if ((u1 == 0.5) && (u0 >= 0.0)) {
      y = std::sqrt(u0);
    } else if ((u0 < 0.0) && (u1 > std::floor(u1))) {
      y = (rtNaN);
    } else {
      y = pow(u0, u1);
    }
  }

  return y;
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
void TPModelClass::G2splines(real_T xa, real_T ya, real_T thetaa, real_T ka,
  real_T xb, real_T yb, real_T thetab, real_T kb, real_T path_length, real_T
  X_0[11], real_T Y[11], real_T XP[6], real_T YP[6], real_T K[11], real_T K_1[11],
  real_T *L_path)
{
  real_T x1;
  real_T x2;
  real_T x3;
  real_T x4;
  real_T x5;
  real_T b_y1;
  real_T y2;
  real_T y3;
  real_T y4;
  real_T y5;
  real_T X_1[11];
  real_T X_2[11];
  real_T Y_1[11];
  real_T Y_2[11];
  real_T b_z1[11];
  static const real_T s_a[11] = { 0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0 };

  static const real_T b[11] = { 0.0, 0.010000000000000002, 0.040000000000000008,
    0.090000000000000024, 0.16000000000000003, 0.25, 0.36, 0.48999999999999994,
    0.64000000000000012, 0.81, 1.0 };

  static const real_T b_b[11] = { 0.0, 0.0010000000000000002,
    0.0080000000000000019, 0.02700000000000001, 0.064000000000000015, 0.125,
    0.21599999999999997, 0.34299999999999992, 0.51200000000000012,
    0.72900000000000009, 1.0 };

  static const real_T c_b[11] = { 0.0, 0.00010000000000000002,
    0.0016000000000000003, 0.0081000000000000048, 0.025600000000000005, 0.0625,
    0.1296, 0.24009999999999992, 0.40960000000000008, 0.65610000000000013, 1.0 };

  int32_T i;
  real_T x2_tmp;
  real_T x2_tmp_0;
  real_T x3_tmp;
  real_T x3_tmp_0;
  real_T x3_tmp_tmp;
  real_T y3_tmp;
  real_T y3_tmp_0;
  y5 = std::cos(thetaa);
  x1 = path_length * y5;
  x2_tmp = path_length * path_length;
  x2_tmp_0 = std::sin(thetaa);
  y2 = x2_tmp * ka;
  x2 = (0.0 * y5 - y2 * x2_tmp_0) * 0.5;
  x5 = xb - xa;
  y4 = std::cos(thetab);
  x3_tmp_tmp = x2_tmp * 1.5 * ka;
  x4 = x3_tmp_tmp * x2_tmp_0;
  x3_tmp = std::sin(thetab);
  x3_tmp_0 = x2_tmp * 0.5;
  y3 = x3_tmp_0 * kb;
  b_y1 = y3 * x3_tmp;
  x3 = (((x5 * 10.0 - 6.0 * path_length * y5) - 4.0 * path_length * y4) + x4) -
    b_y1;
  x2_tmp *= kb;
  x4 = (((x5 * -15.0 + 8.0 * path_length * y5) + 7.0 * path_length * y4) - x4) +
    x2_tmp * x3_tmp;
  x3_tmp_0 *= ka;
  x5 = (((x5 * 6.0 - 3.0 * path_length * y5) - 3.0 * path_length * y4) +
        x3_tmp_0 * x2_tmp_0) - b_y1;
  XP[0] = xa;
  XP[1] = x1;
  XP[2] = x2;
  XP[3] = x3;
  XP[4] = x4;
  XP[5] = x5;
  b_y1 = path_length * x2_tmp_0;
  y2 = (y2 * y5 + 0.0 * x2_tmp_0) * 0.5;
  y3_tmp = yb - ya;
  x3_tmp_tmp *= y5;
  y3_tmp_0 = y3 * y4;
  y3 = (((y3_tmp * 10.0 - 6.0 * path_length * x2_tmp_0) - 4.0 * path_length *
         x3_tmp) - x3_tmp_tmp) + y3_tmp_0;
  y4 = (((y3_tmp * -15.0 + 8.0 * path_length * x2_tmp_0) + 7.0 * path_length *
         x3_tmp) + x3_tmp_tmp) - x2_tmp * y4;
  y5 = (((y3_tmp * 6.0 - 3.0 * path_length * x2_tmp_0) - 3.0 * path_length *
         x3_tmp) - x3_tmp_0 * y5) + y3_tmp_0;
  YP[0] = ya;
  YP[1] = b_y1;
  YP[2] = y2;
  YP[3] = y3;
  YP[4] = y4;
  YP[5] = y5;
  x2_tmp_0 = x2 * 2.0;
  x3_tmp = x3 * 3.0;
  x2_tmp = x4 * 4.0;
  x3_tmp_0 = x5 * 5.0;
  for (i = 0; i < 11; i++) {
    x3_tmp_tmp = s_a[i] * s_a[i];
    X_0[i] = ((((x1 * s_a[i] + xa) + x3_tmp_tmp * x2) + x3 * rt_powd_snf(s_a[i],
                3.0)) + x4 * rt_powd_snf(s_a[i], 4.0)) + x5 * rt_powd_snf(s_a[i],
      5.0);
    X_1[i] = (((x2_tmp_0 * s_a[i] + x1) + x3_tmp * b[i]) + x2_tmp * b_b[i]) +
      x3_tmp_0 * c_b[i];
    Y[i] = ((((b_y1 * s_a[i] + ya) + y2 * x3_tmp_tmp) + y3 * rt_powd_snf(s_a[i],
              3.0)) + y4 * rt_powd_snf(s_a[i], 4.0)) + y5 * rt_powd_snf(s_a[i],
      5.0);
  }

  x1 = x3 * 6.0;
  x2_tmp_0 = x4 * 12.0;
  x3_tmp = x5 * 20.0;
  for (i = 0; i < 11; i++) {
    X_2[i] = ((x2 * 2.0 + x1 * s_a[i]) + x2_tmp_0 * b[i]) + x3_tmp * b_b[i];
  }

  x2 = x4 * 24.0;
  x5 *= 60.0;
  x4 = y2 * 2.0;
  x1 = y3 * 3.0;
  x2_tmp_0 = y4 * 4.0;
  x3_tmp = y5 * 5.0;
  for (i = 0; i < 11; i++) {
    Y_1[i] = (((x4 * s_a[i] + b_y1) + x1 * b[i]) + x2_tmp_0 * b_b[i]) + x3_tmp *
      c_b[i];
  }

  b_y1 = y3 * 6.0;
  x4 = y4 * 12.0;
  x1 = y5 * 20.0;
  for (i = 0; i < 11; i++) {
    Y_2[i] = ((y2 * 2.0 + b_y1 * s_a[i]) + x4 * b[i]) + x1 * b_b[i];
  }

  y2 = y4 * 24.0;
  y5 *= 60.0;
  for (i = 0; i < 11; i++) {
    K[i] = (X_1[i] * Y_2[i] - X_2[i] * Y_1[i]) / rt_powd_snf(X_1[i] * X_1[i] +
      Y_1[i] * Y_1[i], 1.5);
    K_1[i] = rt_powd_snf(X_1[i] * X_1[i] + Y_1[i] * Y_1[i], 1.5);
    b_z1[i] = std::sqrt(X_1[i] * X_1[i] + Y_1[i] * Y_1[i]);
  }

  y3 *= 6.0;
  x3 *= 6.0;
  for (i = 0; i < 11; i++) {
    x1 = X_2[i] * Y_2[i];
    K_1[i] = ((((((y2 * s_a[i] + y3) + y5 * b[i]) * X_1[i] + x1) - ((x2 * s_a[i]
      + x3) + x5 * b[i]) * Y_1[i]) - x1) * K_1[i] - (X_1[i] * Y_2[i] - X_2[i] *
               Y_1[i]) * 1.5 * (2.0 * X_1[i] * X_2[i] + 2.0 * Y_1[i] * Y_2[i]) *
              b_z1[i]) / rt_powd_snf(X_1[i] * X_1[i] + Y_1[i] * Y_1[i], 3.0);
  }

  *L_path = path_length;
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
void TPModelClass::power_bv(const real_T a[143], real_T y[143])
{
  int32_T k;
  for (k = 0; k < 143; k++) {
    y[k] = a[k] * a[k];
  }
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
void TPModelClass::power_bvw(const real_T a[143], real_T y[143])
{
  int32_T k;
  for (k = 0; k < 143; k++) {
    y[k] = std::sqrt(a[k]);
  }
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
real_T TPModelClass::std(const real_T x[13])
{
  real_T y;
  real_T absdiff[13];
  real_T xbar;
  int32_T b_k;
  real_T t;
  xbar = x[0];
  for (b_k = 0; b_k < 12; b_k++) {
    xbar += x[b_k + 1];
  }

  xbar /= 13.0;
  for (b_k = 0; b_k < 13; b_k++) {
    absdiff[b_k] = std::abs(x[b_k] - xbar);
  }

  y = 0.0;
  xbar = 3.3121686421112381E-170;
  for (b_k = 0; b_k < 13; b_k++) {
    if (absdiff[b_k] > xbar) {
      t = xbar / absdiff[b_k];
      y = y * t * t + 1.0;
      xbar = absdiff[b_k];
    } else {
      t = absdiff[b_k] / xbar;
      y += t * t;
    }
  }

  y = xbar * std::sqrt(y);
  y /= 3.4641016151377544;
  return y;
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
void TPModelClass::power_bvwt(const real_T a[13], real_T y[13])
{
  int32_T k;
  for (k = 0; k < 13; k++) {
    y[k] = a[k] * a[k];
  }
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
void TPModelClass::exp_n(real_T x[13])
{
  int32_T k;
  for (k = 0; k < 13; k++) {
    x[k] = std::exp(x[k]);
  }
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
real_T TPModelClass::sum(const real_T x[13])
{
  real_T y;
  int32_T k;
  y = x[0];
  for (k = 0; k < 12; k++) {
    y += x[k + 1];
  }

  return y;
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
void TPModelClass::power_b(const real_T a[11], real_T y[11])
{
  int32_T k;
  for (k = 0; k < 11; k++) {
    y[k] = a[k] * a[k];
  }
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
void TPModelClass::sqrt_f(real_T x[11])
{
  int32_T k;
  for (k = 0; k < 11; k++) {
    x[k] = std::sqrt(x[k]);
  }
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
void TPModelClass::power_bvwts(const real_T a_data[], const int32_T a_size[2],
  real_T y_data[], int32_T y_size[2])
{
  int32_T nx;
  int32_T k;
  y_size[0] = (int8_T)a_size[0];
  y_size[1] = 2;
  nx = (int8_T)a_size[0] << 1;
  for (k = 0; k < nx; k++) {
    y_data[k] = a_data[k] * a_data[k];
  }
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
void TPModelClass::sum_p(const real_T x_data[], const int32_T x_size[2], real_T
  y_data[], int32_T y_size[2])
{
  int32_T xpageoffset;
  int32_T i;
  y_size[0] = 1;
  y_size[1] = (int8_T)x_size[1];
  for (i = 0; i < x_size[1]; i++) {
    xpageoffset = i << 1;
    y_data[i] = x_data[xpageoffset];
    y_data[i] += x_data[xpageoffset + 1];
  }
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
void TPModelClass::sqrt_fh(real_T x_data[], int32_T x_size[2])
{
  int32_T k;
  for (k = 0; k < x_size[1]; k++) {
    x_data[k] = std::sqrt(x_data[k]);
  }
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
real_T TPModelClass::mod(real_T x)
{
  real_T r;
  if (!rtIsNaN(x)) {
    r = std::fmod(x, 2.0);
    if (r == 0.0) {
      r = 0.0;
    }
  } else {
    r = (rtNaN);
  }

  return r;
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
void TPModelClass::point2safetylevel(const real_T X_data[], const int32_T
  X_size[2], const real_T Y_data[], const int32_T Y_size[2], const real_T
  Freespace[37500], real_T X_grid_data[], int32_T X_grid_size[2], real_T
  Y_grid_data[], int32_T Y_grid_size[2], real_T *safety_level)
{
  int32_T loop_ub;
  uint8_T b_X_idx_1;
  *safety_level = 0.0;
  b_X_idx_1 = (uint8_T)X_size[1];
  X_grid_size[0] = 1;
  X_grid_size[1] = b_X_idx_1;
  loop_ub = b_X_idx_1 - 1;
  if (0 <= loop_ub) {
    memset(&X_grid_data[0], 0, (loop_ub + 1) * sizeof(real_T));
  }

  b_X_idx_1 = (uint8_T)Y_size[1];
  Y_grid_size[0] = 1;
  Y_grid_size[1] = b_X_idx_1;
  loop_ub = b_X_idx_1 - 1;
  if (0 <= loop_ub) {
    memset(&Y_grid_data[0], 0, (loop_ub + 1) * sizeof(real_T));
  }

  for (loop_ub = 0; loop_ub < X_size[1]; loop_ub++) {
    if (X_data[loop_ub] > 0.0) {
      X_grid_data[loop_ub] = -std::ceil(X_data[loop_ub] / 0.2) + 175.0;
    } else {
      X_grid_data[loop_ub] = -std::floor(X_data[loop_ub] / 0.2) + 175.0;
    }

    if (Y_data[loop_ub] > 0.0) {
      Y_grid_data[loop_ub] = -std::ceil(Y_data[loop_ub] / 0.2) + 75.0;
    } else {
      Y_grid_data[loop_ub] = -std::floor(Y_data[loop_ub] / 0.2) + 75.0;
    }

    if (X_grid_data[loop_ub] < 1.0) {
      (*safety_level)++;
    } else if (X_grid_data[loop_ub] > 250.0) {
      (*safety_level)++;
    } else if (Y_grid_data[loop_ub] < 1.0) {
      (*safety_level)++;
    } else if (Y_grid_data[loop_ub] > 150.0) {
      (*safety_level)++;
    } else {
      *safety_level += Freespace[(((int32_T)Y_grid_data[loop_ub] - 1) * 250 +
        (int32_T)X_grid_data[loop_ub]) - 1];
    }
  }
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
void TPModelClass::FreespaceDetectCollision(const real_T Freespace[37500], const
  real_T XP[6], const real_T YP[6], const real_T Vehicle_state[3], real_T
  forward_length, real_T safe_range, const real_T Veh_size[2], const real_T
  Veh_CG[2], real_T *U_c, real_T *safety_level_all, real_T *forward_length_free)
{
  real_T XP_vehicle[6];
  real_T YP_vehicle[6];
  real_T rotate_ang;
  real_T u_0[200];
  real_T u_i;
  real_T X_ui;
  real_T Y_ui;
  real_T X_ui_1;
  real_T Y_ui_1;
  real_T XY_difflen;
  real_T Path_vehFLX_j;
  real_T Path_vehFLY_j;
  real_T Path_vehFRX_j;
  real_T Path_vehFRY_j;
  real_T Path_vehRLX_j;
  real_T Path_vehRLY_j;
  real_T Path_vehRRX_j;
  real_T Path_vehRRY_j;
  real_T h_j_0[200];
  real_T h_m_j;
  real_T m;
  real_T h_j_data[200];
  real_T v_j_data[200];
  real_T Y_FL2FR_j_grid_data[200];
  int32_T j;
  int32_T b_m;
  int32_T i;
  real_T Path_vehFLX_j_data[200];
  real_T Path_vehFLY_j_data[200];
  int32_T loop_ub;
  int32_T loop_ub_0;
  int32_T X_FL2FR_j_grid_size[2];
  int32_T Y_FL2FR_j_grid_size[2];
  int32_T Path_vehFRX_j_size[2];
  int32_T Path_vehFRY_j_size[2];
  int32_T Path_vehFLX_j_size[2];
  int32_T Path_vehFLY_j_size[2];
  int32_T Path_vehRLX_j_size[2];
  int32_T Path_vehRLY_j_size[2];
  int32_T Path_vehFLX_j_size_0[2];
  int32_T Path_vehFLY_j_size_0[2];
  real_T Path_vehFLX_j_tmp_tmp;
  real_T Path_vehRLX_j_tmp;
  boolean_T exitg1;
  for (i = 0; i < 6; i++) {
    XP_vehicle[i] = 0.0;
    YP_vehicle[i] = 0.0;
  }

  u_i = XP[0] - Vehicle_state[0];
  X_ui = YP[0] - Vehicle_state[1];
  rotate_ang = std::sin(-Vehicle_state[2]);
  Y_ui = std::cos(-Vehicle_state[2]);
  XP_vehicle[0] = Y_ui * u_i + -rotate_ang * X_ui;
  YP_vehicle[0] = rotate_ang * u_i + Y_ui * X_ui;
  for (i = 0; i < 5; i++) {
    XP_vehicle[i + 1] = XP[i + 1] * Y_ui + YP[i + 1] * -std::sin(-Vehicle_state
      [2]);
    YP_vehicle[i + 1] = XP[i + 1] * rotate_ang + YP[i + 1] * Y_ui;
  }

  for (b_m = 0; b_m < 200; b_m++) {
    u_0[b_m] = 1.0;
  }

  u_i = 0.4 / forward_length;
  X_ui = std::ceil(1.0 / u_i);
  rotate_ang = 1.0;
  for (i = 0; i < (int32_T)X_ui; i++) {
    rotate_ang = 1.0 + (real_T)i;
    u_0[i] = ((1.0 + (real_T)i) - 1.0) * u_i;
  }

  u_i = safe_range / 0.2;
  *U_c = 1.0;
  *safety_level_all = 0.0;
  i = 0;
  exitg1 = false;
  while ((!exitg1) && (i <= (int32_T)(rotate_ang + 1.0) - 1)) {
    if (1 + i < (int32_T)(rotate_ang + 1.0)) {
      X_ui = ((((XP_vehicle[1] * u_0[i] + XP_vehicle[0]) + u_0[i] * u_0[i] *
                XP_vehicle[2]) + XP_vehicle[3] * rt_powd_snf(u_0[i], 3.0)) +
              XP_vehicle[4] * rt_powd_snf(u_0[i], 4.0)) + XP_vehicle[5] *
        rt_powd_snf(u_0[i], 5.0);
      Y_ui = ((((YP_vehicle[1] * u_0[i] + YP_vehicle[0]) + u_0[i] * u_0[i] *
                YP_vehicle[2]) + YP_vehicle[3] * rt_powd_snf(u_0[i], 3.0)) +
              YP_vehicle[4] * rt_powd_snf(u_0[i], 4.0)) + YP_vehicle[5] *
        rt_powd_snf(u_0[i], 5.0);
      Y_ui_1 = u_0[i + 1] * u_0[i + 1];
      X_ui_1 = ((((u_0[i + 1] * XP_vehicle[1] + XP_vehicle[0]) + Y_ui_1 *
                  XP_vehicle[2]) + rt_powd_snf(u_0[i + 1], 3.0) * XP_vehicle[3])
                + rt_powd_snf(u_0[i + 1], 4.0) * XP_vehicle[4]) + rt_powd_snf
        (u_0[i + 1], 5.0) * XP_vehicle[5];
      Y_ui_1 = ((((u_0[i + 1] * YP_vehicle[1] + YP_vehicle[0]) + Y_ui_1 *
                  YP_vehicle[2]) + rt_powd_snf(u_0[i + 1], 3.0) * YP_vehicle[3])
                + rt_powd_snf(u_0[i + 1], 4.0) * YP_vehicle[4]) + rt_powd_snf
        (u_0[i + 1], 5.0) * YP_vehicle[5];
    } else {
      Y_ui = u_0[i - 1] * u_0[i - 1];
      X_ui = ((((u_0[i - 1] * XP_vehicle[1] + XP_vehicle[0]) + Y_ui *
                XP_vehicle[2]) + rt_powd_snf(u_0[i - 1], 3.0) * XP_vehicle[3]) +
              rt_powd_snf(u_0[i - 1], 4.0) * XP_vehicle[4]) + rt_powd_snf(u_0[i
        - 1], 5.0) * XP_vehicle[5];
      Y_ui = ((((u_0[i - 1] * YP_vehicle[1] + YP_vehicle[0]) + Y_ui *
                YP_vehicle[2]) + rt_powd_snf(u_0[i - 1], 3.0) * YP_vehicle[3]) +
              rt_powd_snf(u_0[i - 1], 4.0) * YP_vehicle[4]) + rt_powd_snf(u_0[i
        - 1], 5.0) * YP_vehicle[5];
      Y_ui_1 = u_0[i] * u_0[i];
      X_ui_1 = ((((XP_vehicle[1] * u_0[i] + XP_vehicle[0]) + Y_ui_1 *
                  XP_vehicle[2]) + XP_vehicle[3] * rt_powd_snf(u_0[i], 3.0)) +
                XP_vehicle[4] * rt_powd_snf(u_0[i], 4.0)) + XP_vehicle[5] *
        rt_powd_snf(u_0[i], 5.0);
      Y_ui_1 = ((((YP_vehicle[1] * u_0[i] + YP_vehicle[0]) + Y_ui_1 *
                  YP_vehicle[2]) + YP_vehicle[3] * rt_powd_snf(u_0[i], 3.0)) +
                YP_vehicle[4] * rt_powd_snf(u_0[i], 4.0)) + YP_vehicle[5] *
        rt_powd_snf(u_0[i], 5.0);
    }

    X_ui_1 -= X_ui;
    Y_ui_1 -= Y_ui;
    XY_difflen = std::sqrt(X_ui_1 * X_ui_1 + Y_ui_1 * Y_ui_1);
    X_ui_1 /= XY_difflen;
    Y_ui_1 /= XY_difflen;
    XY_difflen = 0.0;
    for (j = 0; j < (int32_T)(u_i + 3.0); j++) {
      h_m_j = (-2.0 + (real_T)j) * 0.2;
      Path_vehRRX_j = h_m_j + Veh_size[0];
      Path_vehFLY_j = Path_vehRRX_j * Veh_CG[0];
      Path_vehFLX_j_tmp_tmp = h_m_j + Veh_size[1];
      Path_vehFRY_j = Path_vehFLX_j_tmp_tmp * Veh_CG[1];
      Path_vehFRX_j = Path_vehFRY_j * X_ui_1;
      Path_vehRLX_j = Path_vehFLY_j * -Y_ui_1 + X_ui;
      Path_vehFLX_j = Path_vehRLX_j + Path_vehFRX_j;
      Path_vehFRY_j *= Y_ui_1;
      Path_vehRLY_j = Path_vehFLY_j * X_ui_1 + Y_ui;
      Path_vehFLY_j = Path_vehRLY_j + Path_vehFRY_j;
      Path_vehRRY_j = Path_vehRRX_j * (1.0 - Veh_CG[0]);
      Path_vehRRX_j = X_ui - Path_vehRRY_j * -Y_ui_1;
      Path_vehFRX_j += Path_vehRRX_j;
      Path_vehRRY_j = Y_ui - Path_vehRRY_j * X_ui_1;
      Path_vehFRY_j += Path_vehRRY_j;
      Path_vehRLX_j_tmp = Path_vehFLX_j_tmp_tmp * (1.0 - Veh_CG[1]);
      Path_vehFLX_j_tmp_tmp = Path_vehRLX_j_tmp * X_ui_1;
      Path_vehRLX_j -= Path_vehFLX_j_tmp_tmp;
      Path_vehRLX_j_tmp *= Y_ui_1;
      Path_vehRLY_j -= Path_vehRLX_j_tmp;
      Path_vehRRX_j -= Path_vehFLX_j_tmp_tmp;
      Path_vehRRY_j -= Path_vehRLX_j_tmp;
      for (b_m = 0; b_m < 200; b_m++) {
        h_j_0[b_m] = 1.0;
      }

      Path_vehFLX_j_tmp_tmp = h_m_j * 2.0;
      h_m_j = 0.2 / (Path_vehFLX_j_tmp_tmp + Veh_size[0]);
      Path_vehRLX_j_tmp = std::ceil(1.0 / h_m_j);
      m = 1.0;
      for (b_m = 0; b_m < (int32_T)Path_vehRLX_j_tmp; b_m++) {
        m = 1.0 + (real_T)b_m;
        h_j_0[b_m] = ((1.0 + (real_T)b_m) - 1.0) * h_m_j;
      }

      loop_ub = (int32_T)(m + 1.0);
      if (0 <= loop_ub - 1) {
        memcpy(&h_j_data[0], &h_j_0[0], loop_ub * sizeof(real_T));
      }

      for (b_m = 0; b_m < 200; b_m++) {
        h_j_0[b_m] = 1.0;
      }

      h_m_j = 0.2 / (Path_vehFLX_j_tmp_tmp + Veh_size[1]);
      Path_vehRLX_j_tmp = std::ceil(1.0 / h_m_j);
      m = 1.0;
      for (b_m = 0; b_m < (int32_T)Path_vehRLX_j_tmp; b_m++) {
        m = 1.0 + (real_T)b_m;
        h_j_0[b_m] = ((1.0 + (real_T)b_m) - 1.0) * h_m_j;
      }

      loop_ub_0 = (int32_T)(m + 1.0);
      if (0 <= loop_ub_0 - 1) {
        memcpy(&v_j_data[0], &h_j_0[0], loop_ub_0 * sizeof(real_T));
      }

      Path_vehFLX_j_size_0[0] = 1;
      Path_vehFLX_j_size_0[1] = loop_ub;
      h_m_j = Path_vehFRX_j - Path_vehFLX_j;
      for (b_m = 0; b_m < loop_ub; b_m++) {
        Path_vehFLX_j_data[b_m] = h_m_j * h_j_data[b_m] + Path_vehFLX_j;
      }

      Path_vehFLY_j_size_0[0] = 1;
      Path_vehFLY_j_size_0[1] = loop_ub;
      h_m_j = Path_vehFRY_j - Path_vehFLY_j;
      for (b_m = 0; b_m < loop_ub; b_m++) {
        Path_vehFLY_j_data[b_m] = h_m_j * h_j_data[b_m] + Path_vehFLY_j;
      }

      point2safetylevel(Path_vehFLX_j_data, Path_vehFLX_j_size_0,
                        Path_vehFLY_j_data, Path_vehFLY_j_size_0, Freespace,
                        h_j_0, X_FL2FR_j_grid_size, Y_FL2FR_j_grid_data,
                        Y_FL2FR_j_grid_size, &h_m_j);
      Path_vehRLX_j_size[0] = 1;
      Path_vehRLX_j_size[1] = loop_ub;
      Path_vehFLX_j_tmp_tmp = Path_vehRRX_j - Path_vehRLX_j;
      for (b_m = 0; b_m < loop_ub; b_m++) {
        Path_vehFLX_j_data[b_m] = Path_vehFLX_j_tmp_tmp * h_j_data[b_m] +
          Path_vehRLX_j;
      }

      Path_vehRLY_j_size[0] = 1;
      Path_vehRLY_j_size[1] = loop_ub;
      Path_vehFLX_j_tmp_tmp = Path_vehRRY_j - Path_vehRLY_j;
      for (b_m = 0; b_m < loop_ub; b_m++) {
        Path_vehFLY_j_data[b_m] = Path_vehFLX_j_tmp_tmp * h_j_data[b_m] +
          Path_vehRLY_j;
      }

      point2safetylevel(Path_vehFLX_j_data, Path_vehRLX_j_size,
                        Path_vehFLY_j_data, Path_vehRLY_j_size, Freespace,
                        h_j_data, Y_FL2FR_j_grid_size, h_j_0,
                        X_FL2FR_j_grid_size, &Path_vehRLX_j_tmp);
      Path_vehFLX_j_size[0] = 1;
      Path_vehFLX_j_size[1] = loop_ub_0;
      Path_vehRLX_j -= Path_vehFLX_j;
      for (b_m = 0; b_m < loop_ub_0; b_m++) {
        Path_vehFLX_j_data[b_m] = Path_vehRLX_j * v_j_data[b_m] + Path_vehFLX_j;
      }

      Path_vehFLY_j_size[0] = 1;
      Path_vehFLY_j_size[1] = loop_ub_0;
      Path_vehRLY_j -= Path_vehFLY_j;
      for (b_m = 0; b_m < loop_ub_0; b_m++) {
        Path_vehFLY_j_data[b_m] = Path_vehRLY_j * v_j_data[b_m] + Path_vehFLY_j;
      }

      point2safetylevel(Path_vehFLX_j_data, Path_vehFLX_j_size,
                        Path_vehFLY_j_data, Path_vehFLY_j_size, Freespace,
                        h_j_data, Y_FL2FR_j_grid_size, h_j_0,
                        X_FL2FR_j_grid_size, &m);
      Path_vehFRX_j_size[0] = 1;
      Path_vehFRX_j_size[1] = loop_ub_0;
      Path_vehRRX_j -= Path_vehFRX_j;
      for (b_m = 0; b_m < loop_ub_0; b_m++) {
        Path_vehFLX_j_data[b_m] = Path_vehRRX_j * v_j_data[b_m] + Path_vehFRX_j;
      }

      Path_vehFRY_j_size[0] = 1;
      Path_vehFRY_j_size[1] = loop_ub_0;
      Path_vehRRY_j -= Path_vehFRY_j;
      for (b_m = 0; b_m < loop_ub_0; b_m++) {
        Path_vehFLY_j_data[b_m] = Path_vehRRY_j * v_j_data[b_m] + Path_vehFRY_j;
      }

      point2safetylevel(Path_vehFLX_j_data, Path_vehFRX_j_size,
                        Path_vehFLY_j_data, Path_vehFRY_j_size, Freespace,
                        h_j_data, Y_FL2FR_j_grid_size, h_j_0,
                        X_FL2FR_j_grid_size, &Path_vehFLX_j);
      XY_difflen = (((XY_difflen + h_m_j) + Path_vehRLX_j_tmp) + m) +
        Path_vehFLX_j;
    }

    if (XY_difflen > 0.0) {
      *U_c = ((1.0 + (real_T)i) - 1.0) / ((real_T)(int32_T)(rotate_ang + 1.0) -
        1.0);
      *safety_level_all = XY_difflen;
      exitg1 = true;
    } else {
      i++;
    }
  }

  *forward_length_free = forward_length * *U_c;
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning'
void TPModelClass::abs_n(const real_T x[143], real_T y[143])
{
  int32_T k;
  for (k = 0; k < 143; k++) {
    y[k] = std::abs(x[k]);
  }
}

// Function for MATLAB Function: '<S1>/EndPointDecision1'
void TPModelClass::power_j(const real_T a_data[], const int32_T *a_size, real_T
  y_data[], int32_T *y_size)
{
  int32_T loop_ub;
  int16_T a_idx_0;
  a_idx_0 = (int16_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&rtDW.z1_data_c[0], &y_data[0], a_idx_0 * sizeof(real_T));
  }

  for (loop_ub = 0; loop_ub < a_idx_0; loop_ub++) {
    rtDW.z1_data_c[loop_ub] = a_data[loop_ub] * a_data[loop_ub];
  }

  *y_size = (int16_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&y_data[0], &rtDW.z1_data_c[0], a_idx_0 * sizeof(real_T));
  }
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning1'
void TPModelClass::G2splines_k(real_T xa, real_T ya, real_T thetaa, real_T ka,
  real_T xb, real_T yb, real_T thetab, real_T kb, real_T path_length, real_T
  X_1[11], real_T Y[11], real_T XP[6], real_T YP[6], real_T K[11], real_T K_1[11],
  real_T *L_path)
{
  real_T x1;
  real_T x2;
  real_T x3;
  real_T x4;
  real_T x5;
  real_T b_y1;
  real_T y2;
  real_T y3;
  real_T y4;
  real_T y5;
  real_T X_1_0[11];
  real_T X_2[11];
  real_T Y_1[11];
  real_T Y_2[11];
  real_T b_z1[11];
  static const real_T s_a[11] = { 0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0 };

  static const real_T b[11] = { 0.0, 0.010000000000000002, 0.040000000000000008,
    0.090000000000000024, 0.16000000000000003, 0.25, 0.36, 0.48999999999999994,
    0.64000000000000012, 0.81, 1.0 };

  static const real_T b_b[11] = { 0.0, 0.0010000000000000002,
    0.0080000000000000019, 0.02700000000000001, 0.064000000000000015, 0.125,
    0.21599999999999997, 0.34299999999999992, 0.51200000000000012,
    0.72900000000000009, 1.0 };

  static const real_T c_b[11] = { 0.0, 0.00010000000000000002,
    0.0016000000000000003, 0.0081000000000000048, 0.025600000000000005, 0.0625,
    0.1296, 0.24009999999999992, 0.40960000000000008, 0.65610000000000013, 1.0 };

  int32_T i;
  real_T x2_tmp;
  real_T x2_tmp_0;
  real_T x3_tmp;
  real_T x3_tmp_0;
  real_T x3_tmp_tmp;
  real_T y3_tmp;
  real_T y3_tmp_0;
  y5 = std::cos(thetaa);
  x1 = path_length * y5;
  x2_tmp = path_length * path_length;
  x2_tmp_0 = std::sin(thetaa);
  y2 = x2_tmp * ka;
  x2 = (0.0 * y5 - y2 * x2_tmp_0) * 0.5;
  x5 = xb - xa;
  y4 = std::cos(thetab);
  x3_tmp_tmp = x2_tmp * 1.5 * ka;
  x4 = x3_tmp_tmp * x2_tmp_0;
  x3_tmp = std::sin(thetab);
  x3_tmp_0 = x2_tmp * 0.5;
  y3 = x3_tmp_0 * kb;
  b_y1 = y3 * x3_tmp;
  x3 = (((x5 * 10.0 - 6.0 * path_length * y5) - 4.0 * path_length * y4) + x4) -
    b_y1;
  x2_tmp *= kb;
  x4 = (((x5 * -15.0 + 8.0 * path_length * y5) + 7.0 * path_length * y4) - x4) +
    x2_tmp * x3_tmp;
  x3_tmp_0 *= ka;
  x5 = (((x5 * 6.0 - 3.0 * path_length * y5) - 3.0 * path_length * y4) +
        x3_tmp_0 * x2_tmp_0) - b_y1;
  XP[0] = xa;
  XP[1] = x1;
  XP[2] = x2;
  XP[3] = x3;
  XP[4] = x4;
  XP[5] = x5;
  b_y1 = path_length * x2_tmp_0;
  y2 = (y2 * y5 + 0.0 * x2_tmp_0) * 0.5;
  y3_tmp = yb - ya;
  x3_tmp_tmp *= y5;
  y3_tmp_0 = y3 * y4;
  y3 = (((y3_tmp * 10.0 - 6.0 * path_length * x2_tmp_0) - 4.0 * path_length *
         x3_tmp) - x3_tmp_tmp) + y3_tmp_0;
  y4 = (((y3_tmp * -15.0 + 8.0 * path_length * x2_tmp_0) + 7.0 * path_length *
         x3_tmp) + x3_tmp_tmp) - x2_tmp * y4;
  y5 = (((y3_tmp * 6.0 - 3.0 * path_length * x2_tmp_0) - 3.0 * path_length *
         x3_tmp) - x3_tmp_0 * y5) + y3_tmp_0;
  YP[0] = ya;
  YP[1] = b_y1;
  YP[2] = y2;
  YP[3] = y3;
  YP[4] = y4;
  YP[5] = y5;
  x2_tmp_0 = x2 * 2.0;
  x3_tmp = x3 * 3.0;
  x2_tmp = x4 * 4.0;
  x3_tmp_0 = x5 * 5.0;
  for (i = 0; i < 11; i++) {
    x3_tmp_tmp = s_a[i] * s_a[i];
    X_1[i] = ((((x1 * s_a[i] + xa) + x3_tmp_tmp * x2) + x3 * rt_powd_snf(s_a[i],
                3.0)) + x4 * rt_powd_snf(s_a[i], 4.0)) + x5 * rt_powd_snf(s_a[i],
      5.0);
    X_1_0[i] = (((x2_tmp_0 * s_a[i] + x1) + x3_tmp * b[i]) + x2_tmp * b_b[i]) +
      x3_tmp_0 * c_b[i];
    Y[i] = ((((b_y1 * s_a[i] + ya) + y2 * x3_tmp_tmp) + y3 * rt_powd_snf(s_a[i],
              3.0)) + y4 * rt_powd_snf(s_a[i], 4.0)) + y5 * rt_powd_snf(s_a[i],
      5.0);
  }

  x1 = x3 * 6.0;
  x2_tmp_0 = x4 * 12.0;
  x3_tmp = x5 * 20.0;
  for (i = 0; i < 11; i++) {
    X_2[i] = ((x2 * 2.0 + x1 * s_a[i]) + x2_tmp_0 * b[i]) + x3_tmp * b_b[i];
  }

  x2 = x4 * 24.0;
  x5 *= 60.0;
  x4 = y2 * 2.0;
  x1 = y3 * 3.0;
  x2_tmp_0 = y4 * 4.0;
  x3_tmp = y5 * 5.0;
  for (i = 0; i < 11; i++) {
    Y_1[i] = (((x4 * s_a[i] + b_y1) + x1 * b[i]) + x2_tmp_0 * b_b[i]) + x3_tmp *
      c_b[i];
  }

  b_y1 = y3 * 6.0;
  x4 = y4 * 12.0;
  x1 = y5 * 20.0;
  for (i = 0; i < 11; i++) {
    Y_2[i] = ((y2 * 2.0 + b_y1 * s_a[i]) + x4 * b[i]) + x1 * b_b[i];
  }

  y2 = y4 * 24.0;
  y5 *= 60.0;
  for (i = 0; i < 11; i++) {
    K[i] = (X_1_0[i] * Y_2[i] - X_2[i] * Y_1[i]) / rt_powd_snf(X_1_0[i] *
      X_1_0[i] + Y_1[i] * Y_1[i], 1.5);
    K_1[i] = rt_powd_snf(X_1_0[i] * X_1_0[i] + Y_1[i] * Y_1[i], 1.5);
    b_z1[i] = std::sqrt(X_1_0[i] * X_1_0[i] + Y_1[i] * Y_1[i]);
  }

  y3 *= 6.0;
  x3 *= 6.0;
  for (i = 0; i < 11; i++) {
    x1 = X_2[i] * Y_2[i];
    K_1[i] = ((((((y2 * s_a[i] + y3) + y5 * b[i]) * X_1_0[i] + x1) - ((x2 *
      s_a[i] + x3) + x5 * b[i]) * Y_1[i]) - x1) * K_1[i] - (X_1_0[i] * Y_2[i] -
               X_2[i] * Y_1[i]) * 1.5 * (2.0 * X_1_0[i] * X_2[i] + 2.0 * Y_1[i] *
               Y_2[i]) * b_z1[i]) / rt_powd_snf(X_1_0[i] * X_1_0[i] + Y_1[i] *
      Y_1[i], 3.0);
  }

  *L_path = path_length;
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning1'
void TPModelClass::power_pcxfb(const real_T a_data[], const int32_T a_size[2],
  real_T y_data[], int32_T y_size[2])
{
  int32_T nx;
  int32_T k;
  y_size[0] = (int8_T)a_size[0];
  y_size[1] = 2;
  nx = (int8_T)a_size[0] << 1;
  for (k = 0; k < nx; k++) {
    y_data[k] = a_data[k] * a_data[k];
  }
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning1'
void TPModelClass::sum_h1(const real_T x_data[], const int32_T x_size[2], real_T
  y_data[], int32_T y_size[2])
{
  int32_T xpageoffset;
  int32_T i;
  y_size[0] = 1;
  y_size[1] = (int8_T)x_size[1];
  for (i = 0; i < x_size[1]; i++) {
    xpageoffset = i << 1;
    y_data[i] = x_data[xpageoffset];
    y_data[i] += x_data[xpageoffset + 1];
  }
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning1'
void TPModelClass::point2safetylevel_k(const real_T X_data[], const int32_T
  X_size[2], const real_T Y_data[], const int32_T Y_size[2], const real_T
  Freespace[37500], real_T X_grid_data[], int32_T X_grid_size[2], real_T
  Y_grid_data[], int32_T Y_grid_size[2], real_T *safety_level)
{
  int32_T loop_ub;
  uint8_T b_X_idx_1;
  *safety_level = 0.0;
  b_X_idx_1 = (uint8_T)X_size[1];
  X_grid_size[0] = 1;
  X_grid_size[1] = b_X_idx_1;
  loop_ub = b_X_idx_1 - 1;
  if (0 <= loop_ub) {
    memset(&X_grid_data[0], 0, (loop_ub + 1) * sizeof(real_T));
  }

  b_X_idx_1 = (uint8_T)Y_size[1];
  Y_grid_size[0] = 1;
  Y_grid_size[1] = b_X_idx_1;
  loop_ub = b_X_idx_1 - 1;
  if (0 <= loop_ub) {
    memset(&Y_grid_data[0], 0, (loop_ub + 1) * sizeof(real_T));
  }

  for (loop_ub = 0; loop_ub < X_size[1]; loop_ub++) {
    if (X_data[loop_ub] > 0.0) {
      X_grid_data[loop_ub] = -std::ceil(X_data[loop_ub] / 0.2) + 175.0;
    } else {
      X_grid_data[loop_ub] = -std::floor(X_data[loop_ub] / 0.2) + 175.0;
    }

    if (Y_data[loop_ub] > 0.0) {
      Y_grid_data[loop_ub] = -std::ceil(Y_data[loop_ub] / 0.2) + 75.0;
    } else {
      Y_grid_data[loop_ub] = -std::floor(Y_data[loop_ub] / 0.2) + 75.0;
    }

    if (X_grid_data[loop_ub] < 1.0) {
      (*safety_level)++;
    } else if (X_grid_data[loop_ub] > 250.0) {
      (*safety_level)++;
    } else if (Y_grid_data[loop_ub] < 1.0) {
      (*safety_level)++;
    } else if (Y_grid_data[loop_ub] > 150.0) {
      (*safety_level)++;
    } else {
      *safety_level += Freespace[(((int32_T)Y_grid_data[loop_ub] - 1) * 250 +
        (int32_T)X_grid_data[loop_ub]) - 1];
    }
  }
}

// Function for MATLAB Function: '<S1>/DynamicPathPlanning1'
void TPModelClass::FreespaceDetectCollision_m(const real_T Freespace[37500],
  const real_T XP[6], const real_T YP[6], const real_T Vehicle_state[3], real_T
  forward_length, real_T safe_range, const real_T Veh_size[2], const real_T
  Veh_CG[2], real_T *U_c, real_T *safety_level_all, real_T *forward_length_free)
{
  real_T XP_vehicle[6];
  real_T YP_vehicle[6];
  real_T rotate_ang;
  real_T u_0[200];
  real_T u_i;
  real_T X_ui;
  real_T Y_ui;
  real_T X_ui_1;
  real_T Y_ui_1;
  real_T XY_difflen;
  real_T Path_vehFLX_j;
  real_T Path_vehFLY_j;
  real_T Path_vehFRX_j;
  real_T Path_vehFRY_j;
  real_T Path_vehRLX_j;
  real_T Path_vehRLY_j;
  real_T Path_vehRRX_j;
  real_T Path_vehRRY_j;
  real_T h_j_0[200];
  real_T h_m_j;
  real_T m;
  real_T h_j_data[200];
  real_T v_j_data[200];
  real_T Y_FL2FR_j_grid_data[200];
  int32_T j;
  int32_T b_m;
  int32_T i;
  real_T Path_vehFLX_j_data[200];
  real_T Path_vehFLY_j_data[200];
  int32_T loop_ub;
  int32_T loop_ub_0;
  int32_T X_FL2FR_j_grid_size[2];
  int32_T Y_FL2FR_j_grid_size[2];
  int32_T Path_vehFRX_j_size[2];
  int32_T Path_vehFRY_j_size[2];
  int32_T Path_vehFLX_j_size[2];
  int32_T Path_vehFLY_j_size[2];
  int32_T Path_vehRLX_j_size[2];
  int32_T Path_vehRLY_j_size[2];
  int32_T Path_vehFLX_j_size_0[2];
  int32_T Path_vehFLY_j_size_0[2];
  real_T Path_vehFLX_j_tmp_tmp;
  real_T Path_vehRLX_j_tmp;
  boolean_T exitg1;
  for (i = 0; i < 6; i++) {
    XP_vehicle[i] = 0.0;
    YP_vehicle[i] = 0.0;
  }

  u_i = XP[0] - Vehicle_state[0];
  X_ui = YP[0] - Vehicle_state[1];
  rotate_ang = std::sin(-Vehicle_state[2]);
  Y_ui = std::cos(-Vehicle_state[2]);
  XP_vehicle[0] = Y_ui * u_i + -rotate_ang * X_ui;
  YP_vehicle[0] = rotate_ang * u_i + Y_ui * X_ui;
  for (i = 0; i < 5; i++) {
    XP_vehicle[i + 1] = XP[i + 1] * Y_ui + YP[i + 1] * -std::sin(-Vehicle_state
      [2]);
    YP_vehicle[i + 1] = XP[i + 1] * rotate_ang + YP[i + 1] * Y_ui;
  }

  for (b_m = 0; b_m < 200; b_m++) {
    u_0[b_m] = 1.0;
  }

  u_i = 0.4 / forward_length;
  X_ui = std::ceil(1.0 / u_i);
  rotate_ang = 1.0;
  for (i = 0; i < (int32_T)X_ui; i++) {
    rotate_ang = 1.0 + (real_T)i;
    u_0[i] = ((1.0 + (real_T)i) - 1.0) * u_i;
  }

  u_i = safe_range / 0.2;
  *U_c = 1.0;
  *safety_level_all = 0.0;
  i = 0;
  exitg1 = false;
  while ((!exitg1) && (i <= (int32_T)(rotate_ang + 1.0) - 1)) {
    if (1 + i < (int32_T)(rotate_ang + 1.0)) {
      X_ui = ((((XP_vehicle[1] * u_0[i] + XP_vehicle[0]) + u_0[i] * u_0[i] *
                XP_vehicle[2]) + XP_vehicle[3] * rt_powd_snf(u_0[i], 3.0)) +
              XP_vehicle[4] * rt_powd_snf(u_0[i], 4.0)) + XP_vehicle[5] *
        rt_powd_snf(u_0[i], 5.0);
      Y_ui = ((((YP_vehicle[1] * u_0[i] + YP_vehicle[0]) + u_0[i] * u_0[i] *
                YP_vehicle[2]) + YP_vehicle[3] * rt_powd_snf(u_0[i], 3.0)) +
              YP_vehicle[4] * rt_powd_snf(u_0[i], 4.0)) + YP_vehicle[5] *
        rt_powd_snf(u_0[i], 5.0);
      Y_ui_1 = u_0[i + 1] * u_0[i + 1];
      X_ui_1 = ((((u_0[i + 1] * XP_vehicle[1] + XP_vehicle[0]) + Y_ui_1 *
                  XP_vehicle[2]) + rt_powd_snf(u_0[i + 1], 3.0) * XP_vehicle[3])
                + rt_powd_snf(u_0[i + 1], 4.0) * XP_vehicle[4]) + rt_powd_snf
        (u_0[i + 1], 5.0) * XP_vehicle[5];
      Y_ui_1 = ((((u_0[i + 1] * YP_vehicle[1] + YP_vehicle[0]) + Y_ui_1 *
                  YP_vehicle[2]) + rt_powd_snf(u_0[i + 1], 3.0) * YP_vehicle[3])
                + rt_powd_snf(u_0[i + 1], 4.0) * YP_vehicle[4]) + rt_powd_snf
        (u_0[i + 1], 5.0) * YP_vehicle[5];
    } else {
      Y_ui = u_0[i - 1] * u_0[i - 1];
      X_ui = ((((u_0[i - 1] * XP_vehicle[1] + XP_vehicle[0]) + Y_ui *
                XP_vehicle[2]) + rt_powd_snf(u_0[i - 1], 3.0) * XP_vehicle[3]) +
              rt_powd_snf(u_0[i - 1], 4.0) * XP_vehicle[4]) + rt_powd_snf(u_0[i
        - 1], 5.0) * XP_vehicle[5];
      Y_ui = ((((u_0[i - 1] * YP_vehicle[1] + YP_vehicle[0]) + Y_ui *
                YP_vehicle[2]) + rt_powd_snf(u_0[i - 1], 3.0) * YP_vehicle[3]) +
              rt_powd_snf(u_0[i - 1], 4.0) * YP_vehicle[4]) + rt_powd_snf(u_0[i
        - 1], 5.0) * YP_vehicle[5];
      Y_ui_1 = u_0[i] * u_0[i];
      X_ui_1 = ((((XP_vehicle[1] * u_0[i] + XP_vehicle[0]) + Y_ui_1 *
                  XP_vehicle[2]) + XP_vehicle[3] * rt_powd_snf(u_0[i], 3.0)) +
                XP_vehicle[4] * rt_powd_snf(u_0[i], 4.0)) + XP_vehicle[5] *
        rt_powd_snf(u_0[i], 5.0);
      Y_ui_1 = ((((YP_vehicle[1] * u_0[i] + YP_vehicle[0]) + Y_ui_1 *
                  YP_vehicle[2]) + YP_vehicle[3] * rt_powd_snf(u_0[i], 3.0)) +
                YP_vehicle[4] * rt_powd_snf(u_0[i], 4.0)) + YP_vehicle[5] *
        rt_powd_snf(u_0[i], 5.0);
    }

    X_ui_1 -= X_ui;
    Y_ui_1 -= Y_ui;
    XY_difflen = std::sqrt(X_ui_1 * X_ui_1 + Y_ui_1 * Y_ui_1);
    X_ui_1 /= XY_difflen;
    Y_ui_1 /= XY_difflen;
    XY_difflen = 0.0;
    for (j = 0; j < (int32_T)(u_i + 3.0); j++) {
      h_m_j = (-2.0 + (real_T)j) * 0.2;
      Path_vehRRX_j = h_m_j + Veh_size[0];
      Path_vehFLY_j = Path_vehRRX_j * Veh_CG[0];
      Path_vehFLX_j_tmp_tmp = h_m_j + Veh_size[1];
      Path_vehFRY_j = Path_vehFLX_j_tmp_tmp * Veh_CG[1];
      Path_vehFRX_j = Path_vehFRY_j * X_ui_1;
      Path_vehRLX_j = Path_vehFLY_j * -Y_ui_1 + X_ui;
      Path_vehFLX_j = Path_vehRLX_j + Path_vehFRX_j;
      Path_vehFRY_j *= Y_ui_1;
      Path_vehRLY_j = Path_vehFLY_j * X_ui_1 + Y_ui;
      Path_vehFLY_j = Path_vehRLY_j + Path_vehFRY_j;
      Path_vehRRY_j = Path_vehRRX_j * (1.0 - Veh_CG[0]);
      Path_vehRRX_j = X_ui - Path_vehRRY_j * -Y_ui_1;
      Path_vehFRX_j += Path_vehRRX_j;
      Path_vehRRY_j = Y_ui - Path_vehRRY_j * X_ui_1;
      Path_vehFRY_j += Path_vehRRY_j;
      Path_vehRLX_j_tmp = Path_vehFLX_j_tmp_tmp * (1.0 - Veh_CG[1]);
      Path_vehFLX_j_tmp_tmp = Path_vehRLX_j_tmp * X_ui_1;
      Path_vehRLX_j -= Path_vehFLX_j_tmp_tmp;
      Path_vehRLX_j_tmp *= Y_ui_1;
      Path_vehRLY_j -= Path_vehRLX_j_tmp;
      Path_vehRRX_j -= Path_vehFLX_j_tmp_tmp;
      Path_vehRRY_j -= Path_vehRLX_j_tmp;
      for (b_m = 0; b_m < 200; b_m++) {
        h_j_0[b_m] = 1.0;
      }

      Path_vehFLX_j_tmp_tmp = h_m_j * 2.0;
      h_m_j = 0.2 / (Path_vehFLX_j_tmp_tmp + Veh_size[0]);
      Path_vehRLX_j_tmp = std::ceil(1.0 / h_m_j);
      m = 1.0;
      for (b_m = 0; b_m < (int32_T)Path_vehRLX_j_tmp; b_m++) {
        m = 1.0 + (real_T)b_m;
        h_j_0[b_m] = ((1.0 + (real_T)b_m) - 1.0) * h_m_j;
      }

      loop_ub = (int32_T)(m + 1.0);
      if (0 <= loop_ub - 1) {
        memcpy(&h_j_data[0], &h_j_0[0], loop_ub * sizeof(real_T));
      }

      for (b_m = 0; b_m < 200; b_m++) {
        h_j_0[b_m] = 1.0;
      }

      h_m_j = 0.2 / (Path_vehFLX_j_tmp_tmp + Veh_size[1]);
      Path_vehRLX_j_tmp = std::ceil(1.0 / h_m_j);
      m = 1.0;
      for (b_m = 0; b_m < (int32_T)Path_vehRLX_j_tmp; b_m++) {
        m = 1.0 + (real_T)b_m;
        h_j_0[b_m] = ((1.0 + (real_T)b_m) - 1.0) * h_m_j;
      }

      loop_ub_0 = (int32_T)(m + 1.0);
      if (0 <= loop_ub_0 - 1) {
        memcpy(&v_j_data[0], &h_j_0[0], loop_ub_0 * sizeof(real_T));
      }

      Path_vehFLX_j_size_0[0] = 1;
      Path_vehFLX_j_size_0[1] = loop_ub;
      h_m_j = Path_vehFRX_j - Path_vehFLX_j;
      for (b_m = 0; b_m < loop_ub; b_m++) {
        Path_vehFLX_j_data[b_m] = h_m_j * h_j_data[b_m] + Path_vehFLX_j;
      }

      Path_vehFLY_j_size_0[0] = 1;
      Path_vehFLY_j_size_0[1] = loop_ub;
      h_m_j = Path_vehFRY_j - Path_vehFLY_j;
      for (b_m = 0; b_m < loop_ub; b_m++) {
        Path_vehFLY_j_data[b_m] = h_m_j * h_j_data[b_m] + Path_vehFLY_j;
      }

      point2safetylevel_k(Path_vehFLX_j_data, Path_vehFLX_j_size_0,
                          Path_vehFLY_j_data, Path_vehFLY_j_size_0, Freespace,
                          h_j_0, X_FL2FR_j_grid_size, Y_FL2FR_j_grid_data,
                          Y_FL2FR_j_grid_size, &h_m_j);
      Path_vehRLX_j_size[0] = 1;
      Path_vehRLX_j_size[1] = loop_ub;
      Path_vehFLX_j_tmp_tmp = Path_vehRRX_j - Path_vehRLX_j;
      for (b_m = 0; b_m < loop_ub; b_m++) {
        Path_vehFLX_j_data[b_m] = Path_vehFLX_j_tmp_tmp * h_j_data[b_m] +
          Path_vehRLX_j;
      }

      Path_vehRLY_j_size[0] = 1;
      Path_vehRLY_j_size[1] = loop_ub;
      Path_vehFLX_j_tmp_tmp = Path_vehRRY_j - Path_vehRLY_j;
      for (b_m = 0; b_m < loop_ub; b_m++) {
        Path_vehFLY_j_data[b_m] = Path_vehFLX_j_tmp_tmp * h_j_data[b_m] +
          Path_vehRLY_j;
      }

      point2safetylevel_k(Path_vehFLX_j_data, Path_vehRLX_j_size,
                          Path_vehFLY_j_data, Path_vehRLY_j_size, Freespace,
                          h_j_data, Y_FL2FR_j_grid_size, h_j_0,
                          X_FL2FR_j_grid_size, &Path_vehRLX_j_tmp);
      Path_vehFLX_j_size[0] = 1;
      Path_vehFLX_j_size[1] = loop_ub_0;
      Path_vehRLX_j -= Path_vehFLX_j;
      for (b_m = 0; b_m < loop_ub_0; b_m++) {
        Path_vehFLX_j_data[b_m] = Path_vehRLX_j * v_j_data[b_m] + Path_vehFLX_j;
      }

      Path_vehFLY_j_size[0] = 1;
      Path_vehFLY_j_size[1] = loop_ub_0;
      Path_vehRLY_j -= Path_vehFLY_j;
      for (b_m = 0; b_m < loop_ub_0; b_m++) {
        Path_vehFLY_j_data[b_m] = Path_vehRLY_j * v_j_data[b_m] + Path_vehFLY_j;
      }

      point2safetylevel_k(Path_vehFLX_j_data, Path_vehFLX_j_size,
                          Path_vehFLY_j_data, Path_vehFLY_j_size, Freespace,
                          h_j_data, Y_FL2FR_j_grid_size, h_j_0,
                          X_FL2FR_j_grid_size, &m);
      Path_vehFRX_j_size[0] = 1;
      Path_vehFRX_j_size[1] = loop_ub_0;
      Path_vehRRX_j -= Path_vehFRX_j;
      for (b_m = 0; b_m < loop_ub_0; b_m++) {
        Path_vehFLX_j_data[b_m] = Path_vehRRX_j * v_j_data[b_m] + Path_vehFRX_j;
      }

      Path_vehFRY_j_size[0] = 1;
      Path_vehFRY_j_size[1] = loop_ub_0;
      Path_vehRRY_j -= Path_vehFRY_j;
      for (b_m = 0; b_m < loop_ub_0; b_m++) {
        Path_vehFLY_j_data[b_m] = Path_vehRRY_j * v_j_data[b_m] + Path_vehFRY_j;
      }

      point2safetylevel_k(Path_vehFLX_j_data, Path_vehFRX_j_size,
                          Path_vehFLY_j_data, Path_vehFRY_j_size, Freespace,
                          h_j_data, Y_FL2FR_j_grid_size, h_j_0,
                          X_FL2FR_j_grid_size, &Path_vehFLX_j);
      XY_difflen = (((XY_difflen + h_m_j) + Path_vehRLX_j_tmp) + m) +
        Path_vehFLX_j;
    }

    if (XY_difflen > 0.0) {
      *U_c = ((1.0 + (real_T)i) - 1.0) / ((real_T)(int32_T)(rotate_ang + 1.0) -
        1.0);
      *safety_level_all = XY_difflen;
      exitg1 = true;
    } else {
      i++;
    }
  }

  *forward_length_free = forward_length * *U_c;
}

// Model step function
void TPModelClass::step()
{
  int32_T c;
  int32_T Static_PathCycle;
  real_T total_length;
  int32_T end_ind_0;
  int32_T case_0;
  int32_T Forward_Static_Path_length;
  int16_T cb_data[1000];
  boolean_T varargin_1_data[1000];
  boolean_T ex;
  real_T End_x;
  real_T End_y;
  real_T count;
  real_T count_1;
  int32_T break_count;
  real_T target_k;
  real_T Length_1;
  real_T ang_1;
  real_T vehicle_heading;
  real_T Forward_Static_Path_id_0_data[113];
  int16_T cb_data_0[113];
  boolean_T varargin_1_data_0[113];
  real_T y;
  real_T OBXY_m[8];
  real_T D[4];
  real_T offset_5;
  real_T offset[13];
  real_T x_endpoint1;
  real_T y_endpoint1;
  real_T x_endpoint2;
  real_T x_endpoint3;
  real_T y_endpoint3;
  real_T x_endpoint4;
  real_T y_endpoint4;
  real_T x_endpoint5;
  real_T y_endpoint5;
  real_T x_endpoint6;
  real_T y_endpoint6;
  real_T x_endpoint8;
  real_T y_endpoint8;
  real_T x_endpoint9;
  real_T y_endpoint9;
  real_T x_endpoint10;
  real_T y_endpoint10;
  real_T x_endpoint11;
  real_T y_endpoint11;
  real_T x_endpoint12;
  real_T y_endpoint12;
  real_T x_endpoint13;
  real_T X_2[143];
  real_T Y[143];
  real_T K[143];
  real_T K_1[143];
  real_T Path_col[52];
  real_T OBXY_EL[400];
  real_T X_diff_0[130];
  real_T X_diff[143];
  real_T Y_diff[143];
  real_T XY_difflen[143];
  real_T Path_vehFLX[143];
  real_T Path_vehFLY[143];
  real_T Path_vehFRX[143];
  real_T Path_vehFRY[143];
  real_T Path_vehRLX[143];
  real_T Path_vehRLY[143];
  real_T Path_vehRRX[143];
  real_T Path_vehRRY[143];
  real_T proj_veh[16];
  real_T proj_ob[16];
  real_T minmax_veh[8];
  real_T minmax_obj[8];
  real_T Cobs_0[13];
  real_T Cobs[13];
  real_T Clane[13];
  real_T Cc_0[13];
  real_T LastPath_overlap_data[22];
  real_T Path_overlap_data[22];
  real_T Path_dis_data[121];
  real_T X1[11];
  real_T XP1[6];
  real_T YP1[6];
  real_T K1[11];
  real_T K_11[11];
  real_T X2[11];
  real_T Y2[11];
  real_T XP2[6];
  real_T YP2[6];
  real_T K2[11];
  real_T K_12[11];
  real_T X3[11];
  real_T Y3[11];
  real_T XP3[6];
  real_T YP3[6];
  real_T K3[11];
  real_T K_13[11];
  real_T X4[11];
  real_T Y4[11];
  real_T XP4[6];
  real_T YP4[6];
  real_T K4[11];
  real_T K_14[11];
  real_T X5[11];
  real_T Y5[11];
  real_T XP5[6];
  real_T YP5[6];
  real_T K5[11];
  real_T K_15[11];
  real_T X6[11];
  real_T Y6[11];
  real_T XP6[6];
  real_T YP6[6];
  real_T K6[11];
  real_T K_16[11];
  real_T X7[11];
  real_T Y7[11];
  real_T XP7[6];
  real_T YP7[6];
  real_T K7[11];
  real_T K_17[11];
  real_T X8[11];
  real_T Y8[11];
  real_T XP8[6];
  real_T YP8[6];
  real_T K8[11];
  real_T K_18[11];
  real_T X9[11];
  real_T Y9[11];
  real_T XP9[6];
  real_T YP9[6];
  real_T K9[11];
  real_T K_19[11];
  real_T X10[11];
  real_T Y10[11];
  real_T XP10[6];
  real_T YP10[6];
  real_T K10[11];
  real_T K_110[11];
  real_T X11[11];
  real_T Y11[11];
  real_T XP11[6];
  real_T YP11[6];
  real_T K11[11];
  real_T K_111[11];
  real_T X12[11];
  real_T Y12[11];
  real_T XP12[6];
  real_T YP12[6];
  real_T K12[11];
  real_T K_112[11];
  real_T X13[11];
  real_T Y13[11];
  real_T XP13[6];
  real_T YP13[6];
  real_T K13[11];
  real_T K_113[11];
  real_T b_Path_dis_data[11];
  int8_T p_data[9];
  int32_T t_data[1000];
  int32_T u_data[1000];
  int32_T v_data[1000];
  int32_T Forward_Static_Path_length_0;
  int32_T t_data_0[1000];
  int32_T u_data_0[1000];
  int32_T v_data_0[1000];
  real_T J_minvalue_diff;
  static const real_T a[11] = { 0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0 };

  real_T x_target;
  real_T rtb_Forward_length_final;
  real_T rtb_Look_ahead_time;
  real_T rtb_UnitDelay18;
  real_T rtb_TmpSignalConversionAtSFun_n[3];
  real_T rtb_H_x_out[4];
  real_T rtb_H_y_out[4];
  real_T rtb_J_out_a[13];
  real_T rtb_U_c_n[13];
  real_T rtb_safety_level_all_b[13];
  real_T rtb_forward_length_free_f[13];
  real_T rtb_forward_length_free[13];
  real_T rtb_forward_length_free_2[13];
  real_T rtb_V_boundingbox[400];
  real_T rtb_Forward_Static_Path_id[113];
  real_T rtb_XP_i[78];
  real_T rtb_YP_p[78];
  real_T rtb_XP[78];
  real_T rtb_YP[78];
  int32_T i;
  real_T tmp[4];
  real_T ang_1_0[4];
  real_T tmp_0[4];
  real_T LastPath_overlap_data_0[22];
  int32_T loop_ub;
  int16_T tmp_1;
  int32_T tmp_data[1000];
  real_T seg_id_data[226];
  real_T rtb_TmpSignalConversionAtSFun_1[8];
  real_T rtb_TmpSignalConversionAtSFun_2[8];
  real_T OBXY_EL_0[8];
  real_T b_Path_dis_data_0[121];
  int32_T J_minvalue_diff_0;
  int32_T J_minvalue_diff_1;
  int32_T J_minvalue_diff_2;
  int32_T Path_overlap_size[2];
  int32_T b_Path_dis_size[2];
  int32_T Path_overlap_size_0[2];
  int32_T LastPath_overlap_size[2];
  int32_T Path_overlap_size_1[2];
  int32_T LastPath_overlap_size_0[2];
  int32_T Path_overlap_size_2[2];
  int32_T LastPath_overlap_size_1[2];
  int32_T Path_overlap_size_3[2];
  int32_T LastPath_overlap_size_2[2];
  int32_T Path_RES_1_size_idx_0;
  int32_T xy_ends_POS_size_idx_0;
  real_T xy_end_point_idx_25;
  real_T xy_end_point_idx_2;
  real_T xy_end_point_idx_1;
  real_T xy_end_point_idx_0;
  int32_T Path_RES_0_size_idx_1;
  real_T End_x_tmp_tmp;
  real_T ang_1_tmp;
  real_T J_minvalue_diff_tmp;
  boolean_T exitg1;
  boolean_T exitg2;
  boolean_T exitg3;
  boolean_T guard1 = false;

  // MATLAB Function: '<S1>/MATLAB Function1' incorporates:
  //   Inport: '<Root>/Static_Path_0_i'
  //   Inport: '<Root>/Static_Path_0_length1_i'
  //   Inport: '<Root>/Static_Path_0_length2_i'

  if (1.0 > rtU.Static_Path_0_length1_i) {
    case_0 = 0;
  } else {
    case_0 = (int32_T)rtU.Static_Path_0_length1_i;
  }

  if (1.0 > rtU.Static_Path_0_length2_i) {
    c = 0;
  } else {
    c = (int32_T)rtU.Static_Path_0_length2_i;
  }

  rtDW.SFunction_DIMS2_g[0] = case_0;
  rtDW.SFunction_DIMS2_g[1] = c;
  for (i = 0; i < c; i++) {
    for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < case_0;
         Path_RES_0_size_idx_1++) {
      rtDW.Static_Path_0[Path_RES_0_size_idx_1 + rtDW.SFunction_DIMS2_g[0] * i] =
        rtU.Static_Path_0_i[1000 * i + Path_RES_0_size_idx_1];
    }
  }

  // End of MATLAB Function: '<S1>/MATLAB Function1'

  // MATLAB Function: '<S1>/MATLAB Function2' incorporates:
  //   Inport: '<Root>/ID_turn'
  //   Inport: '<Root>/Look_ahead_time_straight'
  //   Inport: '<Root>/Look_ahead_time_turn'
  //   Inport: '<Root>/seg_id_near_i'

  if ((rtU.seg_id_near_i >= rtU.ID_turn[0]) && (rtU.seg_id_near_i <=
       rtU.ID_turn[1])) {
    rtb_Look_ahead_time = rtU.Look_ahead_time_turn;
  } else if ((rtU.seg_id_near_i >= rtU.ID_turn[2]) && (rtU.seg_id_near_i <=
              rtU.ID_turn[3])) {
    rtb_Look_ahead_time = rtU.Look_ahead_time_turn;
  } else if ((rtU.seg_id_near_i >= rtU.ID_turn[4]) && (rtU.seg_id_near_i <=
              rtU.ID_turn[5])) {
    rtb_Look_ahead_time = rtU.Look_ahead_time_turn;
  } else if (rtU.seg_id_near_i >= rtU.ID_turn[6]) {
    if (rtU.seg_id_near_i <= rtU.ID_turn[7]) {
      rtb_Look_ahead_time = rtU.Look_ahead_time_turn;
    } else {
      rtb_Look_ahead_time = rtU.Look_ahead_time_straight;
    }
  } else {
    rtb_Look_ahead_time = rtU.Look_ahead_time_straight;
  }

  // End of MATLAB Function: '<S1>/MATLAB Function2'

  // MATLAB Function: '<S1>/Forward_Length_Decision1' incorporates:
  //   Inport: '<Root>/Look_ahead_S0'
  //   Inport: '<Root>/Speed_mps1'
  //   UnitDelay: '<S1>/Unit Delay14'
  //   UnitDelay: '<S1>/Unit Delay16'

  if (rtDW.UnitDelay14_DSTATE == 1.0) {
    rtb_Forward_length_final = rtDW.UnitDelay16_DSTATE;
  } else {
    rtb_Forward_length_final = rtU.Speed_mps1 * rtb_Look_ahead_time +
      rtU.Look_ahead_S0;
  }

  // End of MATLAB Function: '<S1>/Forward_Length_Decision1'

  // MATLAB Function: '<S1>/Forward_Seg' incorporates:
  //   Inport: '<Root>/Oi_near_i'
  //   Inport: '<Root>/seg_id_near_i'

  xy_ends_POS_size_idx_0 = rtDW.SFunction_DIMS2_g[0];
  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    rtDW.xy_ends_POS_data[i] = rtDW.Static_Path_0[i + rtDW.SFunction_DIMS2_g[0]];
  }

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    rtDW.xy_ends_POS_data[i + xy_ends_POS_size_idx_0] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS2_g[0] << 1) + i];
  }

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    rtDW.xy_ends_POS_data[i + (xy_ends_POS_size_idx_0 << 1)] =
      rtDW.Static_Path_0[rtDW.SFunction_DIMS2_g[0] * 3 + i];
  }

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    rtDW.xy_ends_POS_data[i + xy_ends_POS_size_idx_0 * 3] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS2_g[0] << 2) + i];
  }

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  if (0 <= loop_ub - 1) {
    memcpy(&rtDW.seg_id_data[0], &rtDW.Static_Path_0[0], loop_ub * sizeof(real_T));
  }

  if (rtDW.Static_Path_0[(rtDW.SFunction_DIMS2_g[0] * 3 +
                          rtDW.SFunction_DIMS2_g[0]) - 1] ==
      rtDW.Static_Path_0[rtDW.SFunction_DIMS2_g[0]]) {
    Static_PathCycle = (rtDW.Static_Path_0[((rtDW.SFunction_DIMS2_g[0] << 2) +
      rtDW.SFunction_DIMS2_g[0]) - 1] ==
                        rtDW.Static_Path_0[rtDW.SFunction_DIMS2_g[0] << 1]);
  } else {
    Static_PathCycle = 0;
  }

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    varargin_1_data[i] = (rtDW.seg_id_data[i] == rtU.seg_id_near_i);
  }

  Forward_Static_Path_length = 1;
  ex = varargin_1_data[0];
  for (case_0 = 2; case_0 <= rtDW.SFunction_DIMS2_g[0]; case_0++) {
    if ((int32_T)ex < (int32_T)varargin_1_data[case_0 - 1]) {
      ex = varargin_1_data[case_0 - 1];
      Forward_Static_Path_length = case_0;
    }
  }

  ang_1 = rtU.Oi_near_i[0] - rtDW.Static_Path_0[(rtDW.SFunction_DIMS2_g[0] * 3 +
    Forward_Static_Path_length) - 1];
  J_minvalue_diff = rtU.Oi_near_i[1] - rtDW.Static_Path_0
    [((rtDW.SFunction_DIMS2_g[0] << 2) + Forward_Static_Path_length) - 1];
  total_length = std::sqrt(ang_1 * ang_1 + J_minvalue_diff * J_minvalue_diff);
  end_ind_0 = Forward_Static_Path_length;
  case_0 = 0;
  break_count = 0;
  Forward_Static_Path_length_0 = 0;
  exitg1 = false;
  while ((!exitg1) && (Forward_Static_Path_length_0 <= rtDW.SFunction_DIMS2_g[0]
                       - 1)) {
    if (total_length > rtb_Forward_length_final) {
      break_count = end_ind_0;
      exitg1 = true;
    } else {
      i = Forward_Static_Path_length + Forward_Static_Path_length_0;
      Path_RES_0_size_idx_1 = i + 1;
      if (Path_RES_0_size_idx_1 <= rtDW.SFunction_DIMS2_g[0]) {
        total_length += rtDW.Static_Path_0[i + (rtDW.SFunction_DIMS2_g[0] << 3)];
        end_ind_0 = Path_RES_0_size_idx_1;
        case_0 = 1;
        Forward_Static_Path_length_0++;
      } else if (Static_PathCycle == 1) {
        i -= rtDW.SFunction_DIMS2_g[0];
        total_length += rtDW.Static_Path_0[i + (rtDW.SFunction_DIMS2_g[0] << 3)];
        end_ind_0 = i + 1;
        case_0 = 2;
        Forward_Static_Path_length_0++;
      } else {
        break_count = end_ind_0;
        case_0 = 3;
        exitg1 = true;
      }
    }
  }

  Forward_Static_Path_length_0 = rtDW.SFunction_DIMS2_g[0] - 1;
  if (0 <= Forward_Static_Path_length_0) {
    memset(&rtDW.Forward_Static_Path_id_0_data[0], 0,
           (Forward_Static_Path_length_0 + 1) * sizeof(real_T));
  }

  if ((case_0 == 1) || (case_0 == 0)) {
    if (Forward_Static_Path_length > break_count) {
      c = 0;
      case_0 = 0;
    } else {
      c = Forward_Static_Path_length - 1;
      case_0 = break_count;
    }

    Path_RES_0_size_idx_1 = case_0 - c;
    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i] = rtDW.xy_ends_POS_data[c + i];
    }

    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(c + i) + xy_ends_POS_size_idx_0];
    }

    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i + (Path_RES_0_size_idx_1 << 1)] =
        rtDW.xy_ends_POS_data[(c + i) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i + Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[(c + i) + xy_ends_POS_size_idx_0 * 3];
    }

    if (Forward_Static_Path_length > break_count) {
      Forward_Static_Path_length_0 = 1;
      case_0 = 0;
    } else {
      Forward_Static_Path_length_0 = Forward_Static_Path_length;
      case_0 = break_count;
    }

    loop_ub = case_0 - Forward_Static_Path_length_0;
    for (i = 0; i <= loop_ub; i++) {
      rtDW.Forward_Static_Path_id_0_data[i] = rtDW.seg_id_data
        [(Forward_Static_Path_length_0 + i) - 1];
    }

    if (Forward_Static_Path_length > break_count) {
      Forward_Static_Path_length = 1;
      break_count = 0;
    }

    Forward_Static_Path_length = (break_count - Forward_Static_Path_length) + 1;
  } else if (case_0 == 2) {
    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      case_0 = 0;
      Forward_Static_Path_length_0 = 0;
    } else {
      case_0 = Forward_Static_Path_length - 1;
      Forward_Static_Path_length_0 = rtDW.SFunction_DIMS2_g[0];
    }

    if (1 > break_count) {
      loop_ub = 0;
    } else {
      loop_ub = break_count;
    }

    Static_PathCycle = Forward_Static_Path_length_0 - case_0;
    Path_RES_0_size_idx_1 = Static_PathCycle + loop_ub;
    for (i = 0; i < Static_PathCycle; i++) {
      rtDW.Static_Path_ends_POS_data[i] = rtDW.xy_ends_POS_data[case_0 + i];
    }

    for (i = 0; i < Static_PathCycle; i++) {
      rtDW.Static_Path_ends_POS_data[i + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(case_0 + i) + xy_ends_POS_size_idx_0];
    }

    for (i = 0; i < Static_PathCycle; i++) {
      rtDW.Static_Path_ends_POS_data[i + (Path_RES_0_size_idx_1 << 1)] =
        rtDW.xy_ends_POS_data[(case_0 + i) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (i = 0; i < Static_PathCycle; i++) {
      rtDW.Static_Path_ends_POS_data[i + Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[(case_0 + i) + xy_ends_POS_size_idx_0 * 3];
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Static_Path_ends_POS_data[(i + Forward_Static_Path_length_0) - case_0]
        = rtDW.xy_ends_POS_data[i];
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Static_Path_ends_POS_data[((i + Forward_Static_Path_length_0) -
        case_0) + Path_RES_0_size_idx_1] = rtDW.xy_ends_POS_data[i +
        xy_ends_POS_size_idx_0];
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Static_Path_ends_POS_data[((i + Forward_Static_Path_length_0) -
        case_0) + (Path_RES_0_size_idx_1 << 1)] = rtDW.xy_ends_POS_data
        [(xy_ends_POS_size_idx_0 << 1) + i];
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Static_Path_ends_POS_data[((i + Forward_Static_Path_length_0) -
        case_0) + Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[xy_ends_POS_size_idx_0 * 3 + i];
    }

    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Static_PathCycle = 0;
      Forward_Static_Path_length_0 = 0;
    } else {
      Static_PathCycle = Forward_Static_Path_length - 1;
      Forward_Static_Path_length_0 = rtDW.SFunction_DIMS2_g[0];
    }

    case_0 = ((rtDW.SFunction_DIMS2_g[0] - Forward_Static_Path_length) +
              break_count) + 1;
    if (1 > case_0) {
      tmp_1 = 0;
    } else {
      tmp_1 = (int16_T)case_0;
    }

    case_0 = tmp_1;
    loop_ub = tmp_1 - 1;
    for (i = 0; i <= loop_ub; i++) {
      cb_data[i] = (int16_T)i;
    }

    if (1 > break_count) {
      i = 0;
    } else {
      i = break_count;
    }

    loop_ub = i - 1;
    end_ind_0 = Forward_Static_Path_length_0 - Static_PathCycle;
    for (i = 0; i < end_ind_0; i++) {
      rtDW.seg_id_data_c[i] = rtDW.seg_id_data[Static_PathCycle + i];
    }

    for (i = 0; i <= loop_ub; i++) {
      rtDW.seg_id_data_c[(i + Forward_Static_Path_length_0) - Static_PathCycle] =
        rtDW.seg_id_data[i];
    }

    for (i = 0; i < case_0; i++) {
      rtDW.Forward_Static_Path_id_0_data[cb_data[i]] = rtDW.seg_id_data_c[i];
    }

    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Forward_Static_Path_length = 1;
      Static_PathCycle = 1;
    } else {
      Static_PathCycle = rtDW.SFunction_DIMS2_g[0] + 1;
    }

    if (1 > break_count) {
      break_count = 0;
    }

    Forward_Static_Path_length = (Static_PathCycle - Forward_Static_Path_length)
      + break_count;
  } else {
    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Forward_Static_Path_length_0 = 0;
      Static_PathCycle = 0;
    } else {
      Forward_Static_Path_length_0 = Forward_Static_Path_length - 1;
      Static_PathCycle = rtDW.SFunction_DIMS2_g[0];
    }

    Path_RES_0_size_idx_1 = Static_PathCycle - Forward_Static_Path_length_0;
    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i] =
        rtDW.xy_ends_POS_data[Forward_Static_Path_length_0 + i];
    }

    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(Forward_Static_Path_length_0 + i) +
        xy_ends_POS_size_idx_0];
    }

    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i + (Path_RES_0_size_idx_1 << 1)] =
        rtDW.xy_ends_POS_data[(Forward_Static_Path_length_0 + i) +
        (xy_ends_POS_size_idx_0 << 1)];
    }

    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i + Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[(Forward_Static_Path_length_0 + i) +
        xy_ends_POS_size_idx_0 * 3];
    }

    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Forward_Static_Path_length_0 = 1;
      break_count = 0;
    } else {
      Forward_Static_Path_length_0 = Forward_Static_Path_length;
      break_count = rtDW.SFunction_DIMS2_g[0];
    }

    loop_ub = break_count - Forward_Static_Path_length_0;
    for (i = 0; i <= loop_ub; i++) {
      rtDW.Forward_Static_Path_id_0_data[i] = rtDW.seg_id_data
        [(Forward_Static_Path_length_0 + i) - 1];
    }

    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Forward_Static_Path_length = 1;
      case_0 = 1;
    } else {
      case_0 = rtDW.SFunction_DIMS2_g[0] + 1;
    }

    Forward_Static_Path_length = case_0 - Forward_Static_Path_length;
  }

  if (1 > Forward_Static_Path_length) {
    end_ind_0 = 0;
  } else {
    end_ind_0 = Forward_Static_Path_length;
  }

  Forward_Static_Path_length = Path_RES_0_size_idx_1 + 1;
  loop_ub = (Forward_Static_Path_length << 1) - 1;
  if (0 <= loop_ub) {
    memset(&rtDW.Forward_Static_Path_data[0], 0, (loop_ub + 1) * sizeof(real_T));
  }

  loop_ub = Path_RES_0_size_idx_1 - 1;
  if (0 <= loop_ub) {
    memcpy(&rtDW.Forward_Static_Path_data[0], &rtDW.Static_Path_ends_POS_data[0],
           (loop_ub + 1) * sizeof(real_T));
  }

  for (i = 0; i <= loop_ub; i++) {
    rtDW.Forward_Static_Path_data[i + Forward_Static_Path_length] =
      rtDW.Static_Path_ends_POS_data[i + Path_RES_0_size_idx_1];
  }

  i = Path_RES_0_size_idx_1 - 1;
  rtDW.Forward_Static_Path_data[Path_RES_0_size_idx_1] =
    rtDW.Static_Path_ends_POS_data[(Path_RES_0_size_idx_1 << 1) + i];
  rtDW.Forward_Static_Path_data[Path_RES_0_size_idx_1 +
    Forward_Static_Path_length] =
    rtDW.Static_Path_ends_POS_data[Path_RES_0_size_idx_1 * 3 + i];
  rtDW.SFunction_DIMS2_c[0] = Forward_Static_Path_length;
  rtDW.SFunction_DIMS2_c[1] = 1;
  loop_ub = Forward_Static_Path_length - 1;
  if (0 <= loop_ub) {
    memcpy(&rtDW.Forward_Static_Path_data_b[0], &rtDW.Forward_Static_Path_data[0],
           (loop_ub + 1) * sizeof(real_T));
  }

  if (0 <= Forward_Static_Path_length - 1) {
    memcpy(&rtDW.Forward_Static_Path_x_p[0], &rtDW.Forward_Static_Path_data_b[0],
           Forward_Static_Path_length * sizeof(real_T));
  }

  rtDW.SFunction_DIMS3_n[0] = Forward_Static_Path_length;
  rtDW.SFunction_DIMS3_n[1] = 1;
  loop_ub = Forward_Static_Path_length - 1;
  for (i = 0; i <= loop_ub; i++) {
    rtDW.Forward_Static_Path_data_b[i] = rtDW.Forward_Static_Path_data[i +
      Forward_Static_Path_length];
  }

  if (0 <= Forward_Static_Path_length - 1) {
    memcpy(&rtDW.Forward_Static_Path_y_gb[0], &rtDW.Forward_Static_Path_data_b[0],
           Forward_Static_Path_length * sizeof(real_T));
  }

  rtDW.SFunction_DIMS4_k[0] = end_ind_0;
  rtDW.SFunction_DIMS4_k[1] = 1;
  for (i = 0; i < end_ind_0; i++) {
    tmp_data[i] = 1 + i;
  }

  for (i = 0; i < end_ind_0; i++) {
    rtDW.Forward_Static_Path_id_h[i] =
      rtDW.Forward_Static_Path_id_0_data[tmp_data[i] - 1];
  }

  // MATLAB Function: '<S1>/EndPointDecision' incorporates:
  //   Inport: '<Root>/X_UKF_SLAM_i1'
  //   MATLAB Function: '<S1>/MATLAB Function'

  xy_ends_POS_size_idx_0 = 20000;
  Path_RES_0_size_idx_1 = 2;
  memset(&rtDW.Path_RES_0_data[0], 0, 40000U * sizeof(real_T));
  memset(&rtDW.Path_RES_0_1[0], 0, 40000U * sizeof(real_T));
  count = 0.0;
  count_1 = 0.0;
  break_count = 0;
  target_k = std::floor(rtb_Forward_length_final / 0.1);
  ang_1 = rtDW.Forward_Static_Path_x_p[1] - rtDW.Forward_Static_Path_x_p[0];
  J_minvalue_diff = rtDW.Forward_Static_Path_y_gb[1] -
    rtDW.Forward_Static_Path_y_gb[0];
  Length_1 = std::sqrt(ang_1 * ang_1 + J_minvalue_diff * J_minvalue_diff);
  ang_1 = rt_atan2d_snf(rtDW.Forward_Static_Path_y_gb[1] -
                        rtDW.Forward_Static_Path_y_gb[0],
                        rtDW.Forward_Static_Path_x_p[1] -
                        rtDW.Forward_Static_Path_x_p[0]);
  if (Length_1 > 0.1) {
    Length_1 = rt_roundd_snf(Length_1 / 0.1);
    for (case_0 = 0; case_0 < (int32_T)Length_1; case_0++) {
      x_endpoint1 = ((1.0 + (real_T)case_0) - 1.0) * 0.1;
      rtDW.Path_RES_0_1[case_0] = x_endpoint1 * std::cos(ang_1) +
        rtDW.Forward_Static_Path_x_p[0];
      rtDW.Path_RES_0_1[20000 + case_0] = x_endpoint1 * std::sin(ang_1) +
        rtDW.Forward_Static_Path_y_gb[0];
      count_1 = 1.0 + (real_T)case_0;
    }
  } else {
    rtDW.Path_RES_0_1[0] = rtDW.Forward_Static_Path_x_p[0];
    rtDW.Path_RES_0_1[20000] = rtDW.Forward_Static_Path_y_gb[0];
    count_1 = 1.0;
  }

  if (1.0 > count_1) {
    c = 0;
  } else {
    c = (int32_T)count_1;
  }

  Path_RES_1_size_idx_0 = c;
  if (0 <= c - 1) {
    memcpy(&rtDW.Path_RES_1_data[0], &rtDW.Path_RES_0_1[0], c * sizeof(real_T));
  }

  for (i = 0; i < c; i++) {
    rtDW.Path_RES_1_data[i + c] = rtDW.Path_RES_0_1[i + 20000];
  }

  loop_ub = c;
  for (i = 0; i < c; i++) {
    rtDW.tmp_data[i] = rtU.X_UKF_SLAM_i1[0] - rtDW.Path_RES_1_data[i];
  }

  power_m(rtDW.tmp_data, &c, rtDW.tmp_data_c, &Static_PathCycle);
  for (i = 0; i < c; i++) {
    rtDW.tmp_data[i] = rtU.X_UKF_SLAM_i1[1] - rtDW.Path_RES_1_data[i + c];
  }

  power_m(rtDW.tmp_data, &c, rtDW.tmp_data_k, &loop_ub);
  for (i = 0; i < Static_PathCycle; i++) {
    rtDW.ob_distance_data[i] = rtDW.tmp_data_c[i] + rtDW.tmp_data_k[i];
  }

  if (Static_PathCycle <= 2) {
    if (Static_PathCycle == 1) {
      Forward_Static_Path_length = 0;
    } else if (rtDW.ob_distance_data[0] > rtDW.ob_distance_data[1]) {
      Forward_Static_Path_length = 1;
    } else if (rtIsNaN(rtDW.ob_distance_data[0])) {
      if (!rtIsNaN(rtDW.ob_distance_data[1])) {
        i = 2;
      } else {
        i = 1;
      }

      Forward_Static_Path_length = i - 1;
    } else {
      Forward_Static_Path_length = 0;
    }
  } else {
    if (!rtIsNaN(rtDW.ob_distance_data[0])) {
      Forward_Static_Path_length = 0;
    } else {
      Forward_Static_Path_length = -1;
      case_0 = 2;
      exitg1 = false;
      while ((!exitg1) && (case_0 <= Static_PathCycle)) {
        if (!rtIsNaN(rtDW.ob_distance_data[case_0 - 1])) {
          Forward_Static_Path_length = case_0 - 1;
          exitg1 = true;
        } else {
          case_0++;
        }
      }
    }

    if (Forward_Static_Path_length + 1 == 0) {
      Forward_Static_Path_length = 0;
    } else {
      ang_1 = rtDW.ob_distance_data[Forward_Static_Path_length];
      for (Forward_Static_Path_length_0 = Forward_Static_Path_length + 1;
           Forward_Static_Path_length_0 < Static_PathCycle;
           Forward_Static_Path_length_0++) {
        if (ang_1 > rtDW.ob_distance_data[Forward_Static_Path_length_0]) {
          ang_1 = rtDW.ob_distance_data[Forward_Static_Path_length_0];
          Forward_Static_Path_length = Forward_Static_Path_length_0;
        }
      }
    }
  }

  ang_1 = count_1 - (real_T)(Forward_Static_Path_length + 1);
  if (rtDW.SFunction_DIMS2_c[0] - 2 >= 1) {
    for (Forward_Static_Path_length_0 = 1; Forward_Static_Path_length_0 - 1 <=
         rtDW.SFunction_DIMS2_c[0] - 3; Forward_Static_Path_length_0++) {
      if (break_count == 0) {
        J_minvalue_diff =
          rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0 + 1] -
          rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0];
        Length_1 = rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0 +
          1] - rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0];
        J_minvalue_diff = std::sqrt(J_minvalue_diff * J_minvalue_diff + Length_1
          * Length_1);
        Length_1 = rt_atan2d_snf
          (rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0 + 1] -
           rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0],
           rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0 + 1] -
           rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0]);
        if (J_minvalue_diff >= 0.1) {
          J_minvalue_diff = rt_roundd_snf(J_minvalue_diff / 0.1);
          for (Static_PathCycle = 0; Static_PathCycle < (int32_T)J_minvalue_diff;
               Static_PathCycle++) {
            x_endpoint1 = ((1.0 + (real_T)Static_PathCycle) - 1.0) * 0.1;
            i = (int32_T)((1.0 + (real_T)Static_PathCycle) + count);
            rtDW.Path_RES_0_data[i - 1] = x_endpoint1 * std::cos(Length_1) +
              rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0];
            rtDW.Path_RES_0_data[i + 19999] = x_endpoint1 * std::sin(Length_1) +
              rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0];
          }

          count += J_minvalue_diff;
        } else {
          rtDW.Path_RES_0_data[(int32_T)(1.0 + count) - 1] =
            rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0];
          rtDW.Path_RES_0_data[(int32_T)(1.0 + count) + 19999] =
            rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0];
          count++;
        }

        if (count > target_k - ang_1) {
          break_count = 1;
        }
      }
    }
  } else {
    xy_ends_POS_size_idx_0 = 0;
    Path_RES_0_size_idx_1 = 0;
  }

  Length_1 = (real_T)(Forward_Static_Path_length + 1) + target_k;
  if ((xy_ends_POS_size_idx_0 == 0) || (Path_RES_0_size_idx_1 == 0)) {
    if (Length_1 <= c) {
      if (Forward_Static_Path_length + 1 > Length_1) {
        Forward_Static_Path_length = 0;
      }

      i = Forward_Static_Path_length + (int32_T)target_k;
      End_x = rtDW.Path_RES_1_data[i - 1];
      End_y = rtDW.Path_RES_1_data[(i + c) - 1];
      x_target = target_k * 0.1;
    } else {
      if (Forward_Static_Path_length + 1 > c) {
        Forward_Static_Path_length = 0;
        Forward_Static_Path_length_0 = 0;
      } else {
        Forward_Static_Path_length_0 = c;
      }

      break_count = Forward_Static_Path_length_0 - Forward_Static_Path_length;
      i = break_count + Forward_Static_Path_length;
      End_x = rtDW.Path_RES_1_data[i - 1];
      End_y = rtDW.Path_RES_1_data[(i + c) - 1];
      if (break_count == 0) {
        break_count = 0;
      } else {
        if (!(break_count > 2)) {
          break_count = 2;
        }
      }

      x_target = (real_T)break_count * 0.1;
    }
  } else {
    if (Forward_Static_Path_length + 1 > c) {
      Forward_Static_Path_length = 0;
      break_count = 0;
    } else {
      break_count = c;
    }

    if (1.0 > count) {
      Static_PathCycle = 0;
    } else {
      Static_PathCycle = (int32_T)count;
    }

    loop_ub = break_count - Forward_Static_Path_length;
    if (!(loop_ub == 0)) {
      case_0 = 2;
      Forward_Static_Path_length_0 = loop_ub;
    } else {
      if (!(Static_PathCycle == 0)) {
        case_0 = Path_RES_0_size_idx_1;
      } else {
        case_0 = 2;
      }

      Forward_Static_Path_length_0 = 0;
    }

    if (!(Static_PathCycle == 0)) {
      c = Static_PathCycle;
    } else {
      c = 0;
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Path_RES_0_1[i] = rtDW.Path_RES_1_data[Forward_Static_Path_length + i];
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Path_RES_0_1[i + loop_ub] = rtDW.Path_RES_1_data
        [(Forward_Static_Path_length + i) + Path_RES_1_size_idx_0];
    }

    loop_ub = Path_RES_0_size_idx_1 - 1;
    for (i = 0; i <= loop_ub; i++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < Static_PathCycle;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_0_data_p[Path_RES_0_size_idx_1 + Static_PathCycle * i] =
          rtDW.Path_RES_0_data[xy_ends_POS_size_idx_0 * i +
          Path_RES_0_size_idx_1];
      }
    }

    Forward_Static_Path_length = Forward_Static_Path_length_0 + c;
    for (i = 0; i < case_0; i++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 <
           Forward_Static_Path_length_0; Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_data[Path_RES_0_size_idx_1 + Forward_Static_Path_length *
          i] = rtDW.Path_RES_0_1[Forward_Static_Path_length_0 * i +
          Path_RES_0_size_idx_1];
      }
    }

    for (i = 0; i < case_0; i++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < c;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_data[(Path_RES_0_size_idx_1 + Forward_Static_Path_length_0)
          + Forward_Static_Path_length * i] = rtDW.Path_RES_0_data_p[c * i +
          Path_RES_0_size_idx_1];
      }
    }

    if (target_k - ang_1 <= count) {
      End_x = rtDW.Path_RES_data[(int32_T)target_k - 1];
      End_y = rtDW.Path_RES_data[((int32_T)target_k + Forward_Static_Path_length)
        - 1];
      x_target = target_k * 0.1;
    } else {
      End_x_tmp_tmp = count + ang_1;
      i = (int32_T)End_x_tmp_tmp;
      End_x = rtDW.Path_RES_data[i - 1];
      End_y = rtDW.Path_RES_data[(i + Forward_Static_Path_length) - 1];
      x_target = End_x_tmp_tmp * 0.1;
    }
  }

  // MATLAB Function: '<S1>/Forward_Seg1' incorporates:
  //   MATLAB Function: '<S1>/EndPointDecision'
  //   MATLAB Function: '<S1>/Forward_Seg'

  xy_ends_POS_size_idx_0 = rtDW.SFunction_DIMS2_g[0];
  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    rtDW.xy_ends_POS_data[i] = rtDW.Static_Path_0[i + rtDW.SFunction_DIMS2_g[0]];
  }

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    rtDW.xy_ends_POS_data[i + xy_ends_POS_size_idx_0] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS2_g[0] << 1) + i];
  }

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    rtDW.xy_ends_POS_data[i + (xy_ends_POS_size_idx_0 << 1)] =
      rtDW.Static_Path_0[rtDW.SFunction_DIMS2_g[0] * 3 + i];
  }

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    rtDW.xy_ends_POS_data[i + xy_ends_POS_size_idx_0 * 3] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS2_g[0] << 2) + i];
  }

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  if (0 <= loop_ub - 1) {
    memcpy(&rtDW.seg_id_data[0], &rtDW.Static_Path_0[0], loop_ub * sizeof(real_T));
  }

  if (rtDW.Static_Path_0[(rtDW.SFunction_DIMS2_g[0] * 3 +
                          rtDW.SFunction_DIMS2_g[0]) - 1] ==
      rtDW.Static_Path_0[rtDW.SFunction_DIMS2_g[0]]) {
    Static_PathCycle = (rtDW.Static_Path_0[((rtDW.SFunction_DIMS2_g[0] << 2) +
      rtDW.SFunction_DIMS2_g[0]) - 1] ==
                        rtDW.Static_Path_0[rtDW.SFunction_DIMS2_g[0] << 1]);
  } else {
    Static_PathCycle = 0;
  }

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    varargin_1_data[i] = (rtDW.Forward_Static_Path_id_0_data[end_ind_0 - 1] ==
                          rtDW.seg_id_data[i]);
  }

  Forward_Static_Path_length = 1;
  ex = varargin_1_data[0];
  for (case_0 = 2; case_0 <= rtDW.SFunction_DIMS2_g[0]; case_0++) {
    if ((int32_T)ex < (int32_T)varargin_1_data[case_0 - 1]) {
      ex = varargin_1_data[case_0 - 1];
      Forward_Static_Path_length = case_0;
    }
  }

  ang_1 = End_x - rtDW.Static_Path_0[(rtDW.SFunction_DIMS2_g[0] * 3 +
    Forward_Static_Path_length) - 1];
  J_minvalue_diff = End_y - rtDW.Static_Path_0[((rtDW.SFunction_DIMS2_g[0] << 2)
    + Forward_Static_Path_length) - 1];
  total_length = std::sqrt(ang_1 * ang_1 + J_minvalue_diff * J_minvalue_diff);
  end_ind_0 = Forward_Static_Path_length;
  case_0 = 0;
  break_count = 0;
  Forward_Static_Path_length_0 = 0;
  exitg1 = false;
  while ((!exitg1) && (Forward_Static_Path_length_0 <= rtDW.SFunction_DIMS2_g[0]
                       - 1)) {
    if (total_length > rtU.forward_length_2) {
      break_count = end_ind_0;
      exitg1 = true;
    } else {
      i = Forward_Static_Path_length + Forward_Static_Path_length_0;
      Path_RES_0_size_idx_1 = i + 1;
      if (Path_RES_0_size_idx_1 <= rtDW.SFunction_DIMS2_g[0]) {
        total_length += rtDW.Static_Path_0[i + (rtDW.SFunction_DIMS2_g[0] << 3)];
        end_ind_0 = Path_RES_0_size_idx_1;
        case_0 = 1;
        Forward_Static_Path_length_0++;
      } else if (Static_PathCycle == 1) {
        i -= rtDW.SFunction_DIMS2_g[0];
        total_length += rtDW.Static_Path_0[i + (rtDW.SFunction_DIMS2_g[0] << 3)];
        end_ind_0 = i + 1;
        case_0 = 2;
        Forward_Static_Path_length_0++;
      } else {
        break_count = end_ind_0;
        case_0 = 3;
        exitg1 = true;
      }
    }
  }

  Forward_Static_Path_length_0 = rtDW.SFunction_DIMS2_g[0] - 1;
  if (0 <= Forward_Static_Path_length_0) {
    memset(&rtDW.Forward_Static_Path_id_0_data[0], 0,
           (Forward_Static_Path_length_0 + 1) * sizeof(real_T));
  }

  if ((case_0 == 1) || (case_0 == 0)) {
    if (Forward_Static_Path_length > break_count) {
      c = 0;
      case_0 = 0;
    } else {
      c = Forward_Static_Path_length - 1;
      case_0 = break_count;
    }

    Path_RES_0_size_idx_1 = case_0 - c;
    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i] = rtDW.xy_ends_POS_data[c + i];
    }

    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(c + i) + xy_ends_POS_size_idx_0];
    }

    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i + (Path_RES_0_size_idx_1 << 1)] =
        rtDW.xy_ends_POS_data[(c + i) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i + Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[(c + i) + xy_ends_POS_size_idx_0 * 3];
    }

    if (Forward_Static_Path_length > break_count) {
      Forward_Static_Path_length_0 = 1;
      case_0 = 0;
    } else {
      Forward_Static_Path_length_0 = Forward_Static_Path_length;
      case_0 = break_count;
    }

    loop_ub = case_0 - Forward_Static_Path_length_0;
    for (i = 0; i <= loop_ub; i++) {
      rtDW.Forward_Static_Path_id_0_data[i] = rtDW.seg_id_data
        [(Forward_Static_Path_length_0 + i) - 1];
    }

    if (Forward_Static_Path_length > break_count) {
      Forward_Static_Path_length = 1;
      break_count = 0;
    }

    Forward_Static_Path_length = (break_count - Forward_Static_Path_length) + 1;
  } else if (case_0 == 2) {
    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      case_0 = 0;
      Forward_Static_Path_length_0 = 0;
    } else {
      case_0 = Forward_Static_Path_length - 1;
      Forward_Static_Path_length_0 = rtDW.SFunction_DIMS2_g[0];
    }

    if (1 > break_count) {
      loop_ub = 0;
    } else {
      loop_ub = break_count;
    }

    Static_PathCycle = Forward_Static_Path_length_0 - case_0;
    Path_RES_0_size_idx_1 = Static_PathCycle + loop_ub;
    for (i = 0; i < Static_PathCycle; i++) {
      rtDW.Static_Path_ends_POS_data[i] = rtDW.xy_ends_POS_data[case_0 + i];
    }

    for (i = 0; i < Static_PathCycle; i++) {
      rtDW.Static_Path_ends_POS_data[i + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(case_0 + i) + xy_ends_POS_size_idx_0];
    }

    for (i = 0; i < Static_PathCycle; i++) {
      rtDW.Static_Path_ends_POS_data[i + (Path_RES_0_size_idx_1 << 1)] =
        rtDW.xy_ends_POS_data[(case_0 + i) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (i = 0; i < Static_PathCycle; i++) {
      rtDW.Static_Path_ends_POS_data[i + Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[(case_0 + i) + xy_ends_POS_size_idx_0 * 3];
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Static_Path_ends_POS_data[(i + Forward_Static_Path_length_0) - case_0]
        = rtDW.xy_ends_POS_data[i];
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Static_Path_ends_POS_data[((i + Forward_Static_Path_length_0) -
        case_0) + Path_RES_0_size_idx_1] = rtDW.xy_ends_POS_data[i +
        xy_ends_POS_size_idx_0];
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Static_Path_ends_POS_data[((i + Forward_Static_Path_length_0) -
        case_0) + (Path_RES_0_size_idx_1 << 1)] = rtDW.xy_ends_POS_data
        [(xy_ends_POS_size_idx_0 << 1) + i];
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Static_Path_ends_POS_data[((i + Forward_Static_Path_length_0) -
        case_0) + Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[xy_ends_POS_size_idx_0 * 3 + i];
    }

    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Static_PathCycle = 0;
      Forward_Static_Path_length_0 = 0;
    } else {
      Static_PathCycle = Forward_Static_Path_length - 1;
      Forward_Static_Path_length_0 = rtDW.SFunction_DIMS2_g[0];
    }

    case_0 = ((rtDW.SFunction_DIMS2_g[0] - Forward_Static_Path_length) +
              break_count) + 1;
    if (1 > case_0) {
      tmp_1 = 0;
    } else {
      tmp_1 = (int16_T)case_0;
    }

    case_0 = tmp_1;
    loop_ub = tmp_1 - 1;
    for (i = 0; i <= loop_ub; i++) {
      cb_data[i] = (int16_T)i;
    }

    if (1 > break_count) {
      i = 0;
    } else {
      i = break_count;
    }

    loop_ub = i - 1;
    end_ind_0 = Forward_Static_Path_length_0 - Static_PathCycle;
    for (i = 0; i < end_ind_0; i++) {
      rtDW.seg_id_data_c[i] = rtDW.seg_id_data[Static_PathCycle + i];
    }

    for (i = 0; i <= loop_ub; i++) {
      rtDW.seg_id_data_c[(i + Forward_Static_Path_length_0) - Static_PathCycle] =
        rtDW.seg_id_data[i];
    }

    for (i = 0; i < case_0; i++) {
      rtDW.Forward_Static_Path_id_0_data[cb_data[i]] = rtDW.seg_id_data_c[i];
    }

    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Forward_Static_Path_length = 1;
      Static_PathCycle = 1;
    } else {
      Static_PathCycle = rtDW.SFunction_DIMS2_g[0] + 1;
    }

    if (1 > break_count) {
      break_count = 0;
    }

    Forward_Static_Path_length = (Static_PathCycle - Forward_Static_Path_length)
      + break_count;
  } else {
    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Forward_Static_Path_length_0 = 0;
      Static_PathCycle = 0;
    } else {
      Forward_Static_Path_length_0 = Forward_Static_Path_length - 1;
      Static_PathCycle = rtDW.SFunction_DIMS2_g[0];
    }

    Path_RES_0_size_idx_1 = Static_PathCycle - Forward_Static_Path_length_0;
    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i] =
        rtDW.xy_ends_POS_data[Forward_Static_Path_length_0 + i];
    }

    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(Forward_Static_Path_length_0 + i) +
        xy_ends_POS_size_idx_0];
    }

    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i + (Path_RES_0_size_idx_1 << 1)] =
        rtDW.xy_ends_POS_data[(Forward_Static_Path_length_0 + i) +
        (xy_ends_POS_size_idx_0 << 1)];
    }

    for (i = 0; i < Path_RES_0_size_idx_1; i++) {
      rtDW.Static_Path_ends_POS_data[i + Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[(Forward_Static_Path_length_0 + i) +
        xy_ends_POS_size_idx_0 * 3];
    }

    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Forward_Static_Path_length_0 = 1;
      break_count = 0;
    } else {
      Forward_Static_Path_length_0 = Forward_Static_Path_length;
      break_count = rtDW.SFunction_DIMS2_g[0];
    }

    loop_ub = break_count - Forward_Static_Path_length_0;
    for (i = 0; i <= loop_ub; i++) {
      rtDW.Forward_Static_Path_id_0_data[i] = rtDW.seg_id_data
        [(Forward_Static_Path_length_0 + i) - 1];
    }

    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Forward_Static_Path_length = 1;
      case_0 = 1;
    } else {
      case_0 = rtDW.SFunction_DIMS2_g[0] + 1;
    }

    Forward_Static_Path_length = case_0 - Forward_Static_Path_length;
  }

  if (1 > Forward_Static_Path_length) {
    end_ind_0 = 0;
  } else {
    end_ind_0 = Forward_Static_Path_length;
  }

  Forward_Static_Path_length = Path_RES_0_size_idx_1 + 1;
  loop_ub = (Forward_Static_Path_length << 1) - 1;
  if (0 <= loop_ub) {
    memset(&rtDW.Forward_Static_Path_data_m[0], 0, (loop_ub + 1) * sizeof(real_T));
  }

  loop_ub = Path_RES_0_size_idx_1 - 1;
  if (0 <= loop_ub) {
    memcpy(&rtDW.Forward_Static_Path_data_m[0], &rtDW.Static_Path_ends_POS_data
           [0], (loop_ub + 1) * sizeof(real_T));
  }

  for (i = 0; i <= loop_ub; i++) {
    rtDW.Forward_Static_Path_data_m[i + Forward_Static_Path_length] =
      rtDW.Static_Path_ends_POS_data[i + Path_RES_0_size_idx_1];
  }

  i = Path_RES_0_size_idx_1 - 1;
  rtDW.Forward_Static_Path_data_m[Path_RES_0_size_idx_1] =
    rtDW.Static_Path_ends_POS_data[(Path_RES_0_size_idx_1 << 1) + i];
  rtDW.Forward_Static_Path_data_m[Path_RES_0_size_idx_1 +
    Forward_Static_Path_length] =
    rtDW.Static_Path_ends_POS_data[Path_RES_0_size_idx_1 * 3 + i];
  rtDW.SFunction_DIMS2_i[0] = Forward_Static_Path_length;
  rtDW.SFunction_DIMS2_i[1] = 1;
  loop_ub = Forward_Static_Path_length - 1;
  if (0 <= loop_ub) {
    memcpy(&rtDW.Forward_Static_Path_data_b[0],
           &rtDW.Forward_Static_Path_data_m[0], (loop_ub + 1) * sizeof(real_T));
  }

  if (0 <= Forward_Static_Path_length - 1) {
    memcpy(&rtDW.Forward_Static_Path_x_p[0], &rtDW.Forward_Static_Path_data_b[0],
           Forward_Static_Path_length * sizeof(real_T));
  }

  rtDW.SFunction_DIMS3_a[0] = Forward_Static_Path_length;
  rtDW.SFunction_DIMS3_a[1] = 1;
  loop_ub = Forward_Static_Path_length - 1;
  for (i = 0; i <= loop_ub; i++) {
    rtDW.Forward_Static_Path_data_b[i] = rtDW.Forward_Static_Path_data_m[i +
      Forward_Static_Path_length];
  }

  if (0 <= Forward_Static_Path_length - 1) {
    memcpy(&rtDW.Forward_Static_Path_y_gb[0], &rtDW.Forward_Static_Path_data_b[0],
           Forward_Static_Path_length * sizeof(real_T));
  }

  rtDW.SFunction_DIMS4_l[0] = end_ind_0;
  rtDW.SFunction_DIMS4_l[1] = 1;
  for (i = 0; i < end_ind_0; i++) {
    tmp_data[i] = 1 + i;
  }

  for (i = 0; i < end_ind_0; i++) {
    rtDW.Forward_Static_Path_id_g[i] =
      rtDW.Forward_Static_Path_id_0_data[tmp_data[i] - 1];
  }

  // End of MATLAB Function: '<S1>/Forward_Seg1'

  // MATLAB Function: '<S1>/Boundingbox_trans' incorporates:
  //   Inport: '<Root>/BB_all_XY'
  //   Inport: '<Root>/BB_num'
  //   Inport: '<Root>/VirBB_mode'
  //   Inport: '<Root>/X_UKF_SLAM_i1'
  //   MATLAB Function: '<S1>/MATLAB Function'

  vehicle_heading = rtU.X_UKF_SLAM_i1[2];
  memcpy(&rtb_V_boundingbox[0], &rtU.BB_all_XY[0], 400U * sizeof(real_T));
  if (rtU.VirBB_mode == 0.0) {
    for (Static_PathCycle = 0; Static_PathCycle < (int32_T)rtU.BB_num;
         Static_PathCycle++) {
      offset_5 = (1.0 + (real_T)Static_PathCycle) * 2.0;
      for (Forward_Static_Path_length_0 = 0; Forward_Static_Path_length_0 < 4;
           Forward_Static_Path_length_0++) {
        End_x_tmp_tmp = std::sin(vehicle_heading);
        rtb_UnitDelay18 = std::cos(vehicle_heading);
        rtb_V_boundingbox[((int32_T)(offset_5 + -1.0) + 100 *
                           Forward_Static_Path_length_0) - 1] = (rtU.BB_all_XY
          [((int32_T)(offset_5 + -1.0) + 100 * Forward_Static_Path_length_0) - 1]
          * rtb_UnitDelay18 + rtU.BB_all_XY[(100 * Forward_Static_Path_length_0
          + (int32_T)offset_5) - 1] * -End_x_tmp_tmp) + rtU.X_UKF_SLAM_i1[0];
        rtb_V_boundingbox[((int32_T)offset_5 + 100 *
                           Forward_Static_Path_length_0) - 1] = (rtU.BB_all_XY
          [((int32_T)(offset_5 + -1.0) + 100 * Forward_Static_Path_length_0) - 1]
          * End_x_tmp_tmp + rtU.BB_all_XY[(100 * Forward_Static_Path_length_0 +
          (int32_T)offset_5) - 1] * rtb_UnitDelay18) + rtU.X_UKF_SLAM_i1[1];
      }
    }
  }

  // End of MATLAB Function: '<S1>/Boundingbox_trans'

  // MATLAB Function: '<S1>/target_seg_id_search' incorporates:
  //   Inport: '<Root>/Look_ahead_S0'
  //   Inport: '<Root>/Oi_near_i'
  //   Inport: '<Root>/Speed_mps1'
  //   Inport: '<Root>/seg_id_near_i'

  offset_5 = rtU.Speed_mps1 * rtb_Look_ahead_time + rtU.Look_ahead_S0;
  loop_ub = rtDW.SFunction_DIMS2_g[0];
  if (0 <= loop_ub - 1) {
    memcpy(&rtDW.seg_id_data[0], &rtDW.Static_Path_0[0], loop_ub * sizeof(real_T));
  }

  if (rtDW.Static_Path_0[(rtDW.SFunction_DIMS2_g[0] * 3 +
                          rtDW.SFunction_DIMS2_g[0]) - 1] ==
      rtDW.Static_Path_0[rtDW.SFunction_DIMS2_g[0]]) {
    Static_PathCycle = (rtDW.Static_Path_0[((rtDW.SFunction_DIMS2_g[0] << 2) +
      rtDW.SFunction_DIMS2_g[0]) - 1] ==
                        rtDW.Static_Path_0[rtDW.SFunction_DIMS2_g[0] << 1]);
  } else {
    Static_PathCycle = 0;
  }

  Forward_Static_Path_length_0 = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < Forward_Static_Path_length_0; i++) {
    varargin_1_data_0[i] = (rtDW.seg_id_data[i] == rtU.seg_id_near_i);
  }

  Forward_Static_Path_length = 1;
  ex = varargin_1_data_0[0];
  for (case_0 = 2; case_0 <= Forward_Static_Path_length_0; case_0++) {
    if ((int32_T)ex < (int32_T)varargin_1_data_0[case_0 - 1]) {
      ex = varargin_1_data_0[case_0 - 1];
      Forward_Static_Path_length = case_0;
    }
  }

  ang_1 = rtU.Oi_near_i[0] - rtDW.Static_Path_0[(rtDW.SFunction_DIMS2_g[0] * 3 +
    Forward_Static_Path_length) - 1];
  J_minvalue_diff = rtU.Oi_near_i[1] - rtDW.Static_Path_0
    [((rtDW.SFunction_DIMS2_g[0] << 2) + Forward_Static_Path_length) - 1];
  total_length = std::sqrt(ang_1 * ang_1 + J_minvalue_diff * J_minvalue_diff);
  end_ind_0 = Forward_Static_Path_length;
  case_0 = 0;
  break_count = 0;
  Forward_Static_Path_length_0 = 0;
  exitg1 = false;
  while ((!exitg1) && (Forward_Static_Path_length_0 <= rtDW.SFunction_DIMS2_g[0]
                       - 1)) {
    if (total_length > offset_5) {
      break_count = end_ind_0;
      exitg1 = true;
    } else {
      i = Forward_Static_Path_length + Forward_Static_Path_length_0;
      Path_RES_0_size_idx_1 = i + 1;
      if (Path_RES_0_size_idx_1 <= rtDW.SFunction_DIMS2_g[0]) {
        total_length += rtDW.Static_Path_0[i + (rtDW.SFunction_DIMS2_g[0] << 3)];
        end_ind_0 = Path_RES_0_size_idx_1;
        case_0 = 1;
        Forward_Static_Path_length_0++;
      } else if (Static_PathCycle == 1) {
        i -= rtDW.SFunction_DIMS2_g[0];
        total_length += rtDW.Static_Path_0[i + (rtDW.SFunction_DIMS2_g[0] << 3)];
        end_ind_0 = i + 1;
        case_0 = 2;
        Forward_Static_Path_length_0++;
      } else {
        break_count = end_ind_0;
        case_0 = 3;
        exitg1 = true;
      }
    }
  }

  Forward_Static_Path_length_0 = rtDW.SFunction_DIMS2_g[0];
  if (0 <= Forward_Static_Path_length_0 - 1) {
    memset(&Forward_Static_Path_id_0_data[0], 0, Forward_Static_Path_length_0 *
           sizeof(real_T));
  }

  if ((case_0 == 1) || (case_0 == 0)) {
    if (Forward_Static_Path_length > break_count) {
      c = 0;
      case_0 = 0;
    } else {
      c = Forward_Static_Path_length - 1;
      case_0 = break_count;
    }

    Path_RES_0_size_idx_1 = case_0 - c;
    if (Forward_Static_Path_length > break_count) {
      Forward_Static_Path_length_0 = 1;
      case_0 = 0;
    } else {
      Forward_Static_Path_length_0 = Forward_Static_Path_length;
      case_0 = break_count;
    }

    loop_ub = case_0 - Forward_Static_Path_length_0;
    for (i = 0; i <= loop_ub; i++) {
      Forward_Static_Path_id_0_data[i] = rtDW.seg_id_data
        [(Forward_Static_Path_length_0 + i) - 1];
    }

    if (Forward_Static_Path_length > break_count) {
      Forward_Static_Path_length = 1;
      break_count = 0;
    }

    Forward_Static_Path_length = (break_count - Forward_Static_Path_length) + 1;
  } else if (case_0 == 2) {
    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      case_0 = 0;
      Forward_Static_Path_length_0 = 0;
    } else {
      case_0 = Forward_Static_Path_length - 1;
      Forward_Static_Path_length_0 = rtDW.SFunction_DIMS2_g[0];
    }

    if (1 > break_count) {
      i = 0;
    } else {
      i = break_count;
    }

    Path_RES_0_size_idx_1 = (Forward_Static_Path_length_0 - case_0) + i;
    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Static_PathCycle = 0;
      Forward_Static_Path_length_0 = 0;
    } else {
      Static_PathCycle = Forward_Static_Path_length - 1;
      Forward_Static_Path_length_0 = rtDW.SFunction_DIMS2_g[0];
    }

    case_0 = ((rtDW.SFunction_DIMS2_g[0] - Forward_Static_Path_length) +
              break_count) + 1;
    if (1 > case_0) {
      tmp_1 = 0;
    } else {
      tmp_1 = (int16_T)case_0;
    }

    case_0 = tmp_1;
    loop_ub = tmp_1 - 1;
    for (i = 0; i <= loop_ub; i++) {
      cb_data_0[i] = (int16_T)i;
    }

    if (1 > break_count) {
      i = 0;
    } else {
      i = break_count;
    }

    loop_ub = i - 1;
    end_ind_0 = Forward_Static_Path_length_0 - Static_PathCycle;
    for (i = 0; i < end_ind_0; i++) {
      seg_id_data[i] = rtDW.seg_id_data[Static_PathCycle + i];
    }

    for (i = 0; i <= loop_ub; i++) {
      seg_id_data[(i + Forward_Static_Path_length_0) - Static_PathCycle] =
        rtDW.seg_id_data[i];
    }

    for (i = 0; i < case_0; i++) {
      Forward_Static_Path_id_0_data[cb_data_0[i]] = seg_id_data[i];
    }

    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Forward_Static_Path_length = 1;
      Static_PathCycle = 1;
    } else {
      Static_PathCycle = rtDW.SFunction_DIMS2_g[0] + 1;
    }

    if (1 > break_count) {
      break_count = 0;
    }

    Forward_Static_Path_length = (Static_PathCycle - Forward_Static_Path_length)
      + break_count;
  } else {
    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Forward_Static_Path_length_0 = 0;
      Static_PathCycle = 0;
    } else {
      Forward_Static_Path_length_0 = Forward_Static_Path_length - 1;
      Static_PathCycle = rtDW.SFunction_DIMS2_g[0];
    }

    Path_RES_0_size_idx_1 = Static_PathCycle - Forward_Static_Path_length_0;
    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Forward_Static_Path_length_0 = 1;
      break_count = 0;
    } else {
      Forward_Static_Path_length_0 = Forward_Static_Path_length;
      break_count = rtDW.SFunction_DIMS2_g[0];
    }

    loop_ub = break_count - Forward_Static_Path_length_0;
    for (i = 0; i <= loop_ub; i++) {
      Forward_Static_Path_id_0_data[i] = rtDW.seg_id_data
        [(Forward_Static_Path_length_0 + i) - 1];
    }

    if (Forward_Static_Path_length > rtDW.SFunction_DIMS2_g[0]) {
      Forward_Static_Path_length = 1;
      case_0 = 1;
    } else {
      case_0 = rtDW.SFunction_DIMS2_g[0] + 1;
    }

    Forward_Static_Path_length = case_0 - Forward_Static_Path_length;
  }

  if (1 > Forward_Static_Path_length) {
    end_ind_0 = 0;
  } else {
    end_ind_0 = Forward_Static_Path_length;
  }

  rtDW.SFunction_DIMS4 = end_ind_0;
  if (0 <= end_ind_0 - 1) {
    memcpy(&rtb_Forward_Static_Path_id[0], &Forward_Static_Path_id_0_data[0],
           end_ind_0 * sizeof(real_T));
  }

  Forward_Static_Path_length = Path_RES_0_size_idx_1 + 1;
  rtDW.SFunction_DIMS2 = Forward_Static_Path_length;
  rtDW.SFunction_DIMS3 = Forward_Static_Path_length;
  rtDW.SFunction_DIMS6[0] = rtDW.SFunction_DIMS2_g[0];
  rtDW.SFunction_DIMS6[1] = 1;

  // Gain: '<S1>/Gain3' incorporates:
  //   Inport: '<Root>/X_UKF_SLAM_i1'
  //   MATLAB Function: '<S1>/MATLAB Function'

  vehicle_heading = 57.295779513082323 * rtU.X_UKF_SLAM_i1[2];

  // MATLAB Function: '<S1>/EndPointDecision2' incorporates:
  //   Inport: '<Root>/takeover_mag'
  //   MATLAB Function: '<S1>/DynamicPathPlanning1'
  //   MATLAB Function: '<S1>/EndPointDecision'
  //   MATLAB Function: '<S1>/EndPointDecision1'

  xy_ends_POS_size_idx_0 = 20000;
  Path_RES_0_size_idx_1 = 2;
  memset(&rtDW.Path_RES_0_data[0], 0, 40000U * sizeof(real_T));
  memset(&rtDW.Path_RES_0_1[0], 0, 40000U * sizeof(real_T));
  count = 0.0;
  count_1 = 0.0;
  break_count = 0;
  End_x_tmp_tmp = rtb_Forward_length_final * rtU.takeover_mag;
  target_k = std::floor((End_x_tmp_tmp + 2.0) / 0.1);
  ang_1_tmp = rtDW.Forward_Static_Path_x_p[1] - rtDW.Forward_Static_Path_x_p[0];
  J_minvalue_diff_tmp = rtDW.Forward_Static_Path_y_gb[1] -
    rtDW.Forward_Static_Path_y_gb[0];
  x_endpoint1 = std::sqrt(ang_1_tmp * ang_1_tmp + J_minvalue_diff_tmp *
    J_minvalue_diff_tmp);
  ang_1 = rt_atan2d_snf(rtDW.Forward_Static_Path_y_gb[1] -
                        rtDW.Forward_Static_Path_y_gb[0],
                        rtDW.Forward_Static_Path_x_p[1] -
                        rtDW.Forward_Static_Path_x_p[0]);
  if (x_endpoint1 > 0.1) {
    Length_1 = rt_roundd_snf(x_endpoint1 / 0.1);
    for (case_0 = 0; case_0 < (int32_T)Length_1; case_0++) {
      x_endpoint1 = ((1.0 + (real_T)case_0) - 1.0) * 0.1;
      rtDW.Path_RES_0_1[case_0] = x_endpoint1 * std::cos(ang_1) +
        rtDW.Forward_Static_Path_x_p[0];
      rtDW.Path_RES_0_1[20000 + case_0] = x_endpoint1 * std::sin(ang_1) +
        rtDW.Forward_Static_Path_y_gb[0];
      count_1 = 1.0 + (real_T)case_0;
    }
  } else {
    rtDW.Path_RES_0_1[0] = rtDW.Forward_Static_Path_x_p[0];
    rtDW.Path_RES_0_1[20000] = rtDW.Forward_Static_Path_y_gb[0];
    count_1 = 1.0;
  }

  if (1.0 > count_1) {
    c = 0;
  } else {
    c = (int32_T)count_1;
  }

  Path_RES_1_size_idx_0 = c;
  if (0 <= c - 1) {
    memcpy(&rtDW.Path_RES_1_data[0], &rtDW.Path_RES_0_1[0], c * sizeof(real_T));
  }

  for (i = 0; i < c; i++) {
    rtDW.Path_RES_1_data[i + c] = rtDW.Path_RES_0_1[i + 20000];
  }

  for (i = 0; i < c; i++) {
    rtDW.tmp_data_c[i] = End_x - rtDW.Path_RES_1_data[i];
  }

  power_a(rtDW.tmp_data_c, &c, rtDW.tmp_data, &loop_ub);
  for (i = 0; i < c; i++) {
    rtDW.tmp_data_k[i] = End_y - rtDW.Path_RES_1_data[i + c];
  }

  power_a(rtDW.tmp_data_k, &c, rtDW.tmp_data_c, &Static_PathCycle);
  for (i = 0; i < loop_ub; i++) {
    rtDW.ob_distance_data[i] = rtDW.tmp_data[i] + rtDW.tmp_data_c[i];
  }

  if (loop_ub <= 2) {
    if (loop_ub == 1) {
      Forward_Static_Path_length = 0;
    } else if (rtDW.ob_distance_data[0] > rtDW.ob_distance_data[1]) {
      Forward_Static_Path_length = 1;
    } else if (rtIsNaN(rtDW.ob_distance_data[0])) {
      if (!rtIsNaN(rtDW.ob_distance_data[1])) {
        i = 2;
      } else {
        i = 1;
      }

      Forward_Static_Path_length = i - 1;
    } else {
      Forward_Static_Path_length = 0;
    }
  } else {
    if (!rtIsNaN(rtDW.ob_distance_data[0])) {
      Forward_Static_Path_length = 0;
    } else {
      Forward_Static_Path_length = -1;
      case_0 = 2;
      exitg1 = false;
      while ((!exitg1) && (case_0 <= loop_ub)) {
        if (!rtIsNaN(rtDW.ob_distance_data[case_0 - 1])) {
          Forward_Static_Path_length = case_0 - 1;
          exitg1 = true;
        } else {
          case_0++;
        }
      }
    }

    if (Forward_Static_Path_length + 1 == 0) {
      Forward_Static_Path_length = 0;
    } else {
      ang_1 = rtDW.ob_distance_data[Forward_Static_Path_length];
      for (Forward_Static_Path_length_0 = Forward_Static_Path_length + 1;
           Forward_Static_Path_length_0 < loop_ub; Forward_Static_Path_length_0
           ++) {
        if (ang_1 > rtDW.ob_distance_data[Forward_Static_Path_length_0]) {
          ang_1 = rtDW.ob_distance_data[Forward_Static_Path_length_0];
          Forward_Static_Path_length = Forward_Static_Path_length_0;
        }
      }
    }
  }

  ang_1 = count_1 - (real_T)(Forward_Static_Path_length + 1);
  if (rtDW.SFunction_DIMS2_i[0] - 2 >= 1) {
    for (Forward_Static_Path_length_0 = 1; Forward_Static_Path_length_0 - 1 <=
         rtDW.SFunction_DIMS2_i[0] - 3; Forward_Static_Path_length_0++) {
      if (break_count == 0) {
        J_minvalue_diff =
          rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0 + 1] -
          rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0];
        Length_1 = rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0 +
          1] - rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0];
        J_minvalue_diff = std::sqrt(J_minvalue_diff * J_minvalue_diff + Length_1
          * Length_1);
        Length_1 = rt_atan2d_snf
          (rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0 + 1] -
           rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0],
           rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0 + 1] -
           rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0]);
        if (J_minvalue_diff >= 0.1) {
          J_minvalue_diff = rt_roundd_snf(J_minvalue_diff / 0.1);
          for (Static_PathCycle = 0; Static_PathCycle < (int32_T)J_minvalue_diff;
               Static_PathCycle++) {
            x_endpoint1 = ((1.0 + (real_T)Static_PathCycle) - 1.0) * 0.1;
            i = (int32_T)((1.0 + (real_T)Static_PathCycle) + count);
            rtDW.Path_RES_0_data[i - 1] = x_endpoint1 * std::cos(Length_1) +
              rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0];
            rtDW.Path_RES_0_data[i + 19999] = x_endpoint1 * std::sin(Length_1) +
              rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0];
          }

          count += J_minvalue_diff;
        } else {
          rtDW.Path_RES_0_data[(int32_T)(1.0 + count) - 1] =
            rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0];
          rtDW.Path_RES_0_data[(int32_T)(1.0 + count) + 19999] =
            rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0];
          count++;
        }

        if (count > target_k - ang_1) {
          break_count = 1;
        }
      }
    }
  } else {
    xy_ends_POS_size_idx_0 = 0;
    Path_RES_0_size_idx_1 = 0;
  }

  Length_1 = (real_T)(Forward_Static_Path_length + 1) + target_k;
  if ((xy_ends_POS_size_idx_0 == 0) || (Path_RES_0_size_idx_1 == 0)) {
    if (Length_1 <= c) {
      if (Forward_Static_Path_length + 1 > Length_1) {
        Forward_Static_Path_length = 0;
      }

      i = Forward_Static_Path_length + (int32_T)target_k;
      x_endpoint1 = rtDW.Path_RES_1_data[i - 1];
      y_endpoint1 = rtDW.Path_RES_1_data[(i + c) - 1];
      count = target_k * 0.1;
    } else {
      if (Forward_Static_Path_length + 1 > c) {
        Forward_Static_Path_length = 0;
        Forward_Static_Path_length_0 = 0;
      } else {
        Forward_Static_Path_length_0 = c;
      }

      break_count = Forward_Static_Path_length_0 - Forward_Static_Path_length;
      i = break_count + Forward_Static_Path_length;
      x_endpoint1 = rtDW.Path_RES_1_data[i - 1];
      y_endpoint1 = rtDW.Path_RES_1_data[(i + c) - 1];
      if (break_count == 0) {
        break_count = 0;
      } else {
        if (!(break_count > 2)) {
          break_count = 2;
        }
      }

      count = (real_T)break_count * 0.1;
    }
  } else {
    if (Forward_Static_Path_length + 1 > c) {
      Forward_Static_Path_length = 0;
      break_count = 0;
    } else {
      break_count = c;
    }

    if (1.0 > count) {
      Static_PathCycle = 0;
    } else {
      Static_PathCycle = (int32_T)count;
    }

    loop_ub = break_count - Forward_Static_Path_length;
    if (!(loop_ub == 0)) {
      case_0 = 2;
      Forward_Static_Path_length_0 = loop_ub;
    } else {
      if (!(Static_PathCycle == 0)) {
        case_0 = Path_RES_0_size_idx_1;
      } else {
        case_0 = 2;
      }

      Forward_Static_Path_length_0 = 0;
    }

    if (!(Static_PathCycle == 0)) {
      c = Static_PathCycle;
    } else {
      c = 0;
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Path_RES_0_1[i] = rtDW.Path_RES_1_data[Forward_Static_Path_length + i];
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Path_RES_0_1[i + loop_ub] = rtDW.Path_RES_1_data
        [(Forward_Static_Path_length + i) + Path_RES_1_size_idx_0];
    }

    loop_ub = Path_RES_0_size_idx_1 - 1;
    for (i = 0; i <= loop_ub; i++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < Static_PathCycle;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_0_data_p[Path_RES_0_size_idx_1 + Static_PathCycle * i] =
          rtDW.Path_RES_0_data[xy_ends_POS_size_idx_0 * i +
          Path_RES_0_size_idx_1];
      }
    }

    Forward_Static_Path_length = Forward_Static_Path_length_0 + c;
    for (i = 0; i < case_0; i++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 <
           Forward_Static_Path_length_0; Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_data[Path_RES_0_size_idx_1 + Forward_Static_Path_length *
          i] = rtDW.Path_RES_0_1[Forward_Static_Path_length_0 * i +
          Path_RES_0_size_idx_1];
      }
    }

    for (i = 0; i < case_0; i++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < c;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_data[(Path_RES_0_size_idx_1 + Forward_Static_Path_length_0)
          + Forward_Static_Path_length * i] = rtDW.Path_RES_0_data_p[c * i +
          Path_RES_0_size_idx_1];
      }
    }

    if (target_k - ang_1 <= count) {
      x_endpoint1 = rtDW.Path_RES_data[(int32_T)target_k - 1];
      y_endpoint1 = rtDW.Path_RES_data[((int32_T)target_k +
        Forward_Static_Path_length) - 1];
      count = target_k * 0.1;
    } else {
      count += ang_1;
      i = (int32_T)count;
      x_endpoint1 = rtDW.Path_RES_data[i - 1];
      y_endpoint1 = rtDW.Path_RES_data[(i + Forward_Static_Path_length) - 1];
      count *= 0.1;
    }
  }

  // UnitDelay: '<S1>/Unit Delay18'
  rtb_UnitDelay18 = rtDW.UnitDelay18_DSTATE;

  // MATLAB Function: '<S1>/DangerousArea1' incorporates:
  //   Inport: '<Root>/X_UKF_SLAM_i1'
  //   Inport: '<Root>/w_off_'
  //   Inport: '<Root>/w_off_avoid'
  //   MATLAB Function: '<S1>/EndPointDecision2'
  //   MATLAB Function: '<S1>/MATLAB Function'
  //   UnitDelay: '<S1>/Unit Delay15'
  //   UnitDelay: '<S1>/Unit Delay17'
  //   UnitDelay: '<S1>/Unit Delay19'

  total_length = rtb_UnitDelay18;
  rtb_H_x_out[0] = rtDW.UnitDelay19_DSTATE[0];
  rtb_H_y_out[0] = rtDW.UnitDelay15_DSTATE[0];
  rtb_H_x_out[1] = rtDW.UnitDelay19_DSTATE[1];
  rtb_H_y_out[1] = rtDW.UnitDelay15_DSTATE[1];
  rtb_H_x_out[2] = rtDW.UnitDelay19_DSTATE[2];
  rtb_H_y_out[2] = rtDW.UnitDelay15_DSTATE[2];
  rtb_H_x_out[3] = rtDW.UnitDelay19_DSTATE[3];
  rtb_H_y_out[3] = rtDW.UnitDelay15_DSTATE[3];
  end_ind_0 = 0;
  target_k = rtU.X_UKF_SLAM_i1[0];
  y = rtU.X_UKF_SLAM_i1[1];
  c = rtDW.SFunction_DIMS4_k[0];
  loop_ub = rtDW.SFunction_DIMS4_k[0] * rtDW.SFunction_DIMS2_g[1] - 1;
  if (0 <= loop_ub) {
    memset(&rtDW.Forward_Static_Path_0_data[0], 0, (loop_ub + 1) * sizeof(real_T));
  }

  for (Forward_Static_Path_length_0 = 0; Forward_Static_Path_length_0 <
       rtDW.SFunction_DIMS4_k[0]; Forward_Static_Path_length_0++) {
    loop_ub = rtDW.SFunction_DIMS2_g[0];
    for (i = 0; i < loop_ub; i++) {
      varargin_1_data[i] =
        (rtDW.Forward_Static_Path_id_h[Forward_Static_Path_length_0] ==
         rtDW.Static_Path_0[i]);
    }

    Forward_Static_Path_length = 0;
    ex = varargin_1_data[0];
    for (case_0 = 1; case_0 < rtDW.SFunction_DIMS2_g[0]; case_0++) {
      if ((int32_T)ex < (int32_T)varargin_1_data[case_0]) {
        ex = varargin_1_data[case_0];
        Forward_Static_Path_length = case_0;
      }
    }

    loop_ub = rtDW.SFunction_DIMS2_g[1];
    for (i = 0; i < loop_ub; i++) {
      rtDW.Forward_Static_Path_0_data[Forward_Static_Path_length_0 + c * i] =
        rtDW.Static_Path_0[rtDW.SFunction_DIMS2_g[0] * i +
        Forward_Static_Path_length];
    }
  }

  Static_PathCycle = 0;
  exitg1 = false;
  while ((!exitg1) && (Static_PathCycle <= (int32_T)rtU.BB_num - 1)) {
    offset_5 = (1.0 + (real_T)Static_PathCycle) * 2.0;
    for (i = 0; i < 4; i++) {
      OBXY_m[i << 1] = rtb_V_boundingbox[((int32_T)(offset_5 + -1.0) + 100 * i)
        - 1];
      OBXY_m[1 + (i << 1)] = rtb_V_boundingbox[(100 * i + (int32_T)offset_5) - 1];
    }

    case_0 = 0;
    exitg2 = false;
    while ((!exitg2) && (case_0 <= rtDW.SFunction_DIMS4_k[0] - 1)) {
      ang_1 = rtDW.Forward_Static_Path_0_data[(c << 2) + case_0] -
        rtDW.Forward_Static_Path_0_data[(c << 1) + case_0];
      Length_1 = rtDW.Forward_Static_Path_0_data[case_0 + c] -
        rtDW.Forward_Static_Path_0_data[c * 3 + case_0];
      total_length = (rtDW.Forward_Static_Path_0_data[(c << 2) + case_0] -
                      rtDW.Forward_Static_Path_0_data[(c << 1) + case_0]) *
        -rtDW.Forward_Static_Path_0_data[case_0 + c] +
        (rtDW.Forward_Static_Path_0_data[c * 3 + case_0] -
         rtDW.Forward_Static_Path_0_data[case_0 + c]) *
        rtDW.Forward_Static_Path_0_data[(c << 1) + case_0];
      offset_5 = Length_1 * Length_1;
      J_minvalue_diff = std::sqrt(ang_1 * ang_1 + offset_5);
      ang_1_0[0] = (ang_1 * OBXY_m[0] + Length_1 * OBXY_m[1]) + total_length;
      ang_1_0[1] = (ang_1 * OBXY_m[2] + Length_1 * OBXY_m[3]) + total_length;
      ang_1_0[2] = (ang_1 * OBXY_m[4] + Length_1 * OBXY_m[5]) + total_length;
      ang_1_0[3] = (ang_1 * OBXY_m[6] + Length_1 * OBXY_m[7]) + total_length;
      abs_i(ang_1_0, tmp_0);
      D[0] = tmp_0[0] / J_minvalue_diff;
      D[1] = tmp_0[1] / J_minvalue_diff;
      D[2] = tmp_0[2] / J_minvalue_diff;
      D[3] = tmp_0[3] / J_minvalue_diff;
      J_minvalue_diff = ang_1 * Length_1;
      x_endpoint3 = ang_1 * ang_1 + offset_5;
      y_endpoint3 = ang_1 * total_length;
      rtb_H_x_out[0] = ((offset_5 * OBXY_m[0] - J_minvalue_diff * OBXY_m[1]) -
                        y_endpoint3) / x_endpoint3;
      rtb_H_x_out[1] = ((offset_5 * OBXY_m[2] - J_minvalue_diff * OBXY_m[3]) -
                        y_endpoint3) / x_endpoint3;
      rtb_H_x_out[2] = ((offset_5 * OBXY_m[4] - J_minvalue_diff * OBXY_m[5]) -
                        y_endpoint3) / x_endpoint3;
      rtb_H_x_out[3] = ((offset_5 * OBXY_m[6] - J_minvalue_diff * OBXY_m[7]) -
                        y_endpoint3) / x_endpoint3;
      J_minvalue_diff = -ang_1 * Length_1;
      count_1 = ang_1 * ang_1;
      x_endpoint3 = ang_1 * ang_1 + offset_5;
      y_endpoint3 = Length_1 * total_length;
      rtb_H_y_out[0] = ((J_minvalue_diff * OBXY_m[0] + count_1 * OBXY_m[1]) -
                        y_endpoint3) / x_endpoint3;
      rtb_H_y_out[1] = ((J_minvalue_diff * OBXY_m[2] + count_1 * OBXY_m[3]) -
                        y_endpoint3) / x_endpoint3;
      rtb_H_y_out[2] = ((J_minvalue_diff * OBXY_m[4] + count_1 * OBXY_m[5]) -
                        y_endpoint3) / x_endpoint3;
      rtb_H_y_out[3] = ((J_minvalue_diff * OBXY_m[6] + count_1 * OBXY_m[7]) -
                        y_endpoint3) / x_endpoint3;
      count_1 = ((offset_5 * target_k - ang_1 * Length_1 * y) - ang_1 *
                 total_length) / (ang_1 * ang_1 + offset_5);
      Length_1 = ((-ang_1 * Length_1 * target_k + ang_1 * ang_1 * y) - Length_1 *
                  total_length) / (ang_1 * ang_1 + offset_5);
      ex = rtIsNaN(rtb_H_x_out[0]);
      if (!ex) {
        Forward_Static_Path_length = 1;
      } else {
        Forward_Static_Path_length = 0;
        break_count = 2;
        exitg3 = false;
        while ((!exitg3) && (break_count < 5)) {
          if (!rtIsNaN(rtb_H_x_out[break_count - 1])) {
            Forward_Static_Path_length = break_count;
            exitg3 = true;
          } else {
            break_count++;
          }
        }
      }

      if (Forward_Static_Path_length == 0) {
        ang_1 = rtb_H_x_out[0];
      } else {
        ang_1 = rtb_H_x_out[Forward_Static_Path_length - 1];
        while (Forward_Static_Path_length + 1 < 5) {
          if (ang_1 > rtb_H_x_out[Forward_Static_Path_length]) {
            ang_1 = rtb_H_x_out[Forward_Static_Path_length];
          }

          Forward_Static_Path_length++;
        }
      }

      if (count_1 < x_endpoint1) {
        total_length = x_endpoint1;
      } else if (rtIsNaN(count_1)) {
        if (!rtIsNaN(x_endpoint1)) {
          total_length = x_endpoint1;
        } else {
          total_length = count_1;
        }
      } else {
        total_length = count_1;
      }

      guard1 = false;
      if (ang_1 <= total_length) {
        if (!ex) {
          Forward_Static_Path_length = 1;
        } else {
          Forward_Static_Path_length = 0;
          Forward_Static_Path_length_0 = 2;
          exitg3 = false;
          while ((!exitg3) && (Forward_Static_Path_length_0 < 5)) {
            if (!rtIsNaN(rtb_H_x_out[Forward_Static_Path_length_0 - 1])) {
              Forward_Static_Path_length = Forward_Static_Path_length_0;
              exitg3 = true;
            } else {
              Forward_Static_Path_length_0++;
            }
          }
        }

        if (Forward_Static_Path_length == 0) {
          ang_1 = rtb_H_x_out[0];
        } else {
          ang_1 = rtb_H_x_out[Forward_Static_Path_length - 1];
          while (Forward_Static_Path_length + 1 < 5) {
            if (ang_1 < rtb_H_x_out[Forward_Static_Path_length]) {
              ang_1 = rtb_H_x_out[Forward_Static_Path_length];
            }

            Forward_Static_Path_length++;
          }
        }

        if (count_1 > x_endpoint1) {
          count_1 = x_endpoint1;
        } else {
          if (rtIsNaN(count_1) && (!rtIsNaN(x_endpoint1))) {
            count_1 = x_endpoint1;
          }
        }

        if (ang_1 >= count_1) {
          ex = rtIsNaN(rtb_H_y_out[0]);
          if (!ex) {
            Forward_Static_Path_length = 1;
          } else {
            Forward_Static_Path_length = 0;
            Forward_Static_Path_length_0 = 2;
            exitg3 = false;
            while ((!exitg3) && (Forward_Static_Path_length_0 < 5)) {
              if (!rtIsNaN(rtb_H_y_out[Forward_Static_Path_length_0 - 1])) {
                Forward_Static_Path_length = Forward_Static_Path_length_0;
                exitg3 = true;
              } else {
                Forward_Static_Path_length_0++;
              }
            }
          }

          if (Forward_Static_Path_length == 0) {
            ang_1 = rtb_H_y_out[0];
          } else {
            ang_1 = rtb_H_y_out[Forward_Static_Path_length - 1];
            while (Forward_Static_Path_length + 1 < 5) {
              if (ang_1 > rtb_H_y_out[Forward_Static_Path_length]) {
                ang_1 = rtb_H_y_out[Forward_Static_Path_length];
              }

              Forward_Static_Path_length++;
            }
          }

          if (Length_1 < y_endpoint1) {
            total_length = y_endpoint1;
          } else if (rtIsNaN(Length_1)) {
            if (!rtIsNaN(y_endpoint1)) {
              total_length = y_endpoint1;
            } else {
              total_length = Length_1;
            }
          } else {
            total_length = Length_1;
          }

          if (ang_1 <= total_length) {
            if (!ex) {
              Forward_Static_Path_length = 1;
            } else {
              Forward_Static_Path_length = 0;
              Forward_Static_Path_length_0 = 2;
              exitg3 = false;
              while ((!exitg3) && (Forward_Static_Path_length_0 < 5)) {
                if (!rtIsNaN(rtb_H_y_out[Forward_Static_Path_length_0 - 1])) {
                  Forward_Static_Path_length = Forward_Static_Path_length_0;
                  exitg3 = true;
                } else {
                  Forward_Static_Path_length_0++;
                }
              }
            }

            if (Forward_Static_Path_length == 0) {
              ang_1 = rtb_H_y_out[0];
            } else {
              ang_1 = rtb_H_y_out[Forward_Static_Path_length - 1];
              while (Forward_Static_Path_length + 1 < 5) {
                if (ang_1 < rtb_H_y_out[Forward_Static_Path_length]) {
                  ang_1 = rtb_H_y_out[Forward_Static_Path_length];
                }

                Forward_Static_Path_length++;
              }
            }

            if (Length_1 > y_endpoint1) {
              Length_1 = y_endpoint1;
            } else {
              if (rtIsNaN(Length_1) && (!rtIsNaN(y_endpoint1))) {
                Length_1 = y_endpoint1;
              }
            }

            if (ang_1 >= Length_1) {
              if (!rtIsNaN(D[0])) {
                Forward_Static_Path_length = 1;
              } else {
                Forward_Static_Path_length = 0;
                Forward_Static_Path_length_0 = 2;
                exitg3 = false;
                while ((!exitg3) && (Forward_Static_Path_length_0 < 5)) {
                  if (!rtIsNaN(D[Forward_Static_Path_length_0 - 1])) {
                    Forward_Static_Path_length = Forward_Static_Path_length_0;
                    exitg3 = true;
                  } else {
                    Forward_Static_Path_length_0++;
                  }
                }
              }

              if (Forward_Static_Path_length == 0) {
                ang_1 = D[0];
              } else {
                ang_1 = D[Forward_Static_Path_length - 1];
                while (Forward_Static_Path_length + 1 < 5) {
                  if (ang_1 > D[Forward_Static_Path_length]) {
                    ang_1 = D[Forward_Static_Path_length];
                  }

                  Forward_Static_Path_length++;
                }
              }

              if (ang_1 <= rtDW.Forward_Static_Path_0_data[c * 10 + case_0] /
                  2.0) {
                total_length = 1.0;
                end_ind_0 = 1;
                exitg2 = true;
              } else {
                guard1 = true;
              }
            } else {
              guard1 = true;
            }
          } else {
            guard1 = true;
          }
        } else {
          guard1 = true;
        }
      } else {
        guard1 = true;
      }

      if (guard1) {
        total_length = rtb_UnitDelay18;
        rtb_H_x_out[0] = rtDW.UnitDelay19_DSTATE[0];
        rtb_H_y_out[0] = rtDW.UnitDelay15_DSTATE[0];
        rtb_H_x_out[1] = rtDW.UnitDelay19_DSTATE[1];
        rtb_H_y_out[1] = rtDW.UnitDelay15_DSTATE[1];
        rtb_H_x_out[2] = rtDW.UnitDelay19_DSTATE[2];
        rtb_H_y_out[2] = rtDW.UnitDelay15_DSTATE[2];
        rtb_H_x_out[3] = rtDW.UnitDelay19_DSTATE[3];
        rtb_H_y_out[3] = rtDW.UnitDelay15_DSTATE[3];
        case_0++;
      }
    }

    if (end_ind_0 == 1) {
      exitg1 = true;
    } else {
      Static_PathCycle++;
    }
  }

  if (total_length == 1.0) {
    ang_1_0[0] = x_endpoint1 - rtb_H_x_out[0];
    ang_1_0[1] = x_endpoint1 - rtb_H_x_out[1];
    ang_1_0[2] = x_endpoint1 - rtb_H_x_out[2];
    ang_1_0[3] = x_endpoint1 - rtb_H_x_out[3];
    power(ang_1_0, tmp_0);
    ang_1_0[0] = y_endpoint1 - rtb_H_y_out[0];
    ang_1_0[1] = y_endpoint1 - rtb_H_y_out[1];
    ang_1_0[2] = y_endpoint1 - rtb_H_y_out[2];
    ang_1_0[3] = y_endpoint1 - rtb_H_y_out[3];
    power(ang_1_0, tmp);
    D[0] = tmp_0[0] + tmp[0];
    D[1] = tmp_0[1] + tmp[1];
    D[2] = tmp_0[2] + tmp[2];
    D[3] = tmp_0[3] + tmp[3];
    if (!rtIsNaN(D[0])) {
      Forward_Static_Path_length = 1;
    } else {
      Forward_Static_Path_length = 0;
      case_0 = 2;
      exitg1 = false;
      while ((!exitg1) && (case_0 < 5)) {
        if (!rtIsNaN(D[case_0 - 1])) {
          Forward_Static_Path_length = case_0;
          exitg1 = true;
        } else {
          case_0++;
        }
      }
    }

    if (Forward_Static_Path_length == 0) {
      ang_1 = D[0];
    } else {
      ang_1 = D[Forward_Static_Path_length - 1];
      while (Forward_Static_Path_length + 1 < 5) {
        if (ang_1 > D[Forward_Static_Path_length]) {
          ang_1 = D[Forward_Static_Path_length];
        }

        Forward_Static_Path_length++;
      }
    }

    if (std::sqrt(ang_1) > (rtb_Forward_length_final + count) +
        4.666666666666667) {
      total_length = 0.0;
    }
  }

  if (total_length == 1.0) {
    rtb_UnitDelay18 = 100.0;
    count = rtU.w_off_avoid;
  } else {
    rtb_UnitDelay18 = rtDW.UnitDelay17_DSTATE - 1.0;
    if (rtDW.UnitDelay17_DSTATE - 1.0 < 0.0) {
      rtb_UnitDelay18 = 0.0;
    }

    count = rtU.w_off_;
  }

  // SignalConversion: '<S4>/TmpSignal ConversionAt SFunction Inport7' incorporates:
  //   Gain: '<S1>/Gain1'
  //   Inport: '<Root>/X_UKF_SLAM_i1'
  //   MATLAB Function: '<S1>/DynamicPathPlanning'
  //   MATLAB Function: '<S1>/MATLAB Function'

  rtb_TmpSignalConversionAtSFun_n[0] = rtU.X_UKF_SLAM_i1[0];
  rtb_TmpSignalConversionAtSFun_n[1] = rtU.X_UKF_SLAM_i1[1];
  rtb_TmpSignalConversionAtSFun_n[2] = 0.017453292519943295 * vehicle_heading;

  // MATLAB Function: '<S1>/DynamicPathPlanning' incorporates:
  //   Constant: '<S1>/Constant12'
  //   Constant: '<S1>/Constant13'
  //   Constant: '<S1>/Constant16'
  //   Inport: '<Root>/BB_num'
  //   Inport: '<Root>/Freespace'
  //   Inport: '<Root>/Freespace_mode'
  //   Inport: '<Root>/W_1'
  //   Inport: '<Root>/X_UKF_SLAM_i1'
  //   Inport: '<Root>/safe_range'
  //   Inport: '<Root>/seg_Curvature_i'
  //   MATLAB Function: '<S1>/DangerousArea1'
  //   MATLAB Function: '<S1>/EndPointDecision'
  //   MATLAB Function: '<S1>/MATLAB Function'
  //   UnitDelay: '<S1>/Unit Delay5'

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    varargin_1_data[i] = (rtDW.Forward_Static_Path_id_h[rtDW.SFunction_DIMS4_k[0]
                          - 1] == rtDW.Static_Path_0[i]);
  }

  Forward_Static_Path_length = rtDW.SFunction_DIMS2_g[0] - 1;
  case_0 = 0;
  for (Forward_Static_Path_length_0 = 0; Forward_Static_Path_length_0 <=
       Forward_Static_Path_length; Forward_Static_Path_length_0++) {
    if (varargin_1_data[Forward_Static_Path_length_0]) {
      case_0++;
    }
  }

  Forward_Static_Path_length_0 = 0;
  for (break_count = 0; break_count <= Forward_Static_Path_length; break_count++)
  {
    if (varargin_1_data[break_count]) {
      t_data[Forward_Static_Path_length_0] = break_count + 1;
      Forward_Static_Path_length_0++;
    }
  }

  for (i = 0; i < case_0; i++) {
    rtDW.seg_id_data[i] = rtDW.Static_Path_0[(rtDW.SFunction_DIMS2_g[0] * 7 +
      t_data[i]) - 1] * 3.1415926535897931 / 180.0;
  }

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    varargin_1_data[i] = (rtDW.Forward_Static_Path_id_h[rtDW.SFunction_DIMS4_k[0]
                          - 1] == rtDW.Static_Path_0[i]);
  }

  Forward_Static_Path_length_0 = 0;
  for (case_0 = 0; case_0 < rtDW.SFunction_DIMS2_g[0]; case_0++) {
    if (varargin_1_data[case_0]) {
      u_data[Forward_Static_Path_length_0] = case_0 + 1;
      Forward_Static_Path_length_0++;
    }
  }

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    varargin_1_data[i] = (rtDW.Forward_Static_Path_id_h[rtDW.SFunction_DIMS4_k[0]
                          - 1] == rtDW.Static_Path_0[i]);
  }

  case_0 = 0;
  for (break_count = 0; break_count < rtDW.SFunction_DIMS2_g[0]; break_count++)
  {
    if (varargin_1_data[break_count]) {
      v_data[case_0] = break_count + 1;
      case_0++;
    }
  }

  target_k = rtDW.Static_Path_0[(rtDW.SFunction_DIMS2_g[0] * 10 + v_data[0]) - 1]
    / 4.0;
  ang_1 = target_k * 2.0;
  J_minvalue_diff = target_k * 3.0;
  Length_1 = target_k * 4.0;
  offset_5 = target_k * 5.0;
  count_1 = target_k * 6.0;
  offset[0] = count_1;
  offset[1] = offset_5;
  offset[2] = Length_1;
  offset[3] = J_minvalue_diff;
  offset[4] = ang_1;
  offset[5] = target_k;
  offset[6] = 0.0;
  offset[7] = target_k;
  offset[8] = ang_1;
  offset[9] = J_minvalue_diff;
  offset[10] = Length_1;
  offset[11] = offset_5;
  offset[12] = count_1;
  x_endpoint6 = std::cos(rtDW.seg_id_data[0] + 1.5707963267948966);
  x_endpoint1 = x_endpoint6 * count_1 + End_x;
  y_endpoint6 = std::sin(rtDW.seg_id_data[0] + 1.5707963267948966);
  y_endpoint1 = y_endpoint6 * count_1 + End_y;
  x_endpoint2 = x_endpoint6 * offset_5 + End_x;
  y = y_endpoint6 * offset_5 + End_y;
  x_endpoint3 = x_endpoint6 * Length_1 + End_x;
  y_endpoint3 = y_endpoint6 * Length_1 + End_y;
  x_endpoint4 = x_endpoint6 * J_minvalue_diff + End_x;
  y_endpoint4 = y_endpoint6 * J_minvalue_diff + End_y;
  x_endpoint5 = x_endpoint6 * ang_1 + End_x;
  y_endpoint5 = y_endpoint6 * ang_1 + End_y;
  x_endpoint6 = x_endpoint6 * target_k + End_x;
  y_endpoint6 = y_endpoint6 * target_k + End_y;
  x_endpoint13 = std::cos(rtDW.seg_id_data[0] - 1.5707963267948966);
  x_endpoint8 = x_endpoint13 * target_k + End_x;
  xy_end_point_idx_0 = std::sin(rtDW.seg_id_data[0] - 1.5707963267948966);
  y_endpoint8 = xy_end_point_idx_0 * target_k + End_y;
  x_endpoint9 = x_endpoint13 * ang_1 + End_x;
  y_endpoint9 = xy_end_point_idx_0 * ang_1 + End_y;
  x_endpoint10 = x_endpoint13 * J_minvalue_diff + End_x;
  y_endpoint10 = xy_end_point_idx_0 * J_minvalue_diff + End_y;
  x_endpoint11 = x_endpoint13 * Length_1 + End_x;
  y_endpoint11 = xy_end_point_idx_0 * Length_1 + End_y;
  x_endpoint12 = x_endpoint13 * offset_5 + End_x;
  y_endpoint12 = xy_end_point_idx_0 * offset_5 + End_y;
  x_endpoint13 = x_endpoint13 * count_1 + End_x;
  count_1 = xy_end_point_idx_0 * count_1 + End_y;
  G2splines(rtU.X_UKF_SLAM_i1[0], rtU.X_UKF_SLAM_i1[1],
            rtb_TmpSignalConversionAtSFun_n[2], rtU.seg_Curvature_i, x_endpoint1,
            y_endpoint1, rtDW.seg_id_data[0], rtDW.Static_Path_0[(u_data[0] +
             rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_target, X1, b_Path_dis_data,
            XP1, YP1, K1, K_11, &rtb_J_out_a[0]);
  G2splines(rtU.X_UKF_SLAM_i1[0], rtU.X_UKF_SLAM_i1[1],
            rtb_TmpSignalConversionAtSFun_n[2], rtU.seg_Curvature_i, x_endpoint2,
            y, rtDW.seg_id_data[0], rtDW.Static_Path_0[(u_data[0] +
             rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_target, X2, Y2, XP2, YP2,
            K2, K_12, &rtb_J_out_a[1]);
  G2splines(rtU.X_UKF_SLAM_i1[0], rtU.X_UKF_SLAM_i1[1],
            rtb_TmpSignalConversionAtSFun_n[2], rtU.seg_Curvature_i, x_endpoint3,
            y_endpoint3, rtDW.seg_id_data[0], rtDW.Static_Path_0[(u_data[0] +
             rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_target, X3, Y3, XP3, YP3,
            K3, K_13, &rtb_J_out_a[2]);
  G2splines(rtU.X_UKF_SLAM_i1[0], rtU.X_UKF_SLAM_i1[1],
            rtb_TmpSignalConversionAtSFun_n[2], rtU.seg_Curvature_i, x_endpoint4,
            y_endpoint4, rtDW.seg_id_data[0], rtDW.Static_Path_0[(u_data[0] +
             rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_target, X4, Y4, XP4, YP4,
            K4, K_14, &rtb_J_out_a[3]);
  G2splines(rtU.X_UKF_SLAM_i1[0], rtU.X_UKF_SLAM_i1[1],
            rtb_TmpSignalConversionAtSFun_n[2], rtU.seg_Curvature_i, x_endpoint5,
            y_endpoint5, rtDW.seg_id_data[0], rtDW.Static_Path_0[(u_data[0] +
             rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_target, X5, Y5, XP5, YP5,
            K5, K_15, &rtb_J_out_a[4]);
  G2splines(rtU.X_UKF_SLAM_i1[0], rtU.X_UKF_SLAM_i1[1],
            rtb_TmpSignalConversionAtSFun_n[2], rtU.seg_Curvature_i, x_endpoint6,
            y_endpoint6, rtDW.seg_id_data[0], rtDW.Static_Path_0[(u_data[0] +
             rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_target, X6, Y6, XP6, YP6,
            K6, K_16, &rtb_J_out_a[5]);
  G2splines(rtU.X_UKF_SLAM_i1[0], rtU.X_UKF_SLAM_i1[1],
            rtb_TmpSignalConversionAtSFun_n[2], rtU.seg_Curvature_i, End_x,
            End_y, rtDW.seg_id_data[0], rtDW.Static_Path_0[(u_data[0] +
             rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_target, X7, Y7, XP7, YP7,
            K7, K_17, &rtb_J_out_a[6]);
  G2splines(rtU.X_UKF_SLAM_i1[0], rtU.X_UKF_SLAM_i1[1],
            rtb_TmpSignalConversionAtSFun_n[2], rtU.seg_Curvature_i, x_endpoint8,
            y_endpoint8, rtDW.seg_id_data[0], rtDW.Static_Path_0[(u_data[0] +
             rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_target, X8, Y8, XP8, YP8,
            K8, K_18, &rtb_J_out_a[7]);
  G2splines(rtU.X_UKF_SLAM_i1[0], rtU.X_UKF_SLAM_i1[1],
            rtb_TmpSignalConversionAtSFun_n[2], rtU.seg_Curvature_i, x_endpoint9,
            y_endpoint9, rtDW.seg_id_data[0], rtDW.Static_Path_0[(u_data[0] +
             rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_target, X9, Y9, XP9, YP9,
            K9, K_19, &rtb_J_out_a[8]);
  G2splines(rtU.X_UKF_SLAM_i1[0], rtU.X_UKF_SLAM_i1[1],
            rtb_TmpSignalConversionAtSFun_n[2], rtU.seg_Curvature_i,
            x_endpoint10, y_endpoint10, rtDW.seg_id_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_target, X10,
            Y10, XP10, YP10, K10, K_110, &rtb_J_out_a[9]);
  G2splines(rtU.X_UKF_SLAM_i1[0], rtU.X_UKF_SLAM_i1[1],
            rtb_TmpSignalConversionAtSFun_n[2], rtU.seg_Curvature_i,
            x_endpoint11, y_endpoint11, rtDW.seg_id_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_target, X11,
            Y11, XP11, YP11, K11, K_111, &rtb_J_out_a[10]);
  G2splines(rtU.X_UKF_SLAM_i1[0], rtU.X_UKF_SLAM_i1[1],
            rtb_TmpSignalConversionAtSFun_n[2], rtU.seg_Curvature_i,
            x_endpoint12, y_endpoint12, rtDW.seg_id_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_target, X12,
            Y12, XP12, YP12, K12, K_112, &rtb_J_out_a[11]);
  G2splines(rtU.X_UKF_SLAM_i1[0], rtU.X_UKF_SLAM_i1[1],
            rtb_TmpSignalConversionAtSFun_n[2], rtU.seg_Curvature_i,
            x_endpoint13, count_1, rtDW.seg_id_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_target, X13,
            Y13, XP13, YP13, K13, K_113, &rtb_J_out_a[12]);
  for (i = 0; i < 11; i++) {
    X_2[i] = X1[i];
    X_2[i + 11] = X2[i];
    X_2[i + 22] = X3[i];
    X_2[i + 33] = X4[i];
    X_2[i + 44] = X5[i];
    X_2[i + 55] = X6[i];
    X_2[i + 66] = X7[i];
    X_2[i + 77] = X8[i];
    X_2[i + 88] = X9[i];
    X_2[i + 99] = X10[i];
    X_2[i + 110] = X11[i];
    X_2[i + 121] = X12[i];
    X_2[i + 132] = X13[i];
    Y[i] = b_Path_dis_data[i];
    Y[i + 11] = Y2[i];
    Y[i + 22] = Y3[i];
    Y[i + 33] = Y4[i];
    Y[i + 44] = Y5[i];
    Y[i + 55] = Y6[i];
    Y[i + 66] = Y7[i];
    Y[i + 77] = Y8[i];
    Y[i + 88] = Y9[i];
    Y[i + 99] = Y10[i];
    Y[i + 110] = Y11[i];
    Y[i + 121] = Y12[i];
    Y[i + 132] = Y13[i];
  }

  for (i = 0; i < 6; i++) {
    rtb_XP_i[i] = XP1[i];
    rtb_XP_i[i + 6] = XP2[i];
    rtb_XP_i[i + 12] = XP3[i];
    rtb_XP_i[i + 18] = XP4[i];
    rtb_XP_i[i + 24] = XP5[i];
    rtb_XP_i[i + 30] = XP6[i];
    rtb_XP_i[i + 36] = XP7[i];
    rtb_XP_i[i + 42] = XP8[i];
    rtb_XP_i[i + 48] = XP9[i];
    rtb_XP_i[i + 54] = XP10[i];
    rtb_XP_i[i + 60] = XP11[i];
    rtb_XP_i[i + 66] = XP12[i];
    rtb_XP_i[i + 72] = XP13[i];
    rtb_YP_p[i] = YP1[i];
    rtb_YP_p[i + 6] = YP2[i];
    rtb_YP_p[i + 12] = YP3[i];
    rtb_YP_p[i + 18] = YP4[i];
    rtb_YP_p[i + 24] = YP5[i];
    rtb_YP_p[i + 30] = YP6[i];
    rtb_YP_p[i + 36] = YP7[i];
    rtb_YP_p[i + 42] = YP8[i];
    rtb_YP_p[i + 48] = YP9[i];
    rtb_YP_p[i + 54] = YP10[i];
    rtb_YP_p[i + 60] = YP11[i];
    rtb_YP_p[i + 66] = YP12[i];
    rtb_YP_p[i + 72] = YP13[i];
  }

  for (i = 0; i < 11; i++) {
    K[i] = K1[i];
    K[i + 11] = K2[i];
    K[i + 22] = K3[i];
    K[i + 33] = K4[i];
    K[i + 44] = K5[i];
    K[i + 55] = K6[i];
    K[i + 66] = K7[i];
    K[i + 77] = K8[i];
    K[i + 88] = K9[i];
    K[i + 99] = K10[i];
    K[i + 110] = K11[i];
    K[i + 121] = K12[i];
    K[i + 132] = K13[i];
    K_1[i] = K_11[i];
    K_1[i + 11] = K_12[i];
    K_1[i + 22] = K_13[i];
    K_1[i + 33] = K_14[i];
    K_1[i + 44] = K_15[i];
    K_1[i + 55] = K_16[i];
    K_1[i + 66] = K_17[i];
    K_1[i + 77] = K_18[i];
    K_1[i + 88] = K_19[i];
    K_1[i + 99] = K_110[i];
    K_1[i + 110] = K_111[i];
    K_1[i + 121] = K_112[i];
    K_1[i + 132] = K_113[i];
  }

  xy_end_point_idx_0 = x_endpoint1;
  xy_end_point_idx_2 = x_endpoint2;
  xy_end_point_idx_1 = y_endpoint1;
  xy_end_point_idx_25 = count_1;
  memset(&Path_col[0], 0, 52U * sizeof(real_T));
  for (i = 0; i < 5; i++) {
    Path_col[3 + ((8 + i) << 2)] = 1.0;
  }

  Path_col[3] = 1.0;
  Path_col[51] = 1.0;
  if ((rtU.Freespace_mode == 0.0) || (rtU.Freespace_mode == 2.0)) {
    memcpy(&OBXY_EL[0], &rtb_V_boundingbox[0], 400U * sizeof(real_T));
    for (Static_PathCycle = 0; Static_PathCycle < (int32_T)rtU.BB_num;
         Static_PathCycle++) {
      offset_5 = (1.0 + (real_T)Static_PathCycle) * 2.0;
      i = (int32_T)(offset_5 + -1.0);
      Forward_Static_Path_length = i - 1;
      OBXY_EL[Forward_Static_Path_length] =
        ((rtb_V_boundingbox[Forward_Static_Path_length] - rtb_V_boundingbox[i +
          99]) * 0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
           Static_PathCycle) * 2.0 + -1.0) - 1]) + (rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0 + -1.0) - 1] -
        rtb_V_boundingbox[i + 299]) * 0.3;
      Forward_Static_Path_length = (int32_T)offset_5;
      Forward_Static_Path_length_0 = Forward_Static_Path_length - 1;
      OBXY_EL[Forward_Static_Path_length_0] =
        ((rtb_V_boundingbox[Forward_Static_Path_length_0] -
          rtb_V_boundingbox[Forward_Static_Path_length + 99]) * 0.15 +
         rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0) - 1])
        + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0) -
           1] - rtb_V_boundingbox[Forward_Static_Path_length + 299]) * 0.3;
      OBXY_EL[(int32_T)(offset_5 + -1.0) + 99] = ((rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0 + -1.0) + 99] -
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0 +
        -1.0) - 1]) * 0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
        Static_PathCycle) * 2.0 + -1.0) + 99]) + (rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0 + -1.0) + 99] -
        rtb_V_boundingbox[i + 199]) * 0.3;
      OBXY_EL[(int32_T)offset_5 + 99] = ((rtb_V_boundingbox[(int32_T)((1.0 +
        (real_T)Static_PathCycle) * 2.0) + 99] - rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0) - 1]) * 0.15 +
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0) + 99])
        + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0) +
           99] - rtb_V_boundingbox[Forward_Static_Path_length + 199]) * 0.3;
      OBXY_EL[(int32_T)(offset_5 + -1.0) + 199] = ((rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0 + -1.0) + 199] -
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0 +
        -1.0) + 299]) * 0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
        Static_PathCycle) * 2.0 + -1.0) + 199]) + (rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0 + -1.0) + 199] -
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0 +
        -1.0) + 99]) * 0.3;
      OBXY_EL[(int32_T)offset_5 + 199] = ((rtb_V_boundingbox[(int32_T)((1.0 +
        (real_T)Static_PathCycle) * 2.0) + 199] - rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0) + 299]) * 0.15 +
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0) +
        199]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) *
                  2.0) + 199] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
        Static_PathCycle) * 2.0) + 99]) * 0.3;
      OBXY_EL[(int32_T)(offset_5 + -1.0) + 299] = ((rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0 + -1.0) + 299] -
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0 +
        -1.0) + 199]) * 0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
        Static_PathCycle) * 2.0 + -1.0) + 299]) + (rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0 + -1.0) + 299] -
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0 +
        -1.0) - 1]) * 0.3;
      OBXY_EL[(int32_T)offset_5 + 299] = ((rtb_V_boundingbox[(int32_T)((1.0 +
        (real_T)Static_PathCycle) * 2.0) + 299] - rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0) + 199]) * 0.15 +
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0) +
        299]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) *
                  2.0) + 299] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
        Static_PathCycle) * 2.0) - 1]) * 0.3;
    }

    for (i = 0; i < 13; i++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 10;
           Path_RES_0_size_idx_1++) {
        Forward_Static_Path_length = 11 * i + Path_RES_0_size_idx_1;
        target_k = X_2[Forward_Static_Path_length + 1] -
          X_2[Forward_Static_Path_length];
        X_diff[Path_RES_0_size_idx_1 + 11 * i] = target_k;
        X_diff_0[Path_RES_0_size_idx_1 + 10 * i] = target_k;
      }

      Forward_Static_Path_length_0 = 10 + 11 * i;
      X_diff[Forward_Static_Path_length_0] = X_diff_0[10 * i + 9];
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 10;
           Path_RES_0_size_idx_1++) {
        Forward_Static_Path_length = 11 * i + Path_RES_0_size_idx_1;
        target_k = Y[Forward_Static_Path_length + 1] -
          Y[Forward_Static_Path_length];
        Y_diff[Path_RES_0_size_idx_1 + 11 * i] = target_k;
        X_diff_0[Path_RES_0_size_idx_1 + 10 * i] = target_k;
      }

      Y_diff[Forward_Static_Path_length_0] = X_diff_0[10 * i + 9];
    }

    power_bv(X_diff, XY_difflen);
    power_bv(Y_diff, Path_vehFLY);
    for (i = 0; i < 143; i++) {
      Path_vehFLX[i] = XY_difflen[i] + Path_vehFLY[i];
    }

    power_bvw(Path_vehFLX, XY_difflen);
    for (i = 0; i < 143; i++) {
      target_k = X_diff[i] / XY_difflen[i];
      x_endpoint1 = Y_diff[i] / XY_difflen[i];
      y_endpoint1 = 1.1 * -x_endpoint1 + X_2[i];
      Path_vehFLX[i] = y_endpoint1 + 1.4000000000000001 * target_k;
      ang_1 = 1.1 * target_k + Y[i];
      Path_vehFLY[i] = ang_1 + 1.4000000000000001 * x_endpoint1;
      J_minvalue_diff = X_2[i] - 1.1 * -x_endpoint1;
      Path_vehFRX[i] = J_minvalue_diff + 1.4000000000000001 * target_k;
      Length_1 = Y[i] - 1.1 * target_k;
      Path_vehFRY[i] = Length_1 + 1.4000000000000001 * x_endpoint1;
      Path_vehRLX[i] = y_endpoint1 - 5.6000000000000005 * target_k;
      Path_vehRLY[i] = ang_1 - 5.6000000000000005 * x_endpoint1;
      Path_vehRRX[i] = J_minvalue_diff - 5.6000000000000005 * target_k;
      Path_vehRRY[i] = Length_1 - 5.6000000000000005 * x_endpoint1;
      X_diff[i] = target_k;
      XY_difflen[i] = -x_endpoint1;
      Y_diff[i] = x_endpoint1;
    }

    for (Forward_Static_Path_length_0 = 0; Forward_Static_Path_length_0 < 13;
         Forward_Static_Path_length_0++) {
      Path_col[Forward_Static_Path_length_0 << 2] = 0.0;
      if (!(Path_col[(Forward_Static_Path_length_0 << 2) + 3] == 1.0)) {
        case_0 = 0;
        exitg1 = false;
        while ((!exitg1) && (case_0 < 11)) {
          loop_ub = 11 * Forward_Static_Path_length_0 + case_0;
          OBXY_m[0] = Path_vehFLX[loop_ub];
          OBXY_m[2] = Path_vehFRX[loop_ub];
          OBXY_m[4] = Path_vehRLX[loop_ub];
          OBXY_m[6] = Path_vehRRX[loop_ub];
          OBXY_m[1] = Path_vehFLY[loop_ub];
          OBXY_m[3] = Path_vehFRY[loop_ub];
          OBXY_m[5] = Path_vehRLY[loop_ub];
          OBXY_m[7] = Path_vehRRY[loop_ub];
          break_count = 0;
          exitg2 = false;
          while ((!exitg2) && (break_count <= (int32_T)rtU.BB_num - 1)) {
            y_endpoint1 = (1.0 + (real_T)break_count) * 2.0;
            i = (int32_T)(y_endpoint1 + -1.0);
            x_endpoint1 = OBXY_EL[i + 99] - OBXY_EL[i - 1];
            target_k = std::sqrt(x_endpoint1 * x_endpoint1 + x_endpoint1 *
                                 x_endpoint1);
            Forward_Static_Path_length = (int32_T)y_endpoint1;
            count_1 = -(OBXY_EL[Forward_Static_Path_length + 99] -
                        OBXY_EL[Forward_Static_Path_length - 1]) / target_k;
            target_k = x_endpoint1 / target_k;
            offset_5 = OBXY_EL[Forward_Static_Path_length + 199] - OBXY_EL
              [(int32_T)((1.0 + (real_T)break_count) * 2.0) + 99];
            x_endpoint1 = OBXY_EL[i + 199] - OBXY_EL[(int32_T)((1.0 + (real_T)
              break_count) * 2.0 + -1.0) + 99];
            J_minvalue_diff = std::sqrt(offset_5 * offset_5 + x_endpoint1 *
              x_endpoint1);
            Length_1 = -offset_5 / J_minvalue_diff;
            x_endpoint1 /= J_minvalue_diff;
            rtb_TmpSignalConversionAtSFun_1[0] = count_1;
            rtb_TmpSignalConversionAtSFun_1[1] = Length_1;
            rtb_TmpSignalConversionAtSFun_1[4] = target_k;
            rtb_TmpSignalConversionAtSFun_1[5] = x_endpoint1;
            rtb_TmpSignalConversionAtSFun_1[2] = X_diff[loop_ub];
            rtb_TmpSignalConversionAtSFun_1[6] = Y_diff[loop_ub];
            rtb_TmpSignalConversionAtSFun_1[3] = XY_difflen[loop_ub];
            rtb_TmpSignalConversionAtSFun_1[7] = X_diff[11 *
              Forward_Static_Path_length_0 + case_0];
            rtb_TmpSignalConversionAtSFun_2[0] = count_1;
            rtb_TmpSignalConversionAtSFun_2[1] = Length_1;
            rtb_TmpSignalConversionAtSFun_2[4] = target_k;
            rtb_TmpSignalConversionAtSFun_2[5] = x_endpoint1;
            rtb_TmpSignalConversionAtSFun_2[2] = X_diff[11 *
              Forward_Static_Path_length_0 + case_0];
            rtb_TmpSignalConversionAtSFun_2[6] = Y_diff[11 *
              Forward_Static_Path_length_0 + case_0];
            rtb_TmpSignalConversionAtSFun_2[3] = XY_difflen[11 *
              Forward_Static_Path_length_0 + case_0];
            rtb_TmpSignalConversionAtSFun_2[7] = X_diff[11 *
              Forward_Static_Path_length_0 + case_0];
            for (i = 0; i < 4; i++) {
              for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 4;
                   Path_RES_0_size_idx_1++) {
                proj_veh[i + (Path_RES_0_size_idx_1 << 2)] = 0.0;
                proj_veh[i + (Path_RES_0_size_idx_1 << 2)] +=
                  OBXY_m[Path_RES_0_size_idx_1 << 1] *
                  rtb_TmpSignalConversionAtSFun_1[i];
                proj_veh[i + (Path_RES_0_size_idx_1 << 2)] += OBXY_m
                  [(Path_RES_0_size_idx_1 << 1) + 1] *
                  rtb_TmpSignalConversionAtSFun_1[i + 4];
              }

              OBXY_EL_0[i << 1] = OBXY_EL[((int32_T)(y_endpoint1 + -1.0) + 100 *
                i) - 1];
              OBXY_EL_0[1 + (i << 1)] = OBXY_EL[(100 * i + (int32_T)y_endpoint1)
                - 1];
            }

            for (Static_PathCycle = 0; Static_PathCycle < 4; Static_PathCycle++)
            {
              for (i = 0; i < 4; i++) {
                proj_ob[Static_PathCycle + (i << 2)] = 0.0;
                proj_ob[Static_PathCycle + (i << 2)] += OBXY_EL_0[i << 1] *
                  rtb_TmpSignalConversionAtSFun_2[Static_PathCycle];
                proj_ob[Static_PathCycle + (i << 2)] += OBXY_EL_0[(i << 1) + 1] *
                  rtb_TmpSignalConversionAtSFun_2[Static_PathCycle + 4];
              }

              D[Static_PathCycle] = proj_veh[Static_PathCycle];
            }

            target_k = proj_veh[0];
            x_endpoint1 = proj_veh[1];
            y_endpoint1 = proj_veh[2];
            ang_1 = proj_veh[3];
            for (Forward_Static_Path_length = 0; Forward_Static_Path_length < 3;
                 Forward_Static_Path_length++) {
              if ((!rtIsNaN(proj_veh[(Forward_Static_Path_length + 1) << 2])) &&
                  (rtIsNaN(D[0]) || (D[0] > proj_veh[(Forward_Static_Path_length
                     + 1) << 2]))) {
                D[0] = proj_veh[(Forward_Static_Path_length + 1) << 2];
              }

              if ((!rtIsNaN(proj_veh[((Forward_Static_Path_length + 1) << 2) + 1]))
                  && (rtIsNaN(D[1]) || (D[1] > proj_veh
                    [((Forward_Static_Path_length + 1) << 2) + 1]))) {
                D[1] = proj_veh[((Forward_Static_Path_length + 1) << 2) + 1];
              }

              if ((!rtIsNaN(proj_veh[((Forward_Static_Path_length + 1) << 2) + 2]))
                  && (rtIsNaN(D[2]) || (D[2] > proj_veh
                    [((Forward_Static_Path_length + 1) << 2) + 2]))) {
                D[2] = proj_veh[((Forward_Static_Path_length + 1) << 2) + 2];
              }

              if ((!rtIsNaN(proj_veh[((Forward_Static_Path_length + 1) << 2) + 3]))
                  && (rtIsNaN(D[3]) || (D[3] > proj_veh
                    [((Forward_Static_Path_length + 1) << 2) + 3]))) {
                D[3] = proj_veh[((Forward_Static_Path_length + 1) << 2) + 3];
              }

              J_minvalue_diff = target_k;
              if ((!rtIsNaN(proj_veh[(Forward_Static_Path_length + 1) << 2])) &&
                  (rtIsNaN(target_k) || (target_k < proj_veh
                    [(Forward_Static_Path_length + 1) << 2]))) {
                J_minvalue_diff = proj_veh[(Forward_Static_Path_length + 1) << 2];
              }

              target_k = J_minvalue_diff;
              J_minvalue_diff = x_endpoint1;
              if ((!rtIsNaN(proj_veh[((Forward_Static_Path_length + 1) << 2) + 1]))
                  && (rtIsNaN(x_endpoint1) || (x_endpoint1 < proj_veh
                    [((Forward_Static_Path_length + 1) << 2) + 1]))) {
                J_minvalue_diff = proj_veh[((Forward_Static_Path_length + 1) <<
                  2) + 1];
              }

              x_endpoint1 = J_minvalue_diff;
              J_minvalue_diff = y_endpoint1;
              if ((!rtIsNaN(proj_veh[((Forward_Static_Path_length + 1) << 2) + 2]))
                  && (rtIsNaN(y_endpoint1) || (y_endpoint1 < proj_veh
                    [((Forward_Static_Path_length + 1) << 2) + 2]))) {
                J_minvalue_diff = proj_veh[((Forward_Static_Path_length + 1) <<
                  2) + 2];
              }

              y_endpoint1 = J_minvalue_diff;
              J_minvalue_diff = ang_1;
              if ((!rtIsNaN(proj_veh[((Forward_Static_Path_length + 1) << 2) + 3]))
                  && (rtIsNaN(ang_1) || (ang_1 < proj_veh
                    [((Forward_Static_Path_length + 1) << 2) + 3]))) {
                J_minvalue_diff = proj_veh[((Forward_Static_Path_length + 1) <<
                  2) + 3];
              }

              ang_1 = J_minvalue_diff;
            }

            minmax_veh[0] = D[0];
            minmax_veh[4] = target_k;
            minmax_veh[1] = D[1];
            minmax_veh[5] = x_endpoint1;
            minmax_veh[2] = D[2];
            minmax_veh[6] = y_endpoint1;
            minmax_veh[3] = D[3];
            minmax_veh[7] = ang_1;
            D[0] = proj_ob[0];
            D[1] = proj_ob[1];
            D[2] = proj_ob[2];
            D[3] = proj_ob[3];
            target_k = proj_ob[0];
            x_endpoint1 = proj_ob[1];
            y_endpoint1 = proj_ob[2];
            ang_1 = proj_ob[3];
            for (Forward_Static_Path_length = 0; Forward_Static_Path_length < 3;
                 Forward_Static_Path_length++) {
              if ((!rtIsNaN(proj_ob[(Forward_Static_Path_length + 1) << 2])) &&
                  (rtIsNaN(D[0]) || (D[0] > proj_ob[(Forward_Static_Path_length
                     + 1) << 2]))) {
                D[0] = proj_ob[(Forward_Static_Path_length + 1) << 2];
              }

              if ((!rtIsNaN(proj_ob[((Forward_Static_Path_length + 1) << 2) + 1]))
                  && (rtIsNaN(D[1]) || (D[1] > proj_ob
                    [((Forward_Static_Path_length + 1) << 2) + 1]))) {
                D[1] = proj_ob[((Forward_Static_Path_length + 1) << 2) + 1];
              }

              if ((!rtIsNaN(proj_ob[((Forward_Static_Path_length + 1) << 2) + 2]))
                  && (rtIsNaN(D[2]) || (D[2] > proj_ob
                    [((Forward_Static_Path_length + 1) << 2) + 2]))) {
                D[2] = proj_ob[((Forward_Static_Path_length + 1) << 2) + 2];
              }

              if ((!rtIsNaN(proj_ob[((Forward_Static_Path_length + 1) << 2) + 3]))
                  && (rtIsNaN(D[3]) || (D[3] > proj_ob
                    [((Forward_Static_Path_length + 1) << 2) + 3]))) {
                D[3] = proj_ob[((Forward_Static_Path_length + 1) << 2) + 3];
              }

              J_minvalue_diff = target_k;
              if ((!rtIsNaN(proj_ob[(Forward_Static_Path_length + 1) << 2])) &&
                  (rtIsNaN(target_k) || (target_k < proj_ob
                    [(Forward_Static_Path_length + 1) << 2]))) {
                J_minvalue_diff = proj_ob[(Forward_Static_Path_length + 1) << 2];
              }

              target_k = J_minvalue_diff;
              J_minvalue_diff = x_endpoint1;
              if ((!rtIsNaN(proj_ob[((Forward_Static_Path_length + 1) << 2) + 1]))
                  && (rtIsNaN(x_endpoint1) || (x_endpoint1 < proj_ob
                    [((Forward_Static_Path_length + 1) << 2) + 1]))) {
                J_minvalue_diff = proj_ob[((Forward_Static_Path_length + 1) << 2)
                  + 1];
              }

              x_endpoint1 = J_minvalue_diff;
              J_minvalue_diff = y_endpoint1;
              if ((!rtIsNaN(proj_ob[((Forward_Static_Path_length + 1) << 2) + 2]))
                  && (rtIsNaN(y_endpoint1) || (y_endpoint1 < proj_ob
                    [((Forward_Static_Path_length + 1) << 2) + 2]))) {
                J_minvalue_diff = proj_ob[((Forward_Static_Path_length + 1) << 2)
                  + 2];
              }

              y_endpoint1 = J_minvalue_diff;
              J_minvalue_diff = ang_1;
              if ((!rtIsNaN(proj_ob[((Forward_Static_Path_length + 1) << 2) + 3]))
                  && (rtIsNaN(ang_1) || (ang_1 < proj_ob
                    [((Forward_Static_Path_length + 1) << 2) + 3]))) {
                J_minvalue_diff = proj_ob[((Forward_Static_Path_length + 1) << 2)
                  + 3];
              }

              ang_1 = J_minvalue_diff;
            }

            minmax_obj[0] = D[0];
            minmax_obj[4] = target_k;
            minmax_obj[1] = D[1];
            minmax_obj[5] = x_endpoint1;
            minmax_obj[2] = D[2];
            minmax_obj[6] = y_endpoint1;
            minmax_obj[3] = D[3];
            minmax_obj[7] = ang_1;
            Static_PathCycle = 0;
            exitg3 = false;
            while ((!exitg3) && (Static_PathCycle < 4)) {
              if (minmax_veh[Static_PathCycle] > minmax_obj[4 + Static_PathCycle])
              {
                Path_col[Forward_Static_Path_length_0 << 2] = 0.0;
                exitg3 = true;
              } else if (minmax_veh[4 + Static_PathCycle] <
                         minmax_obj[Static_PathCycle]) {
                Path_col[Forward_Static_Path_length_0 << 2] = 0.0;
                exitg3 = true;
              } else {
                Path_col[Forward_Static_Path_length_0 << 2] = 1.0;
                Static_PathCycle++;
              }
            }

            if (Path_col[Forward_Static_Path_length_0 << 2] == 1.0) {
              Path_col[2 + (Forward_Static_Path_length_0 << 2)] = 1.0 + (real_T)
                break_count;
              exitg2 = true;
            } else {
              break_count++;
            }
          }

          if (Path_col[Forward_Static_Path_length_0 << 2] == 1.0) {
            Path_col[1 + (Forward_Static_Path_length_0 << 2)] = 1.0 + (real_T)
              case_0;
            exitg1 = true;
          } else {
            case_0++;
          }
        }
      }
    }
  }

  for (i = 0; i < 13; i++) {
    Cobs[i] = Path_col[i << 2];
    Cobs_0[i] = Path_col[i << 2];
  }

  ang_1 = std(Cobs_0);
  if (ang_1 != 0.0) {
    Length_1 = ang_1 * ang_1 * 2.0;
    J_minvalue_diff = 2.5066282746310002 * ang_1;
    for (Forward_Static_Path_length = 0; Forward_Static_Path_length < 13;
         Forward_Static_Path_length++) {
      i = 1 + Forward_Static_Path_length;
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 13;
           Path_RES_0_size_idx_1++) {
        Cc_0[Path_RES_0_size_idx_1] = (i - Path_RES_0_size_idx_1) - 1;
      }

      power_bvwt(Cc_0, rtb_forward_length_free);
      for (i = 0; i < 13; i++) {
        Cc_0[i] = -rtb_forward_length_free[i] / Length_1;
      }

      exp_n(Cc_0);
      for (i = 0; i < 13; i++) {
        Cobs_0[i] = Path_col[i << 2] * (Cc_0[i] / J_minvalue_diff);
      }

      Cobs[Forward_Static_Path_length] = sum(Cobs_0);
      if ((1 + Forward_Static_Path_length == 1) && (Path_col[0] == 1.0)) {
        Cobs[0] += std::exp(-1.0 / (ang_1 * ang_1 * 2.0)) / (2.5066282746310002 *
          ang_1);
      } else {
        if ((1 + Forward_Static_Path_length == 13) && (Path_col[48] == 1.0)) {
          Cobs[12] += std::exp(-1.0 / (ang_1 * ang_1 * 2.0)) /
            (2.5066282746310002 * ang_1);
        }
      }
    }

    ex = rtIsNaN(Cobs[0]);
    if (!ex) {
      Forward_Static_Path_length = 1;
    } else {
      Forward_Static_Path_length = 0;
      case_0 = 2;
      exitg1 = false;
      while ((!exitg1) && (case_0 < 14)) {
        if (!rtIsNaN(Cobs[case_0 - 1])) {
          Forward_Static_Path_length = case_0;
          exitg1 = true;
        } else {
          case_0++;
        }
      }
    }

    if (Forward_Static_Path_length == 0) {
      ang_1 = Cobs[0];
    } else {
      ang_1 = Cobs[Forward_Static_Path_length - 1];
      while (Forward_Static_Path_length + 1 < 14) {
        if (ang_1 < Cobs[Forward_Static_Path_length]) {
          ang_1 = Cobs[Forward_Static_Path_length];
        }

        Forward_Static_Path_length++;
      }
    }

    if (ang_1 != 1.0) {
      if (!ex) {
        Forward_Static_Path_length = 1;
      } else {
        Forward_Static_Path_length = 0;
        case_0 = 2;
        exitg1 = false;
        while ((!exitg1) && (case_0 < 14)) {
          if (!rtIsNaN(Cobs[case_0 - 1])) {
            Forward_Static_Path_length = case_0;
            exitg1 = true;
          } else {
            case_0++;
          }
        }
      }

      if (Forward_Static_Path_length == 0) {
        ang_1 = Cobs[0];
      } else {
        ang_1 = Cobs[Forward_Static_Path_length - 1];
        while (Forward_Static_Path_length + 1 < 14) {
          if (ang_1 < Cobs[Forward_Static_Path_length]) {
            ang_1 = Cobs[Forward_Static_Path_length];
          }

          Forward_Static_Path_length++;
        }
      }

      for (i = 0; i < 13; i++) {
        Cobs[i] /= ang_1;
      }
    }
  }

  for (i = 0; i < 13; i++) {
    Clane[i] = Path_col[(i << 2) + 3];
    Cobs_0[i] = Path_col[(i << 2) + 3];
  }

  ang_1 = std(Cobs_0);
  if (ang_1 != 0.0) {
    target_k = ang_1 * ang_1 * 2.0;
    J_minvalue_diff = 2.5066282746310002 * ang_1;
    for (Forward_Static_Path_length = 0; Forward_Static_Path_length < 13;
         Forward_Static_Path_length++) {
      i = 1 + Forward_Static_Path_length;
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 13;
           Path_RES_0_size_idx_1++) {
        Cc_0[Path_RES_0_size_idx_1] = (i - Path_RES_0_size_idx_1) - 1;
      }

      power_bvwt(Cc_0, rtb_forward_length_free);
      for (i = 0; i < 13; i++) {
        Cc_0[i] = -rtb_forward_length_free[i] / target_k;
      }

      exp_n(Cc_0);
      for (i = 0; i < 13; i++) {
        Cobs_0[i] = Path_col[(i << 2) + 3] * (Cc_0[i] / J_minvalue_diff);
      }

      Clane[Forward_Static_Path_length] = sum(Cobs_0);
      if ((1 + Forward_Static_Path_length == 1) && (Path_col[3] == 1.0)) {
        Clane[0] += std::exp(-1.0 / (ang_1 * ang_1 * 2.0)) / (2.5066282746310002
          * ang_1);
      } else {
        if ((1 + Forward_Static_Path_length == 13) && (Path_col[51] == 1.0)) {
          Clane[12] += std::exp(-1.0 / (ang_1 * ang_1 * 2.0)) /
            (2.5066282746310002 * ang_1);
        }
      }
    }

    ex = rtIsNaN(Clane[0]);
    if (!ex) {
      Forward_Static_Path_length = 1;
    } else {
      Forward_Static_Path_length = 0;
      case_0 = 2;
      exitg1 = false;
      while ((!exitg1) && (case_0 < 14)) {
        if (!rtIsNaN(Clane[case_0 - 1])) {
          Forward_Static_Path_length = case_0;
          exitg1 = true;
        } else {
          case_0++;
        }
      }
    }

    if (Forward_Static_Path_length == 0) {
      ang_1 = Clane[0];
    } else {
      ang_1 = Clane[Forward_Static_Path_length - 1];
      while (Forward_Static_Path_length + 1 < 14) {
        if (ang_1 < Clane[Forward_Static_Path_length]) {
          ang_1 = Clane[Forward_Static_Path_length];
        }

        Forward_Static_Path_length++;
      }
    }

    if (ang_1 != 1.0) {
      if (!ex) {
        Forward_Static_Path_length = 1;
      } else {
        Forward_Static_Path_length = 0;
        case_0 = 2;
        exitg1 = false;
        while ((!exitg1) && (case_0 < 14)) {
          if (!rtIsNaN(Clane[case_0 - 1])) {
            Forward_Static_Path_length = case_0;
            exitg1 = true;
          } else {
            case_0++;
          }
        }
      }

      if (Forward_Static_Path_length == 0) {
        ang_1 = Clane[0];
      } else {
        ang_1 = Clane[Forward_Static_Path_length - 1];
        while (Forward_Static_Path_length + 1 < 14) {
          if (ang_1 < Clane[Forward_Static_Path_length]) {
            ang_1 = Clane[Forward_Static_Path_length];
          }

          Forward_Static_Path_length++;
        }
      }

      for (i = 0; i < 13; i++) {
        Clane[i] /= ang_1;
      }
    }
  }

  for (i = 0; i < 11; i++) {
    X1[i] = rtDW.UnitDelay5_DSTATE[i] - rtb_TmpSignalConversionAtSFun_n[0];
  }

  power_b(X1, K1);
  for (i = 0; i < 11; i++) {
    X1[i] = rtDW.UnitDelay5_DSTATE[11 + i] - rtb_TmpSignalConversionAtSFun_n[1];
  }

  power_b(X1, X2);
  for (i = 0; i < 11; i++) {
    b_Path_dis_data[i] = K1[i] + X2[i];
  }

  sqrt_f(b_Path_dis_data);
  if (!rtIsNaN(b_Path_dis_data[0])) {
    Forward_Static_Path_length = 1;
  } else {
    Forward_Static_Path_length = 0;
    case_0 = 2;
    exitg1 = false;
    while ((!exitg1) && (case_0 < 12)) {
      if (!rtIsNaN(b_Path_dis_data[case_0 - 1])) {
        Forward_Static_Path_length = case_0;
        exitg1 = true;
      } else {
        case_0++;
      }
    }
  }

  if (Forward_Static_Path_length == 0) {
    Forward_Static_Path_length = 1;
  } else {
    ang_1 = b_Path_dis_data[Forward_Static_Path_length - 1];
    for (case_0 = Forward_Static_Path_length; case_0 + 1 < 12; case_0++) {
      if (ang_1 > b_Path_dis_data[case_0]) {
        ang_1 = b_Path_dis_data[case_0];
        Forward_Static_Path_length = case_0 + 1;
      }
    }
  }

  end_ind_0 = 12 - Forward_Static_Path_length;
  loop_ub = -Forward_Static_Path_length;
  for (i = 0; i <= loop_ub + 11; i++) {
    LastPath_overlap_data[i] = rtDW.UnitDelay5_DSTATE
      [(Forward_Static_Path_length + i) - 1];
  }

  loop_ub = -Forward_Static_Path_length;
  for (i = 0; i <= loop_ub + 11; i++) {
    LastPath_overlap_data[i + end_ind_0] = rtDW.UnitDelay5_DSTATE
      [(Forward_Static_Path_length + i) + 10];
  }

  for (case_0 = 0; case_0 < 13; case_0++) {
    for (i = 0; i < 11; i++) {
      b_Path_dis_data[i] = X_2[11 * case_0 + i] - rtDW.UnitDelay5_DSTATE[10];
    }

    power_b(b_Path_dis_data, X1);
    for (i = 0; i < 11; i++) {
      b_Path_dis_data[i] = Y[11 * case_0 + i] - rtDW.UnitDelay5_DSTATE[21];
    }

    power_b(b_Path_dis_data, K1);
    for (i = 0; i < 11; i++) {
      b_Path_dis_data[i] = X1[i] + K1[i];
    }

    sqrt_f(b_Path_dis_data);
    if (!rtIsNaN(b_Path_dis_data[0])) {
      Forward_Static_Path_length_0 = 0;
    } else {
      Forward_Static_Path_length_0 = -1;
      break_count = 2;
      exitg1 = false;
      while ((!exitg1) && (break_count < 12)) {
        if (!rtIsNaN(b_Path_dis_data[break_count - 1])) {
          Forward_Static_Path_length_0 = break_count - 1;
          exitg1 = true;
        } else {
          break_count++;
        }
      }
    }

    if (Forward_Static_Path_length_0 + 1 == 0) {
      Forward_Static_Path_length_0 = 0;
    } else {
      target_k = b_Path_dis_data[Forward_Static_Path_length_0];
      for (break_count = Forward_Static_Path_length_0 + 1; break_count + 1 < 12;
           break_count++) {
        if (target_k > b_Path_dis_data[break_count]) {
          target_k = b_Path_dis_data[break_count];
          Forward_Static_Path_length_0 = break_count;
        }
      }
    }

    Path_overlap_size[0] = Forward_Static_Path_length_0 + 1;
    if (0 <= Forward_Static_Path_length_0) {
      memcpy(&Path_overlap_data[0], &X_2[case_0 * 11],
             (Forward_Static_Path_length_0 + 1) * sizeof(real_T));
    }

    for (i = 0; i <= Forward_Static_Path_length_0; i++) {
      Path_overlap_data[i + Path_overlap_size[0]] = Y[11 * case_0 + i];
    }

    if (12 - Forward_Static_Path_length >= Path_overlap_size[0]) {
      break_count = 13 - (Forward_Static_Path_length + Path_overlap_size[0]);
      if (break_count > 12 - Forward_Static_Path_length) {
        break_count = 1;
        Forward_Static_Path_length_0 = 0;
      } else {
        Forward_Static_Path_length_0 = 12 - Forward_Static_Path_length;
      }

      i = break_count - 1;
      Forward_Static_Path_length_0 -= i;
      LastPath_overlap_size_0[0] = Forward_Static_Path_length_0;
      LastPath_overlap_size_0[1] = 2;
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 <
           Forward_Static_Path_length_0; Path_RES_0_size_idx_1++) {
        LastPath_overlap_data_0[Path_RES_0_size_idx_1] = LastPath_overlap_data[i
          + Path_RES_0_size_idx_1] - Path_overlap_data[Path_RES_0_size_idx_1];
      }

      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 <
           Forward_Static_Path_length_0; Path_RES_0_size_idx_1++) {
        LastPath_overlap_data_0[Path_RES_0_size_idx_1 +
          Forward_Static_Path_length_0] = LastPath_overlap_data[(i +
          Path_RES_0_size_idx_1) + end_ind_0] -
          Path_overlap_data[Path_RES_0_size_idx_1 + Path_overlap_size[0]];
      }

      power_bvwts(LastPath_overlap_data_0, LastPath_overlap_size_0,
                  Path_overlap_data, Path_overlap_size);
      Path_overlap_size_1[0] = 2;
      Path_overlap_size_1[1] = Path_overlap_size[0];
      loop_ub = Path_overlap_size[0];
      for (i = 0; i < loop_ub; i++) {
        LastPath_overlap_data_0[i << 1] = Path_overlap_data[i];
        LastPath_overlap_data_0[1 + (i << 1)] = Path_overlap_data[i +
          Path_overlap_size[0]];
      }

      sum_p(LastPath_overlap_data_0, Path_overlap_size_1, b_Path_dis_data,
            b_Path_dis_size);
      sqrt_fh(b_Path_dis_data, b_Path_dis_size);
      loop_ub = b_Path_dis_size[1];
      for (i = 0; i < loop_ub; i++) {
        K_11[i] = b_Path_dis_data[b_Path_dis_size[0] * i];
      }

      i = b_Path_dis_size[1];
      c = b_Path_dis_size[1];
      if (0 <= i - 1) {
        memcpy(&Path_dis_data[0], &K_11[0], i * sizeof(real_T));
      }
    } else {
      Forward_Static_Path_length_0 = 12 - Forward_Static_Path_length;
      LastPath_overlap_size[0] = Forward_Static_Path_length_0;
      LastPath_overlap_size[1] = 2;
      for (i = 0; i < Forward_Static_Path_length_0; i++) {
        LastPath_overlap_data_0[i] = LastPath_overlap_data[i] -
          Path_overlap_data[i];
      }

      for (i = 0; i < Forward_Static_Path_length_0; i++) {
        LastPath_overlap_data_0[i + Forward_Static_Path_length_0] =
          LastPath_overlap_data[i + end_ind_0] - Path_overlap_data[i +
          Path_overlap_size[0]];
      }

      power_bvwts(LastPath_overlap_data_0, LastPath_overlap_size,
                  Path_overlap_data, Path_overlap_size);
      Path_overlap_size_0[0] = 2;
      Path_overlap_size_0[1] = Path_overlap_size[0];
      loop_ub = Path_overlap_size[0];
      for (i = 0; i < loop_ub; i++) {
        LastPath_overlap_data_0[i << 1] = Path_overlap_data[i];
        LastPath_overlap_data_0[1 + (i << 1)] = Path_overlap_data[i +
          Path_overlap_size[0]];
      }

      sum_p(LastPath_overlap_data_0, Path_overlap_size_0, b_Path_dis_data,
            b_Path_dis_size);
      sqrt_fh(b_Path_dis_data, b_Path_dis_size);
      loop_ub = b_Path_dis_size[1];
      for (i = 0; i < loop_ub; i++) {
        b_Path_dis_data_0[i] = b_Path_dis_data[b_Path_dis_size[0] * i];
      }

      i = b_Path_dis_size[1];
      c = b_Path_dis_size[1];
      if (0 <= i - 1) {
        memcpy(&Path_dis_data[0], &b_Path_dis_data_0[0], i * sizeof(real_T));
      }
    }

    if (c > 1) {
      i = c;
    } else {
      i = 1;
    }

    if (mod((real_T)i) == 0.0) {
      if (c > 1) {
        Forward_Static_Path_length_0 = c - 1;
      } else {
        Forward_Static_Path_length_0 = 0;
      }

      b_Path_dis_size[1] = Forward_Static_Path_length_0;
      loop_ub = Forward_Static_Path_length_0 - 1;
      for (i = 0; i <= loop_ub; i++) {
        b_Path_dis_data[i] = 4.0;
      }
    } else {
      if (c > 1) {
        Forward_Static_Path_length_0 = c;
      } else {
        Forward_Static_Path_length_0 = 1;
      }

      b_Path_dis_size[1] = Forward_Static_Path_length_0;
      loop_ub = Forward_Static_Path_length_0 - 1;
      for (i = 0; i <= loop_ub; i++) {
        b_Path_dis_data[i] = 4.0;
      }
    }

    b_Path_dis_data[0] = 1.0;
    b_Path_dis_data[b_Path_dis_size[1] - 1] = 1.0;
    if (3 > b_Path_dis_size[1] - 2) {
      Static_PathCycle = 1;
      break_count = 1;
      Forward_Static_Path_length_0 = 0;
    } else {
      Static_PathCycle = 3;
      break_count = 2;
      Forward_Static_Path_length_0 = b_Path_dis_size[1] - 2;
    }

    Forward_Static_Path_length_0 = div_nde_s32_floor((int8_T)
      Forward_Static_Path_length_0 - Static_PathCycle, break_count);
    for (i = 0; i <= Forward_Static_Path_length_0; i++) {
      p_data[i] = (int8_T)((int8_T)((int8_T)(break_count * (int8_T)i) +
        Static_PathCycle) - 1);
    }

    for (i = 0; i <= Forward_Static_Path_length_0; i++) {
      b_Path_dis_data[p_data[i]] = 2.0;
    }

    offset_5 = 0.0;
    for (i = 0; i < b_Path_dis_size[1]; i++) {
      offset_5 += b_Path_dis_data[i] * Path_dis_data[i];
    }

    if (!(c > 1)) {
      c = 1;
    }

    Cc_0[case_0] = rtb_J_out_a[case_0] / 11.0 * offset_5 / 3.0 /
      (rtb_J_out_a[case_0] * (real_T)c / 11.0);
  }

  for (i = 0; i < 13; i++) {
    rtb_U_c_n[i] = 1.0;
    rtb_safety_level_all_b[i] = 0.0;
    rtb_forward_length_free_f[i] = x_target;
  }

  if ((rtU.Freespace_mode == 1.0) || (rtU.Freespace_mode == 2.0)) {
    for (Forward_Static_Path_length = 0; Forward_Static_Path_length < 13;
         Forward_Static_Path_length++) {
      FreespaceDetectCollision(rtU.Freespace, &rtb_XP_i[6 *
        Forward_Static_Path_length], &rtb_YP_p[6 * Forward_Static_Path_length],
        rtb_TmpSignalConversionAtSFun_n, x_target, rtU.safe_range,
        rtConstP.pooled5, rtConstP.pooled4,
        &rtb_U_c_n[Forward_Static_Path_length],
        &rtb_safety_level_all_b[Forward_Static_Path_length],
        &rtb_forward_length_free_f[Forward_Static_Path_length]);
    }
  }

  abs_n(K, XY_difflen);
  for (Forward_Static_Path_length = 0; Forward_Static_Path_length < 13;
       Forward_Static_Path_length++) {
    rtb_forward_length_free[Forward_Static_Path_length] = XY_difflen[11 *
      Forward_Static_Path_length];
    for (Forward_Static_Path_length_0 = 0; Forward_Static_Path_length_0 < 10;
         Forward_Static_Path_length_0++) {
      target_k = rtb_forward_length_free[Forward_Static_Path_length];
      i = (11 * Forward_Static_Path_length + Forward_Static_Path_length_0) + 1;
      if ((!rtIsNaN(XY_difflen[i])) && (rtIsNaN
           (rtb_forward_length_free[Forward_Static_Path_length]) ||
           (rtb_forward_length_free[Forward_Static_Path_length] < XY_difflen[i])))
      {
        target_k = XY_difflen[i];
      }

      rtb_forward_length_free[Forward_Static_Path_length] = target_k;
    }
  }

  abs_n(K, XY_difflen);
  for (Forward_Static_Path_length = 0; Forward_Static_Path_length < 13;
       Forward_Static_Path_length++) {
    rtb_forward_length_free_2[Forward_Static_Path_length] = XY_difflen[11 *
      Forward_Static_Path_length];
    for (Static_PathCycle = 0; Static_PathCycle < 10; Static_PathCycle++) {
      x_endpoint1 = rtb_forward_length_free_2[Forward_Static_Path_length];
      i = (11 * Forward_Static_Path_length + Static_PathCycle) + 1;
      if ((!rtIsNaN(XY_difflen[i])) && (rtIsNaN
           (rtb_forward_length_free_2[Forward_Static_Path_length]) ||
           (rtb_forward_length_free_2[Forward_Static_Path_length] < XY_difflen[i])))
      {
        x_endpoint1 = XY_difflen[i];
      }

      rtb_forward_length_free_2[Forward_Static_Path_length] = x_endpoint1;
    }

    rtb_forward_length_free_2[Forward_Static_Path_length] *= 10.0;
  }

  if (!rtIsNaN(rtb_forward_length_free_2[0])) {
    Forward_Static_Path_length = 1;
  } else {
    Forward_Static_Path_length = 0;
    case_0 = 2;
    exitg1 = false;
    while ((!exitg1) && (case_0 < 14)) {
      if (!rtIsNaN(rtb_forward_length_free_2[case_0 - 1])) {
        Forward_Static_Path_length = case_0;
        exitg1 = true;
      } else {
        case_0++;
      }
    }
  }

  if (Forward_Static_Path_length == 0) {
    ang_1 = rtb_forward_length_free_2[0];
  } else {
    ang_1 = rtb_forward_length_free_2[Forward_Static_Path_length - 1];
    while (Forward_Static_Path_length + 1 < 14) {
      if (ang_1 < rtb_forward_length_free_2[Forward_Static_Path_length]) {
        ang_1 = rtb_forward_length_free_2[Forward_Static_Path_length];
      }

      Forward_Static_Path_length++;
    }
  }

  abs_n(K_1, XY_difflen);
  for (Forward_Static_Path_length = 0; Forward_Static_Path_length < 13;
       Forward_Static_Path_length++) {
    rtb_forward_length_free_2[Forward_Static_Path_length] = XY_difflen[11 *
      Forward_Static_Path_length];
    for (Static_PathCycle = 0; Static_PathCycle < 10; Static_PathCycle++) {
      x_endpoint1 = rtb_forward_length_free_2[Forward_Static_Path_length];
      i = (11 * Forward_Static_Path_length + Static_PathCycle) + 1;
      if ((!rtIsNaN(XY_difflen[i])) && (rtIsNaN
           (rtb_forward_length_free_2[Forward_Static_Path_length]) ||
           (rtb_forward_length_free_2[Forward_Static_Path_length] < XY_difflen[i])))
      {
        x_endpoint1 = XY_difflen[i];
      }

      rtb_forward_length_free_2[Forward_Static_Path_length] = x_endpoint1;
    }
  }

  abs_n(K_1, XY_difflen);
  for (Forward_Static_Path_length = 0; Forward_Static_Path_length < 13;
       Forward_Static_Path_length++) {
    Cobs_0[Forward_Static_Path_length] = XY_difflen[11 *
      Forward_Static_Path_length];
    for (Static_PathCycle = 0; Static_PathCycle < 10; Static_PathCycle++) {
      x_endpoint1 = Cobs_0[Forward_Static_Path_length];
      if ((!rtIsNaN(XY_difflen[(11 * Forward_Static_Path_length +
             Static_PathCycle) + 1])) && (rtIsNaN
           (Cobs_0[Forward_Static_Path_length]) ||
           (Cobs_0[Forward_Static_Path_length] < XY_difflen[(11 *
             Forward_Static_Path_length + Static_PathCycle) + 1]))) {
        x_endpoint1 = XY_difflen[(11 * Forward_Static_Path_length +
          Static_PathCycle) + 1];
      }

      Cobs_0[Forward_Static_Path_length] = x_endpoint1;
    }

    Cobs_0[Forward_Static_Path_length] *= 10.0;
  }

  if (!rtIsNaN(Cobs_0[0])) {
    Forward_Static_Path_length = 1;
  } else {
    Forward_Static_Path_length = 0;
    case_0 = 2;
    exitg1 = false;
    while ((!exitg1) && (case_0 < 14)) {
      if (!rtIsNaN(Cobs_0[case_0 - 1])) {
        Forward_Static_Path_length = case_0;
        exitg1 = true;
      } else {
        case_0++;
      }
    }
  }

  if (Forward_Static_Path_length == 0) {
    Length_1 = Cobs_0[0];
  } else {
    Length_1 = Cobs_0[Forward_Static_Path_length - 1];
    while (Forward_Static_Path_length + 1 < 14) {
      if (Length_1 < Cobs_0[Forward_Static_Path_length]) {
        Length_1 = Cobs_0[Forward_Static_Path_length];
      }

      Forward_Static_Path_length++;
    }
  }

  if (!rtIsNaN(offset[0])) {
    Forward_Static_Path_length_0 = 1;
  } else {
    Forward_Static_Path_length_0 = 0;
    break_count = 2;
    exitg1 = false;
    while ((!exitg1) && (break_count < 14)) {
      if (!rtIsNaN(offset[break_count - 1])) {
        Forward_Static_Path_length_0 = break_count;
        exitg1 = true;
      } else {
        break_count++;
      }
    }
  }

  if (Forward_Static_Path_length_0 == 0) {
    J_minvalue_diff = offset[0];
  } else {
    J_minvalue_diff = offset[Forward_Static_Path_length_0 - 1];
    while (Forward_Static_Path_length_0 + 1 < 14) {
      if (J_minvalue_diff < offset[Forward_Static_Path_length_0]) {
        J_minvalue_diff = offset[Forward_Static_Path_length_0];
      }

      Forward_Static_Path_length_0++;
    }
  }

  ex = rtIsNaN(Cc_0[0]);
  if (!ex) {
    Forward_Static_Path_length = 1;
  } else {
    Forward_Static_Path_length = 0;
    case_0 = 2;
    exitg1 = false;
    while ((!exitg1) && (case_0 < 14)) {
      if (!rtIsNaN(Cc_0[case_0 - 1])) {
        Forward_Static_Path_length = case_0;
        exitg1 = true;
      } else {
        case_0++;
      }
    }
  }

  if (Forward_Static_Path_length == 0) {
    target_k = Cc_0[0];
  } else {
    target_k = Cc_0[Forward_Static_Path_length - 1];
    while (Forward_Static_Path_length + 1 < 14) {
      if (target_k < Cc_0[Forward_Static_Path_length]) {
        target_k = Cc_0[Forward_Static_Path_length];
      }

      Forward_Static_Path_length++;
    }
  }

  if (!(target_k == 0.0)) {
    if (!ex) {
      Forward_Static_Path_length = 1;
    } else {
      Forward_Static_Path_length = 0;
      case_0 = 2;
      exitg1 = false;
      while ((!exitg1) && (case_0 < 14)) {
        if (!rtIsNaN(Cc_0[case_0 - 1])) {
          Forward_Static_Path_length = case_0;
          exitg1 = true;
        } else {
          case_0++;
        }
      }
    }

    if (Forward_Static_Path_length == 0) {
      target_k = Cc_0[0];
    } else {
      target_k = Cc_0[Forward_Static_Path_length - 1];
      while (Forward_Static_Path_length + 1 < 14) {
        if (target_k < Cc_0[Forward_Static_Path_length]) {
          target_k = Cc_0[Forward_Static_Path_length];
        }

        Forward_Static_Path_length++;
      }
    }

    for (i = 0; i < 13; i++) {
      Cc_0[i] /= target_k;
    }
  }

  for (i = 0; i < 13; i++) {
    rtb_J_out_a[i] = (((((rtb_forward_length_free[i] * 10.0 / ang_1 * rtU.W_1[1]
                          + rtb_J_out_a[i] / x_target * rtU.W_1[0]) +
                         rtb_forward_length_free_2[i] * 10.0 / Length_1 *
                         rtU.W_1[2]) + offset[i] / J_minvalue_diff * count) +
                       rtU.W_1[3] * Cobs[i]) + rtU.W_1[4] * Cc_0[i]) + rtU.W_1[5]
      * Clane[i];
  }

  // MATLAB Function: '<S1>/EndPointDecision1' incorporates:
  //   Inport: '<Root>/forward_length_2'
  //   MATLAB Function: '<S1>/EndPointDecision'

  xy_ends_POS_size_idx_0 = 20000;
  Path_RES_0_size_idx_1 = 2;
  memset(&rtDW.Path_RES_0_data[0], 0, 40000U * sizeof(real_T));
  memset(&rtDW.Path_RES_0_1[0], 0, 40000U * sizeof(real_T));
  count = 0.0;
  count_1 = 0.0;
  break_count = 0;
  target_k = std::floor(rtU.forward_length_2 / 0.1);
  offset_5 = std::sqrt(ang_1_tmp * ang_1_tmp + J_minvalue_diff_tmp *
                       J_minvalue_diff_tmp);
  ang_1 = rt_atan2d_snf(rtDW.Forward_Static_Path_y_gb[1] -
                        rtDW.Forward_Static_Path_y_gb[0],
                        rtDW.Forward_Static_Path_x_p[1] -
                        rtDW.Forward_Static_Path_x_p[0]);
  if (offset_5 > 0.1) {
    Length_1 = rt_roundd_snf(offset_5 / 0.1);
    for (case_0 = 0; case_0 < (int32_T)Length_1; case_0++) {
      x_endpoint1 = ((1.0 + (real_T)case_0) - 1.0) * 0.1;
      rtDW.Path_RES_0_1[case_0] = x_endpoint1 * std::cos(ang_1) +
        rtDW.Forward_Static_Path_x_p[0];
      rtDW.Path_RES_0_1[20000 + case_0] = x_endpoint1 * std::sin(ang_1) +
        rtDW.Forward_Static_Path_y_gb[0];
      count_1 = 1.0 + (real_T)case_0;
    }
  } else {
    rtDW.Path_RES_0_1[0] = rtDW.Forward_Static_Path_x_p[0];
    rtDW.Path_RES_0_1[20000] = rtDW.Forward_Static_Path_y_gb[0];
    count_1 = 1.0;
  }

  if (1.0 > count_1) {
    c = 0;
  } else {
    c = (int32_T)count_1;
  }

  Path_RES_1_size_idx_0 = c;
  if (0 <= c - 1) {
    memcpy(&rtDW.Path_RES_1_data[0], &rtDW.Path_RES_0_1[0], c * sizeof(real_T));
  }

  for (i = 0; i < c; i++) {
    rtDW.Path_RES_1_data[i + c] = rtDW.Path_RES_0_1[i + 20000];
  }

  for (i = 0; i < c; i++) {
    rtDW.tmp_data_c[i] = End_x - rtDW.Path_RES_1_data[i];
  }

  power_j(rtDW.tmp_data_c, &c, rtDW.tmp_data, &loop_ub);
  for (i = 0; i < c; i++) {
    rtDW.tmp_data_k[i] = End_y - rtDW.Path_RES_1_data[i + c];
  }

  power_j(rtDW.tmp_data_k, &c, rtDW.tmp_data_c, &Static_PathCycle);
  for (i = 0; i < loop_ub; i++) {
    rtDW.ob_distance_data[i] = rtDW.tmp_data[i] + rtDW.tmp_data_c[i];
  }

  if (loop_ub <= 2) {
    if (loop_ub == 1) {
      Forward_Static_Path_length = 0;
    } else if (rtDW.ob_distance_data[0] > rtDW.ob_distance_data[1]) {
      Forward_Static_Path_length = 1;
    } else if (rtIsNaN(rtDW.ob_distance_data[0])) {
      if (!rtIsNaN(rtDW.ob_distance_data[1])) {
        i = 2;
      } else {
        i = 1;
      }

      Forward_Static_Path_length = i - 1;
    } else {
      Forward_Static_Path_length = 0;
    }
  } else {
    if (!rtIsNaN(rtDW.ob_distance_data[0])) {
      Forward_Static_Path_length = 0;
    } else {
      Forward_Static_Path_length = -1;
      case_0 = 2;
      exitg1 = false;
      while ((!exitg1) && (case_0 <= loop_ub)) {
        if (!rtIsNaN(rtDW.ob_distance_data[case_0 - 1])) {
          Forward_Static_Path_length = case_0 - 1;
          exitg1 = true;
        } else {
          case_0++;
        }
      }
    }

    if (Forward_Static_Path_length + 1 == 0) {
      Forward_Static_Path_length = 0;
    } else {
      ang_1 = rtDW.ob_distance_data[Forward_Static_Path_length];
      for (Forward_Static_Path_length_0 = Forward_Static_Path_length + 1;
           Forward_Static_Path_length_0 < loop_ub; Forward_Static_Path_length_0
           ++) {
        if (ang_1 > rtDW.ob_distance_data[Forward_Static_Path_length_0]) {
          ang_1 = rtDW.ob_distance_data[Forward_Static_Path_length_0];
          Forward_Static_Path_length = Forward_Static_Path_length_0;
        }
      }
    }
  }

  ang_1 = count_1 - (real_T)(Forward_Static_Path_length + 1);
  if (rtDW.SFunction_DIMS2_i[0] - 2 >= 1) {
    for (Forward_Static_Path_length_0 = 1; Forward_Static_Path_length_0 - 1 <=
         rtDW.SFunction_DIMS2_i[0] - 3; Forward_Static_Path_length_0++) {
      if (break_count == 0) {
        J_minvalue_diff =
          rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0 + 1] -
          rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0];
        Length_1 = rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0 +
          1] - rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0];
        J_minvalue_diff = std::sqrt(J_minvalue_diff * J_minvalue_diff + Length_1
          * Length_1);
        Length_1 = rt_atan2d_snf
          (rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0 + 1] -
           rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0],
           rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0 + 1] -
           rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0]);
        if (J_minvalue_diff >= 0.1) {
          J_minvalue_diff = rt_roundd_snf(J_minvalue_diff / 0.1);
          for (Static_PathCycle = 0; Static_PathCycle < (int32_T)J_minvalue_diff;
               Static_PathCycle++) {
            x_endpoint1 = ((1.0 + (real_T)Static_PathCycle) - 1.0) * 0.1;
            i = (int32_T)((1.0 + (real_T)Static_PathCycle) + count);
            rtDW.Path_RES_0_data[i - 1] = x_endpoint1 * std::cos(Length_1) +
              rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0];
            rtDW.Path_RES_0_data[i + 19999] = x_endpoint1 * std::sin(Length_1) +
              rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0];
          }

          count += J_minvalue_diff;
        } else {
          rtDW.Path_RES_0_data[(int32_T)(1.0 + count) - 1] =
            rtDW.Forward_Static_Path_x_p[Forward_Static_Path_length_0];
          rtDW.Path_RES_0_data[(int32_T)(1.0 + count) + 19999] =
            rtDW.Forward_Static_Path_y_gb[Forward_Static_Path_length_0];
          count++;
        }

        if (count > target_k - ang_1) {
          break_count = 1;
        }
      }
    }
  } else {
    xy_ends_POS_size_idx_0 = 0;
    Path_RES_0_size_idx_1 = 0;
  }

  Length_1 = (real_T)(Forward_Static_Path_length + 1) + target_k;
  if ((xy_ends_POS_size_idx_0 == 0) || (Path_RES_0_size_idx_1 == 0)) {
    if (Length_1 <= c) {
      if (Forward_Static_Path_length + 1 > Length_1) {
        Forward_Static_Path_length = 0;
      }

      i = Forward_Static_Path_length + (int32_T)target_k;
      x_endpoint1 = rtDW.Path_RES_1_data[i - 1];
      y_endpoint1 = rtDW.Path_RES_1_data[(i + c) - 1];
      count = target_k * 0.1;
    } else {
      if (Forward_Static_Path_length + 1 > c) {
        Forward_Static_Path_length = 0;
        Forward_Static_Path_length_0 = 0;
      } else {
        Forward_Static_Path_length_0 = c;
      }

      break_count = Forward_Static_Path_length_0 - Forward_Static_Path_length;
      i = break_count + Forward_Static_Path_length;
      x_endpoint1 = rtDW.Path_RES_1_data[i - 1];
      y_endpoint1 = rtDW.Path_RES_1_data[(i + c) - 1];
      if (break_count == 0) {
        break_count = 0;
      } else {
        if (!(break_count > 2)) {
          break_count = 2;
        }
      }

      count = (real_T)break_count * 0.1;
    }
  } else {
    if (Forward_Static_Path_length + 1 > c) {
      Forward_Static_Path_length = 0;
      break_count = 0;
    } else {
      break_count = c;
    }

    if (1.0 > count) {
      Static_PathCycle = 0;
    } else {
      Static_PathCycle = (int32_T)count;
    }

    loop_ub = break_count - Forward_Static_Path_length;
    if (!(loop_ub == 0)) {
      case_0 = 2;
      Forward_Static_Path_length_0 = loop_ub;
    } else {
      if (!(Static_PathCycle == 0)) {
        case_0 = Path_RES_0_size_idx_1;
      } else {
        case_0 = 2;
      }

      Forward_Static_Path_length_0 = 0;
    }

    if (!(Static_PathCycle == 0)) {
      c = Static_PathCycle;
    } else {
      c = 0;
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Path_RES_0_1[i] = rtDW.Path_RES_1_data[Forward_Static_Path_length + i];
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Path_RES_0_1[i + loop_ub] = rtDW.Path_RES_1_data
        [(Forward_Static_Path_length + i) + Path_RES_1_size_idx_0];
    }

    loop_ub = Path_RES_0_size_idx_1 - 1;
    for (i = 0; i <= loop_ub; i++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < Static_PathCycle;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_0_data_p[Path_RES_0_size_idx_1 + Static_PathCycle * i] =
          rtDW.Path_RES_0_data[xy_ends_POS_size_idx_0 * i +
          Path_RES_0_size_idx_1];
      }
    }

    Forward_Static_Path_length = Forward_Static_Path_length_0 + c;
    for (i = 0; i < case_0; i++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 <
           Forward_Static_Path_length_0; Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_data[Path_RES_0_size_idx_1 + Forward_Static_Path_length *
          i] = rtDW.Path_RES_0_1[Forward_Static_Path_length_0 * i +
          Path_RES_0_size_idx_1];
      }
    }

    for (i = 0; i < case_0; i++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < c;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_data[(Path_RES_0_size_idx_1 + Forward_Static_Path_length_0)
          + Forward_Static_Path_length * i] = rtDW.Path_RES_0_data_p[c * i +
          Path_RES_0_size_idx_1];
      }
    }

    if (target_k - ang_1 <= count) {
      x_endpoint1 = rtDW.Path_RES_data[(int32_T)target_k - 1];
      y_endpoint1 = rtDW.Path_RES_data[((int32_T)target_k +
        Forward_Static_Path_length) - 1];
      count = target_k * 0.1;
    } else {
      count += ang_1;
      i = (int32_T)count;
      x_endpoint1 = rtDW.Path_RES_data[i - 1];
      y_endpoint1 = rtDW.Path_RES_data[(i + Forward_Static_Path_length) - 1];
      count *= 0.1;
    }
  }

  // SignalConversion: '<S5>/TmpSignal ConversionAt SFunction Inport5' incorporates:
  //   Gain: '<S1>/Gain4'
  //   Inport: '<Root>/X_UKF_SLAM_i1'
  //   MATLAB Function: '<S1>/DynamicPathPlanning1'
  //   MATLAB Function: '<S1>/MATLAB Function'

  rtb_TmpSignalConversionAtSFun_n[0] = rtU.X_UKF_SLAM_i1[0];
  rtb_TmpSignalConversionAtSFun_n[1] = rtU.X_UKF_SLAM_i1[1];
  rtb_TmpSignalConversionAtSFun_n[2] = 0.017453292519943295 * vehicle_heading;

  // MATLAB Function: '<S1>/DynamicPathPlanning1' incorporates:
  //   Constant: '<S1>/Constant14'
  //   Constant: '<S1>/Constant4'
  //   Inport: '<Root>/BB_num'
  //   Inport: '<Root>/Freespace'
  //   Inport: '<Root>/Freespace_mode'
  //   Inport: '<Root>/W_2'
  //   Inport: '<Root>/safe_range'
  //   MATLAB Function: '<S1>/DynamicPathPlanning'
  //   MATLAB Function: '<S1>/EndPointDecision'
  //   MATLAB Function: '<S1>/EndPointDecision1'
  //   UnitDelay: '<S1>/Unit Delay6'

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    varargin_1_data[i] = (rtDW.Forward_Static_Path_id_g[rtDW.SFunction_DIMS4_l[0]
                          - 1] == rtDW.Static_Path_0[i]);
  }

  Forward_Static_Path_length = rtDW.SFunction_DIMS2_g[0] - 1;
  case_0 = 0;
  for (break_count = 0; break_count <= Forward_Static_Path_length; break_count++)
  {
    if (varargin_1_data[break_count]) {
      case_0++;
    }
  }

  break_count = case_0;
  Forward_Static_Path_length_0 = 0;
  for (case_0 = 0; case_0 <= Forward_Static_Path_length; case_0++) {
    if (varargin_1_data[case_0]) {
      t_data_0[Forward_Static_Path_length_0] = case_0 + 1;
      Forward_Static_Path_length_0++;
    }
  }

  for (i = 0; i < break_count; i++) {
    rtDW.Forward_Static_Path_id_0_data[i] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS2_g[0] * 7 + t_data_0[i]) - 1] * 3.1415926535897931;
  }

  for (i = 0; i < break_count; i++) {
    rtDW.end_heading_0_data[i] = rtDW.Forward_Static_Path_id_0_data[i] / 180.0;
  }

  x_endpoint2 = rtDW.Forward_Static_Path_id_0_data[0] / 180.0;
  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    varargin_1_data[i] = (rtDW.Forward_Static_Path_id_g[rtDW.SFunction_DIMS4_l[0]
                          - 1] == rtDW.Static_Path_0[i]);
  }

  Forward_Static_Path_length_0 = 0;
  for (case_0 = 0; case_0 < rtDW.SFunction_DIMS2_g[0]; case_0++) {
    if (varargin_1_data[case_0]) {
      u_data_0[Forward_Static_Path_length_0] = case_0 + 1;
      Forward_Static_Path_length_0++;
    }
  }

  loop_ub = rtDW.SFunction_DIMS2_g[0];
  for (i = 0; i < loop_ub; i++) {
    varargin_1_data[i] = (rtDW.Forward_Static_Path_id_g[rtDW.SFunction_DIMS4_l[0]
                          - 1] == rtDW.Static_Path_0[i]);
  }

  case_0 = 0;
  for (Forward_Static_Path_length_0 = 0; Forward_Static_Path_length_0 <
       rtDW.SFunction_DIMS2_g[0]; Forward_Static_Path_length_0++) {
    if (varargin_1_data[Forward_Static_Path_length_0]) {
      v_data_0[case_0] = Forward_Static_Path_length_0 + 1;
      case_0++;
    }
  }

  target_k = rtDW.Static_Path_0[(rtDW.SFunction_DIMS2_g[0] * 10 + v_data_0[0]) -
    1] / 4.0;
  ang_1 = target_k * 2.0;
  J_minvalue_diff = target_k * 3.0;
  Length_1 = target_k * 4.0;
  offset_5 = target_k * 5.0;
  count_1 = target_k * 6.0;
  G2splines_k(xy_end_point_idx_0, xy_end_point_idx_1, rtDW.seg_id_data[0],
              rtDW.Static_Path_0[(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) -
              1], x_endpoint1 + count_1 * std::cos(x_endpoint2 +
    1.5707963267948966), y_endpoint1 + count_1 * std::sin(x_endpoint2 +
    1.5707963267948966), rtDW.end_heading_0_data[0], rtDW.Static_Path_0
              [(u_data_0[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], count, X1,
              b_Path_dis_data, XP1, YP1, K1, K_11, &Cobs[0]);
  G2splines_k(xy_end_point_idx_2, y, rtDW.seg_id_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_endpoint1 +
              offset_5 * std::cos(x_endpoint2 + 1.5707963267948966), y_endpoint1
              + offset_5 * std::sin(x_endpoint2 + 1.5707963267948966),
              rtDW.end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0] +
    rtDW.SFunction_DIMS2_g[0] * 13) - 1], count, X2, Y2, XP2, YP2, K1, K_11,
              &Cobs[1]);
  G2splines_k(x_endpoint3, y_endpoint3, rtDW.seg_id_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_endpoint1 +
              Length_1 * std::cos(x_endpoint2 + 1.5707963267948966), y_endpoint1
              + Length_1 * std::sin(x_endpoint2 + 1.5707963267948966),
              rtDW.end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0] +
    rtDW.SFunction_DIMS2_g[0] * 13) - 1], count, X3, Y3, XP3, YP3, K1, K_11,
              &Cobs[2]);
  G2splines_k(x_endpoint4, y_endpoint4, rtDW.seg_id_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_endpoint1 +
              J_minvalue_diff * std::cos(x_endpoint2 + 1.5707963267948966),
              y_endpoint1 + J_minvalue_diff * std::sin(x_endpoint2 +
    1.5707963267948966), rtDW.end_heading_0_data[0], rtDW.Static_Path_0
              [(u_data_0[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], count, X4,
              Y4, XP4, YP4, K1, K_11, &Cobs[3]);
  G2splines_k(x_endpoint5, y_endpoint5, rtDW.seg_id_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_endpoint1 +
              ang_1 * std::cos(x_endpoint2 + 1.5707963267948966), y_endpoint1 +
              ang_1 * std::sin(x_endpoint2 + 1.5707963267948966),
              rtDW.end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0] +
    rtDW.SFunction_DIMS2_g[0] * 13) - 1], count, X5, Y5, XP5, YP5, K1, K_11,
              &Cobs[4]);
  G2splines_k(x_endpoint6, y_endpoint6, rtDW.seg_id_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_endpoint1 +
              target_k * std::cos(x_endpoint2 + 1.5707963267948966), y_endpoint1
              + target_k * std::sin(x_endpoint2 + 1.5707963267948966),
              rtDW.end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0] +
    rtDW.SFunction_DIMS2_g[0] * 13) - 1], count, X6, Y6, XP6, YP6, K1, K_11,
              &Cobs[5]);
  G2splines_k(End_x, End_y, rtDW.seg_id_data[0], rtDW.Static_Path_0[(u_data[0] +
    rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_endpoint1, y_endpoint1,
              rtDW.end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0] +
    rtDW.SFunction_DIMS2_g[0] * 13) - 1], count, X7, Y7, XP7, YP7, K1, K_11,
              &Cobs[6]);
  G2splines_k(x_endpoint8, y_endpoint8, rtDW.seg_id_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_endpoint1 +
              target_k * std::cos(x_endpoint2 - 1.5707963267948966), y_endpoint1
              + target_k * std::sin(x_endpoint2 - 1.5707963267948966),
              rtDW.end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0] +
    rtDW.SFunction_DIMS2_g[0] * 13) - 1], count, X8, Y8, XP8, YP8, K1, K_11,
              &Cobs[7]);
  G2splines_k(x_endpoint9, y_endpoint9, rtDW.seg_id_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], x_endpoint1 +
              ang_1 * std::cos(x_endpoint2 - 1.5707963267948966), y_endpoint1 +
              ang_1 * std::sin(x_endpoint2 - 1.5707963267948966),
              rtDW.end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0] +
    rtDW.SFunction_DIMS2_g[0] * 13) - 1], count, X9, Y9, XP9, YP9, K1, K_11,
              &Cobs[8]);
  G2splines_k(x_endpoint10, y_endpoint10, rtDW.seg_id_data[0],
              rtDW.Static_Path_0[(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) -
              1], x_endpoint1 + J_minvalue_diff * std::cos(x_endpoint2 -
    1.5707963267948966), y_endpoint1 + J_minvalue_diff * std::sin(x_endpoint2 -
    1.5707963267948966), rtDW.end_heading_0_data[0], rtDW.Static_Path_0
              [(u_data_0[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], count, X10,
              Y10, XP10, YP10, K1, K_11, &Cobs[9]);
  G2splines_k(x_endpoint11, y_endpoint11, rtDW.seg_id_data[0],
              rtDW.Static_Path_0[(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) -
              1], x_endpoint1 + Length_1 * std::cos(x_endpoint2 -
    1.5707963267948966), y_endpoint1 + Length_1 * std::sin(x_endpoint2 -
    1.5707963267948966), rtDW.end_heading_0_data[0], rtDW.Static_Path_0
              [(u_data_0[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], count, X11,
              Y11, XP11, YP11, K1, K_11, &Cobs[10]);
  G2splines_k(x_endpoint12, y_endpoint12, rtDW.seg_id_data[0],
              rtDW.Static_Path_0[(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) -
              1], x_endpoint1 + offset_5 * std::cos(x_endpoint2 -
    1.5707963267948966), y_endpoint1 + offset_5 * std::sin(x_endpoint2 -
    1.5707963267948966), rtDW.end_heading_0_data[0], rtDW.Static_Path_0
              [(u_data_0[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], count, X12,
              Y12, XP12, YP12, K1, K_11, &Cobs[11]);
  G2splines_k(x_endpoint13, xy_end_point_idx_25, rtDW.seg_id_data[0],
              rtDW.Static_Path_0[(u_data[0] + rtDW.SFunction_DIMS2_g[0] * 13) -
              1], x_endpoint1 + count_1 * std::cos(x_endpoint2 -
    1.5707963267948966), y_endpoint1 + count_1 * std::sin(x_endpoint2 -
    1.5707963267948966), rtDW.end_heading_0_data[0], rtDW.Static_Path_0
              [(u_data_0[0] + rtDW.SFunction_DIMS2_g[0] * 13) - 1], count, K1,
              K_11, XP13, YP13, K13, K_113, &Cobs[12]);
  for (i = 0; i < 11; i++) {
    X_2[i] = X1[i];
    X_2[i + 11] = X2[i];
    X_2[i + 22] = X3[i];
    X_2[i + 33] = X4[i];
    X_2[i + 44] = X5[i];
    X_2[i + 55] = X6[i];
    X_2[i + 66] = X7[i];
    X_2[i + 77] = X8[i];
    X_2[i + 88] = X9[i];
    X_2[i + 99] = X10[i];
    X_2[i + 110] = X11[i];
    X_2[i + 121] = X12[i];
    X_2[i + 132] = K1[i];
    Y[i] = b_Path_dis_data[i];
    Y[i + 11] = Y2[i];
    Y[i + 22] = Y3[i];
    Y[i + 33] = Y4[i];
    Y[i + 44] = Y5[i];
    Y[i + 55] = Y6[i];
    Y[i + 66] = Y7[i];
    Y[i + 77] = Y8[i];
    Y[i + 88] = Y9[i];
    Y[i + 99] = Y10[i];
    Y[i + 110] = Y11[i];
    Y[i + 121] = Y12[i];
    Y[i + 132] = K_11[i];
  }

  for (i = 0; i < 6; i++) {
    rtb_XP[i] = XP1[i];
    rtb_XP[i + 6] = XP2[i];
    rtb_XP[i + 12] = XP3[i];
    rtb_XP[i + 18] = XP4[i];
    rtb_XP[i + 24] = XP5[i];
    rtb_XP[i + 30] = XP6[i];
    rtb_XP[i + 36] = XP7[i];
    rtb_XP[i + 42] = XP8[i];
    rtb_XP[i + 48] = XP9[i];
    rtb_XP[i + 54] = XP10[i];
    rtb_XP[i + 60] = XP11[i];
    rtb_XP[i + 66] = XP12[i];
    rtb_XP[i + 72] = XP13[i];
    rtb_YP[i] = YP1[i];
    rtb_YP[i + 6] = YP2[i];
    rtb_YP[i + 12] = YP3[i];
    rtb_YP[i + 18] = YP4[i];
    rtb_YP[i + 24] = YP5[i];
    rtb_YP[i + 30] = YP6[i];
    rtb_YP[i + 36] = YP7[i];
    rtb_YP[i + 42] = YP8[i];
    rtb_YP[i + 48] = YP9[i];
    rtb_YP[i + 54] = YP10[i];
    rtb_YP[i + 60] = YP11[i];
    rtb_YP[i + 66] = YP12[i];
    rtb_YP[i + 72] = YP13[i];
  }

  memset(&Path_col[0], 0, 52U * sizeof(real_T));
  for (i = 0; i < 5; i++) {
    Path_col[3 + ((8 + i) << 2)] = 1.0;
  }

  Path_col[3] = 1.0;
  Path_col[51] = 1.0;
  if ((rtU.Freespace_mode == 0.0) || (rtU.Freespace_mode == 2.0)) {
    memcpy(&OBXY_EL[0], &rtb_V_boundingbox[0], 400U * sizeof(real_T));
    for (Static_PathCycle = 0; Static_PathCycle < (int32_T)rtU.BB_num;
         Static_PathCycle++) {
      offset_5 = (1.0 + (real_T)Static_PathCycle) * 2.0;
      i = (int32_T)(offset_5 + -1.0);
      Forward_Static_Path_length = i - 1;
      OBXY_EL[Forward_Static_Path_length] =
        ((rtb_V_boundingbox[Forward_Static_Path_length] - rtb_V_boundingbox[i +
          99]) * 0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
           Static_PathCycle) * 2.0 + -1.0) - 1]) + (rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0 + -1.0) - 1] -
        rtb_V_boundingbox[i + 299]) * 0.3;
      Forward_Static_Path_length = (int32_T)offset_5;
      Forward_Static_Path_length_0 = Forward_Static_Path_length - 1;
      OBXY_EL[Forward_Static_Path_length_0] =
        ((rtb_V_boundingbox[Forward_Static_Path_length_0] -
          rtb_V_boundingbox[Forward_Static_Path_length + 99]) * 0.15 +
         rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0) - 1])
        + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0) -
           1] - rtb_V_boundingbox[Forward_Static_Path_length + 299]) * 0.3;
      OBXY_EL[(int32_T)(offset_5 + -1.0) + 99] = ((rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0 + -1.0) + 99] -
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0 +
        -1.0) - 1]) * 0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
        Static_PathCycle) * 2.0 + -1.0) + 99]) + (rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0 + -1.0) + 99] -
        rtb_V_boundingbox[i + 199]) * 0.3;
      OBXY_EL[(int32_T)offset_5 + 99] = ((rtb_V_boundingbox[(int32_T)((1.0 +
        (real_T)Static_PathCycle) * 2.0) + 99] - rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0) - 1]) * 0.15 +
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0) + 99])
        + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0) +
           99] - rtb_V_boundingbox[Forward_Static_Path_length + 199]) * 0.3;
      OBXY_EL[(int32_T)(offset_5 + -1.0) + 199] = ((rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0 + -1.0) + 199] -
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0 +
        -1.0) + 299]) * 0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
        Static_PathCycle) * 2.0 + -1.0) + 199]) + (rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0 + -1.0) + 199] -
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0 +
        -1.0) + 99]) * 0.3;
      OBXY_EL[(int32_T)offset_5 + 199] = ((rtb_V_boundingbox[(int32_T)((1.0 +
        (real_T)Static_PathCycle) * 2.0) + 199] - rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0) + 299]) * 0.15 +
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0) +
        199]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) *
                  2.0) + 199] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
        Static_PathCycle) * 2.0) + 99]) * 0.3;
      OBXY_EL[(int32_T)(offset_5 + -1.0) + 299] = ((rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0 + -1.0) + 299] -
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0 +
        -1.0) + 199]) * 0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
        Static_PathCycle) * 2.0 + -1.0) + 299]) + (rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0 + -1.0) + 299] -
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0 +
        -1.0) - 1]) * 0.3;
      OBXY_EL[(int32_T)offset_5 + 299] = ((rtb_V_boundingbox[(int32_T)((1.0 +
        (real_T)Static_PathCycle) * 2.0) + 299] - rtb_V_boundingbox[(int32_T)
        ((1.0 + (real_T)Static_PathCycle) * 2.0) + 199]) * 0.15 +
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) * 2.0) +
        299]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)Static_PathCycle) *
                  2.0) + 299] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
        Static_PathCycle) * 2.0) - 1]) * 0.3;
    }

    for (i = 0; i < 13; i++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 10;
           Path_RES_0_size_idx_1++) {
        Forward_Static_Path_length = 11 * i + Path_RES_0_size_idx_1;
        target_k = X_2[Forward_Static_Path_length + 1] -
          X_2[Forward_Static_Path_length];
        X_diff[Path_RES_0_size_idx_1 + 11 * i] = target_k;
        X_diff_0[Path_RES_0_size_idx_1 + 10 * i] = target_k;
      }

      Forward_Static_Path_length_0 = 10 + 11 * i;
      X_diff[Forward_Static_Path_length_0] = X_diff_0[10 * i + 9];
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 10;
           Path_RES_0_size_idx_1++) {
        Forward_Static_Path_length = 11 * i + Path_RES_0_size_idx_1;
        target_k = Y[Forward_Static_Path_length + 1] -
          Y[Forward_Static_Path_length];
        Y_diff[Path_RES_0_size_idx_1 + 11 * i] = target_k;
        X_diff_0[Path_RES_0_size_idx_1 + 10 * i] = target_k;
      }

      Y_diff[Forward_Static_Path_length_0] = X_diff_0[10 * i + 9];
    }

    power_bv(X_diff, XY_difflen);
    power_bv(Y_diff, Path_vehFLY);
    for (i = 0; i < 143; i++) {
      Path_vehFLX[i] = XY_difflen[i] + Path_vehFLY[i];
    }

    power_bvw(Path_vehFLX, XY_difflen);
    for (i = 0; i < 143; i++) {
      target_k = X_diff[i] / XY_difflen[i];
      x_endpoint1 = Y_diff[i] / XY_difflen[i];
      y_endpoint1 = 1.1 * -x_endpoint1 + X_2[i];
      Path_vehFLX[i] = y_endpoint1 + 1.4000000000000001 * target_k;
      ang_1 = 1.1 * target_k + Y[i];
      Path_vehFLY[i] = ang_1 + 1.4000000000000001 * x_endpoint1;
      J_minvalue_diff = X_2[i] - 1.1 * -x_endpoint1;
      Path_vehFRX[i] = J_minvalue_diff + 1.4000000000000001 * target_k;
      Length_1 = Y[i] - 1.1 * target_k;
      Path_vehFRY[i] = Length_1 + 1.4000000000000001 * x_endpoint1;
      Path_vehRLX[i] = y_endpoint1 - 5.6000000000000005 * target_k;
      Path_vehRLY[i] = ang_1 - 5.6000000000000005 * x_endpoint1;
      Path_vehRRX[i] = J_minvalue_diff - 5.6000000000000005 * target_k;
      Path_vehRRY[i] = Length_1 - 5.6000000000000005 * x_endpoint1;
      X_diff[i] = target_k;
      XY_difflen[i] = -x_endpoint1;
      Y_diff[i] = x_endpoint1;
    }

    for (Forward_Static_Path_length_0 = 0; Forward_Static_Path_length_0 < 13;
         Forward_Static_Path_length_0++) {
      Path_col[Forward_Static_Path_length_0 << 2] = 0.0;
      if (!(Path_col[(Forward_Static_Path_length_0 << 2) + 3] == 1.0)) {
        case_0 = 0;
        exitg1 = false;
        while ((!exitg1) && (case_0 < 11)) {
          loop_ub = 11 * Forward_Static_Path_length_0 + case_0;
          OBXY_m[0] = Path_vehFLX[loop_ub];
          OBXY_m[2] = Path_vehFRX[loop_ub];
          OBXY_m[4] = Path_vehRLX[loop_ub];
          OBXY_m[6] = Path_vehRRX[loop_ub];
          OBXY_m[1] = Path_vehFLY[loop_ub];
          OBXY_m[3] = Path_vehFRY[loop_ub];
          OBXY_m[5] = Path_vehRLY[loop_ub];
          OBXY_m[7] = Path_vehRRY[loop_ub];
          break_count = 0;
          exitg2 = false;
          while ((!exitg2) && (break_count <= (int32_T)rtU.BB_num - 1)) {
            y_endpoint1 = (1.0 + (real_T)break_count) * 2.0;
            i = (int32_T)(y_endpoint1 + -1.0);
            ang_1_tmp = OBXY_EL[i + 99] - OBXY_EL[i - 1];
            target_k = std::sqrt(ang_1_tmp * ang_1_tmp + ang_1_tmp * ang_1_tmp);
            Forward_Static_Path_length = (int32_T)y_endpoint1;
            count_1 = -(OBXY_EL[Forward_Static_Path_length + 99] -
                        OBXY_EL[Forward_Static_Path_length - 1]) / target_k;
            target_k = ang_1_tmp / target_k;
            J_minvalue_diff_tmp = OBXY_EL[Forward_Static_Path_length + 199] -
              OBXY_EL[(int32_T)((1.0 + (real_T)break_count) * 2.0) + 99];
            x_endpoint1 = OBXY_EL[i + 199] - OBXY_EL[(int32_T)((1.0 + (real_T)
              break_count) * 2.0 + -1.0) + 99];
            J_minvalue_diff = std::sqrt(J_minvalue_diff_tmp *
              J_minvalue_diff_tmp + x_endpoint1 * x_endpoint1);
            Length_1 = -J_minvalue_diff_tmp / J_minvalue_diff;
            x_endpoint1 /= J_minvalue_diff;
            rtb_TmpSignalConversionAtSFun_1[0] = count_1;
            rtb_TmpSignalConversionAtSFun_1[1] = Length_1;
            rtb_TmpSignalConversionAtSFun_1[4] = target_k;
            rtb_TmpSignalConversionAtSFun_1[5] = x_endpoint1;
            rtb_TmpSignalConversionAtSFun_1[2] = X_diff[loop_ub];
            rtb_TmpSignalConversionAtSFun_1[6] = Y_diff[loop_ub];
            rtb_TmpSignalConversionAtSFun_1[3] = XY_difflen[loop_ub];
            rtb_TmpSignalConversionAtSFun_1[7] = X_diff[11 *
              Forward_Static_Path_length_0 + case_0];
            rtb_TmpSignalConversionAtSFun_2[0] = count_1;
            rtb_TmpSignalConversionAtSFun_2[1] = Length_1;
            rtb_TmpSignalConversionAtSFun_2[4] = target_k;
            rtb_TmpSignalConversionAtSFun_2[5] = x_endpoint1;
            rtb_TmpSignalConversionAtSFun_2[2] = X_diff[11 *
              Forward_Static_Path_length_0 + case_0];
            rtb_TmpSignalConversionAtSFun_2[6] = Y_diff[11 *
              Forward_Static_Path_length_0 + case_0];
            rtb_TmpSignalConversionAtSFun_2[3] = XY_difflen[11 *
              Forward_Static_Path_length_0 + case_0];
            rtb_TmpSignalConversionAtSFun_2[7] = X_diff[11 *
              Forward_Static_Path_length_0 + case_0];
            for (i = 0; i < 4; i++) {
              for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 4;
                   Path_RES_0_size_idx_1++) {
                proj_veh[i + (Path_RES_0_size_idx_1 << 2)] = 0.0;
                proj_veh[i + (Path_RES_0_size_idx_1 << 2)] +=
                  OBXY_m[Path_RES_0_size_idx_1 << 1] *
                  rtb_TmpSignalConversionAtSFun_1[i];
                proj_veh[i + (Path_RES_0_size_idx_1 << 2)] += OBXY_m
                  [(Path_RES_0_size_idx_1 << 1) + 1] *
                  rtb_TmpSignalConversionAtSFun_1[i + 4];
              }

              OBXY_EL_0[i << 1] = OBXY_EL[((int32_T)(y_endpoint1 + -1.0) + 100 *
                i) - 1];
              OBXY_EL_0[1 + (i << 1)] = OBXY_EL[(100 * i + (int32_T)y_endpoint1)
                - 1];
            }

            for (Static_PathCycle = 0; Static_PathCycle < 4; Static_PathCycle++)
            {
              for (i = 0; i < 4; i++) {
                proj_ob[Static_PathCycle + (i << 2)] = 0.0;
                proj_ob[Static_PathCycle + (i << 2)] += OBXY_EL_0[i << 1] *
                  rtb_TmpSignalConversionAtSFun_2[Static_PathCycle];
                proj_ob[Static_PathCycle + (i << 2)] += OBXY_EL_0[(i << 1) + 1] *
                  rtb_TmpSignalConversionAtSFun_2[Static_PathCycle + 4];
              }

              D[Static_PathCycle] = proj_veh[Static_PathCycle];
            }

            target_k = proj_veh[0];
            x_endpoint1 = proj_veh[1];
            y_endpoint1 = proj_veh[2];
            ang_1 = proj_veh[3];
            for (Forward_Static_Path_length = 0; Forward_Static_Path_length < 3;
                 Forward_Static_Path_length++) {
              if ((!rtIsNaN(proj_veh[(Forward_Static_Path_length + 1) << 2])) &&
                  (rtIsNaN(D[0]) || (D[0] > proj_veh[(Forward_Static_Path_length
                     + 1) << 2]))) {
                D[0] = proj_veh[(Forward_Static_Path_length + 1) << 2];
              }

              if ((!rtIsNaN(proj_veh[((Forward_Static_Path_length + 1) << 2) + 1]))
                  && (rtIsNaN(D[1]) || (D[1] > proj_veh
                    [((Forward_Static_Path_length + 1) << 2) + 1]))) {
                D[1] = proj_veh[((Forward_Static_Path_length + 1) << 2) + 1];
              }

              if ((!rtIsNaN(proj_veh[((Forward_Static_Path_length + 1) << 2) + 2]))
                  && (rtIsNaN(D[2]) || (D[2] > proj_veh
                    [((Forward_Static_Path_length + 1) << 2) + 2]))) {
                D[2] = proj_veh[((Forward_Static_Path_length + 1) << 2) + 2];
              }

              if ((!rtIsNaN(proj_veh[((Forward_Static_Path_length + 1) << 2) + 3]))
                  && (rtIsNaN(D[3]) || (D[3] > proj_veh
                    [((Forward_Static_Path_length + 1) << 2) + 3]))) {
                D[3] = proj_veh[((Forward_Static_Path_length + 1) << 2) + 3];
              }

              J_minvalue_diff = target_k;
              if ((!rtIsNaN(proj_veh[(Forward_Static_Path_length + 1) << 2])) &&
                  (rtIsNaN(target_k) || (target_k < proj_veh
                    [(Forward_Static_Path_length + 1) << 2]))) {
                J_minvalue_diff = proj_veh[(Forward_Static_Path_length + 1) << 2];
              }

              target_k = J_minvalue_diff;
              J_minvalue_diff = x_endpoint1;
              if ((!rtIsNaN(proj_veh[((Forward_Static_Path_length + 1) << 2) + 1]))
                  && (rtIsNaN(x_endpoint1) || (x_endpoint1 < proj_veh
                    [((Forward_Static_Path_length + 1) << 2) + 1]))) {
                J_minvalue_diff = proj_veh[((Forward_Static_Path_length + 1) <<
                  2) + 1];
              }

              x_endpoint1 = J_minvalue_diff;
              J_minvalue_diff = y_endpoint1;
              if ((!rtIsNaN(proj_veh[((Forward_Static_Path_length + 1) << 2) + 2]))
                  && (rtIsNaN(y_endpoint1) || (y_endpoint1 < proj_veh
                    [((Forward_Static_Path_length + 1) << 2) + 2]))) {
                J_minvalue_diff = proj_veh[((Forward_Static_Path_length + 1) <<
                  2) + 2];
              }

              y_endpoint1 = J_minvalue_diff;
              J_minvalue_diff = ang_1;
              if ((!rtIsNaN(proj_veh[((Forward_Static_Path_length + 1) << 2) + 3]))
                  && (rtIsNaN(ang_1) || (ang_1 < proj_veh
                    [((Forward_Static_Path_length + 1) << 2) + 3]))) {
                J_minvalue_diff = proj_veh[((Forward_Static_Path_length + 1) <<
                  2) + 3];
              }

              ang_1 = J_minvalue_diff;
            }

            minmax_veh[0] = D[0];
            minmax_veh[4] = target_k;
            minmax_veh[1] = D[1];
            minmax_veh[5] = x_endpoint1;
            minmax_veh[2] = D[2];
            minmax_veh[6] = y_endpoint1;
            minmax_veh[3] = D[3];
            minmax_veh[7] = ang_1;
            D[0] = proj_ob[0];
            D[1] = proj_ob[1];
            D[2] = proj_ob[2];
            D[3] = proj_ob[3];
            target_k = proj_ob[0];
            x_endpoint1 = proj_ob[1];
            y_endpoint1 = proj_ob[2];
            ang_1 = proj_ob[3];
            for (Forward_Static_Path_length = 0; Forward_Static_Path_length < 3;
                 Forward_Static_Path_length++) {
              if ((!rtIsNaN(proj_ob[(Forward_Static_Path_length + 1) << 2])) &&
                  (rtIsNaN(D[0]) || (D[0] > proj_ob[(Forward_Static_Path_length
                     + 1) << 2]))) {
                D[0] = proj_ob[(Forward_Static_Path_length + 1) << 2];
              }

              if ((!rtIsNaN(proj_ob[((Forward_Static_Path_length + 1) << 2) + 1]))
                  && (rtIsNaN(D[1]) || (D[1] > proj_ob
                    [((Forward_Static_Path_length + 1) << 2) + 1]))) {
                D[1] = proj_ob[((Forward_Static_Path_length + 1) << 2) + 1];
              }

              if ((!rtIsNaN(proj_ob[((Forward_Static_Path_length + 1) << 2) + 2]))
                  && (rtIsNaN(D[2]) || (D[2] > proj_ob
                    [((Forward_Static_Path_length + 1) << 2) + 2]))) {
                D[2] = proj_ob[((Forward_Static_Path_length + 1) << 2) + 2];
              }

              if ((!rtIsNaN(proj_ob[((Forward_Static_Path_length + 1) << 2) + 3]))
                  && (rtIsNaN(D[3]) || (D[3] > proj_ob
                    [((Forward_Static_Path_length + 1) << 2) + 3]))) {
                D[3] = proj_ob[((Forward_Static_Path_length + 1) << 2) + 3];
              }

              J_minvalue_diff = target_k;
              if ((!rtIsNaN(proj_ob[(Forward_Static_Path_length + 1) << 2])) &&
                  (rtIsNaN(target_k) || (target_k < proj_ob
                    [(Forward_Static_Path_length + 1) << 2]))) {
                J_minvalue_diff = proj_ob[(Forward_Static_Path_length + 1) << 2];
              }

              target_k = J_minvalue_diff;
              J_minvalue_diff = x_endpoint1;
              if ((!rtIsNaN(proj_ob[((Forward_Static_Path_length + 1) << 2) + 1]))
                  && (rtIsNaN(x_endpoint1) || (x_endpoint1 < proj_ob
                    [((Forward_Static_Path_length + 1) << 2) + 1]))) {
                J_minvalue_diff = proj_ob[((Forward_Static_Path_length + 1) << 2)
                  + 1];
              }

              x_endpoint1 = J_minvalue_diff;
              J_minvalue_diff = y_endpoint1;
              if ((!rtIsNaN(proj_ob[((Forward_Static_Path_length + 1) << 2) + 2]))
                  && (rtIsNaN(y_endpoint1) || (y_endpoint1 < proj_ob
                    [((Forward_Static_Path_length + 1) << 2) + 2]))) {
                J_minvalue_diff = proj_ob[((Forward_Static_Path_length + 1) << 2)
                  + 2];
              }

              y_endpoint1 = J_minvalue_diff;
              J_minvalue_diff = ang_1;
              if ((!rtIsNaN(proj_ob[((Forward_Static_Path_length + 1) << 2) + 3]))
                  && (rtIsNaN(ang_1) || (ang_1 < proj_ob
                    [((Forward_Static_Path_length + 1) << 2) + 3]))) {
                J_minvalue_diff = proj_ob[((Forward_Static_Path_length + 1) << 2)
                  + 3];
              }

              ang_1 = J_minvalue_diff;
            }

            minmax_obj[0] = D[0];
            minmax_obj[4] = target_k;
            minmax_obj[1] = D[1];
            minmax_obj[5] = x_endpoint1;
            minmax_obj[2] = D[2];
            minmax_obj[6] = y_endpoint1;
            minmax_obj[3] = D[3];
            minmax_obj[7] = ang_1;
            Static_PathCycle = 0;
            exitg3 = false;
            while ((!exitg3) && (Static_PathCycle < 4)) {
              if (minmax_veh[Static_PathCycle] > minmax_obj[4 + Static_PathCycle])
              {
                Path_col[Forward_Static_Path_length_0 << 2] = 0.0;
                exitg3 = true;
              } else if (minmax_veh[4 + Static_PathCycle] <
                         minmax_obj[Static_PathCycle]) {
                Path_col[Forward_Static_Path_length_0 << 2] = 0.0;
                exitg3 = true;
              } else {
                Path_col[Forward_Static_Path_length_0 << 2] = 1.0;
                Static_PathCycle++;
              }
            }

            if (Path_col[Forward_Static_Path_length_0 << 2] == 1.0) {
              Path_col[2 + (Forward_Static_Path_length_0 << 2)] = 1.0 + (real_T)
                break_count;
              exitg2 = true;
            } else {
              break_count++;
            }
          }

          if (Path_col[Forward_Static_Path_length_0 << 2] == 1.0) {
            Path_col[1 + (Forward_Static_Path_length_0 << 2)] = 1.0 + (real_T)
              case_0;
            exitg1 = true;
          } else {
            case_0++;
          }
        }
      }
    }
  }

  target_k = End_x_tmp_tmp / count * 10.0;
  for (Forward_Static_Path_length = 0; Forward_Static_Path_length < 13;
       Forward_Static_Path_length++) {
    x_endpoint1 = Path_col[Forward_Static_Path_length << 2];
    if (Path_col[(Forward_Static_Path_length << 2) + 1] > target_k) {
      x_endpoint1 = 0.0;
    }

    offset[Forward_Static_Path_length] = x_endpoint1;
    Cobs_0[Forward_Static_Path_length] = x_endpoint1;
  }

  ang_1 = std(Cobs_0);
  if (ang_1 != 0.0) {
    J_minvalue_diff = ang_1 * ang_1 * 2.0;
    Length_1 = 2.5066282746310002 * ang_1;
    for (Forward_Static_Path_length = 0; Forward_Static_Path_length < 13;
         Forward_Static_Path_length++) {
      i = 1 + Forward_Static_Path_length;
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 13;
           Path_RES_0_size_idx_1++) {
        Cc_0[Path_RES_0_size_idx_1] = (i - Path_RES_0_size_idx_1) - 1;
      }

      power_bvwt(Cc_0, rtb_forward_length_free);
      for (i = 0; i < 13; i++) {
        Cc_0[i] = -rtb_forward_length_free[i] / J_minvalue_diff;
      }

      exp_n(Cc_0);
      for (i = 0; i < 13; i++) {
        Clane[i] = Cc_0[i] / Length_1 * Cobs_0[i];
      }

      offset[Forward_Static_Path_length] = sum(Clane);
      if ((1 + Forward_Static_Path_length == 1) && (Cobs_0[0] == 1.0)) {
        offset[0] += std::exp(-1.0 / (ang_1 * ang_1 * 2.0)) /
          (2.5066282746310002 * ang_1);
      } else {
        if ((1 + Forward_Static_Path_length == 13) && (Cobs_0[12] == 1.0)) {
          offset[12] += std::exp(-1.0 / (ang_1 * ang_1 * 2.0)) /
            (2.5066282746310002 * ang_1);
        }
      }
    }

    ex = rtIsNaN(offset[0]);
    if (!ex) {
      Forward_Static_Path_length = 1;
    } else {
      Forward_Static_Path_length = 0;
      case_0 = 2;
      exitg1 = false;
      while ((!exitg1) && (case_0 < 14)) {
        if (!rtIsNaN(offset[case_0 - 1])) {
          Forward_Static_Path_length = case_0;
          exitg1 = true;
        } else {
          case_0++;
        }
      }
    }

    if (Forward_Static_Path_length == 0) {
      ang_1 = offset[0];
    } else {
      ang_1 = offset[Forward_Static_Path_length - 1];
      while (Forward_Static_Path_length + 1 < 14) {
        if (ang_1 < offset[Forward_Static_Path_length]) {
          ang_1 = offset[Forward_Static_Path_length];
        }

        Forward_Static_Path_length++;
      }
    }

    if (ang_1 != 1.0) {
      if (!ex) {
        Forward_Static_Path_length = 1;
      } else {
        Forward_Static_Path_length = 0;
        case_0 = 2;
        exitg1 = false;
        while ((!exitg1) && (case_0 < 14)) {
          if (!rtIsNaN(offset[case_0 - 1])) {
            Forward_Static_Path_length = case_0;
            exitg1 = true;
          } else {
            case_0++;
          }
        }
      }

      if (Forward_Static_Path_length == 0) {
        ang_1 = offset[0];
      } else {
        ang_1 = offset[Forward_Static_Path_length - 1];
        while (Forward_Static_Path_length + 1 < 14) {
          if (ang_1 < offset[Forward_Static_Path_length]) {
            ang_1 = offset[Forward_Static_Path_length];
          }

          Forward_Static_Path_length++;
        }
      }

      for (i = 0; i < 13; i++) {
        offset[i] /= ang_1;
      }
    }
  }

  for (i = 0; i < 13; i++) {
    Clane[i] = Path_col[(i << 2) + 3];
    Cobs_0[i] = Path_col[(i << 2) + 3];
  }

  ang_1 = std(Cobs_0);
  if (ang_1 != 0.0) {
    J_minvalue_diff = ang_1 * ang_1 * 2.0;
    Length_1 = 2.5066282746310002 * ang_1;
    for (Forward_Static_Path_length = 0; Forward_Static_Path_length < 13;
         Forward_Static_Path_length++) {
      i = 1 + Forward_Static_Path_length;
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 13;
           Path_RES_0_size_idx_1++) {
        Cc_0[Path_RES_0_size_idx_1] = (i - Path_RES_0_size_idx_1) - 1;
      }

      power_bvwt(Cc_0, rtb_forward_length_free);
      for (i = 0; i < 13; i++) {
        Cc_0[i] = -rtb_forward_length_free[i] / J_minvalue_diff;
      }

      exp_n(Cc_0);
      for (i = 0; i < 13; i++) {
        Cobs_0[i] = Path_col[(i << 2) + 3] * (Cc_0[i] / Length_1);
      }

      Clane[Forward_Static_Path_length] = sum(Cobs_0);
      if ((1 + Forward_Static_Path_length == 1) && (Path_col[3] == 1.0)) {
        Clane[0] += std::exp(-1.0 / (ang_1 * ang_1 * 2.0)) / (2.5066282746310002
          * ang_1);
      } else {
        if ((1 + Forward_Static_Path_length == 13) && (Path_col[51] == 1.0)) {
          Clane[12] += std::exp(-1.0 / (ang_1 * ang_1 * 2.0)) /
            (2.5066282746310002 * ang_1);
        }
      }
    }

    ex = rtIsNaN(Clane[0]);
    if (!ex) {
      Forward_Static_Path_length = 1;
    } else {
      Forward_Static_Path_length = 0;
      case_0 = 2;
      exitg1 = false;
      while ((!exitg1) && (case_0 < 14)) {
        if (!rtIsNaN(Clane[case_0 - 1])) {
          Forward_Static_Path_length = case_0;
          exitg1 = true;
        } else {
          case_0++;
        }
      }
    }

    if (Forward_Static_Path_length == 0) {
      ang_1 = Clane[0];
    } else {
      ang_1 = Clane[Forward_Static_Path_length - 1];
      while (Forward_Static_Path_length + 1 < 14) {
        if (ang_1 < Clane[Forward_Static_Path_length]) {
          ang_1 = Clane[Forward_Static_Path_length];
        }

        Forward_Static_Path_length++;
      }
    }

    if (ang_1 != 1.0) {
      if (!ex) {
        Forward_Static_Path_length = 1;
      } else {
        Forward_Static_Path_length = 0;
        case_0 = 2;
        exitg1 = false;
        while ((!exitg1) && (case_0 < 14)) {
          if (!rtIsNaN(Clane[case_0 - 1])) {
            Forward_Static_Path_length = case_0;
            exitg1 = true;
          } else {
            case_0++;
          }
        }
      }

      if (Forward_Static_Path_length == 0) {
        ang_1 = Clane[0];
      } else {
        ang_1 = Clane[Forward_Static_Path_length - 1];
        while (Forward_Static_Path_length + 1 < 14) {
          if (ang_1 < Clane[Forward_Static_Path_length]) {
            ang_1 = Clane[Forward_Static_Path_length];
          }

          Forward_Static_Path_length++;
        }
      }

      for (i = 0; i < 13; i++) {
        Clane[i] /= ang_1;
      }
    }
  }

  for (i = 0; i < 11; i++) {
    X1[i] = rtDW.UnitDelay6_DSTATE[i] - End_x;
  }

  power_b(X1, K1);
  for (i = 0; i < 11; i++) {
    X1[i] = rtDW.UnitDelay6_DSTATE[11 + i] - End_y;
  }

  power_b(X1, X2);
  for (i = 0; i < 11; i++) {
    K_11[i] = K1[i] + X2[i];
  }

  sqrt_f(K_11);
  if (!rtIsNaN(K_11[0])) {
    Forward_Static_Path_length = 1;
  } else {
    Forward_Static_Path_length = 0;
    case_0 = 2;
    exitg1 = false;
    while ((!exitg1) && (case_0 < 12)) {
      if (!rtIsNaN(K_11[case_0 - 1])) {
        Forward_Static_Path_length = case_0;
        exitg1 = true;
      } else {
        case_0++;
      }
    }
  }

  if (Forward_Static_Path_length == 0) {
    Forward_Static_Path_length = 1;
  } else {
    ang_1 = K_11[Forward_Static_Path_length - 1];
    for (case_0 = Forward_Static_Path_length; case_0 + 1 < 12; case_0++) {
      if (ang_1 > K_11[case_0]) {
        ang_1 = K_11[case_0];
        Forward_Static_Path_length = case_0 + 1;
      }
    }
  }

  end_ind_0 = 12 - Forward_Static_Path_length;
  loop_ub = -Forward_Static_Path_length;
  for (i = 0; i <= loop_ub + 11; i++) {
    LastPath_overlap_data[i] = rtDW.UnitDelay6_DSTATE
      [(Forward_Static_Path_length + i) - 1];
  }

  loop_ub = -Forward_Static_Path_length;
  for (i = 0; i <= loop_ub + 11; i++) {
    LastPath_overlap_data[i + end_ind_0] = rtDW.UnitDelay6_DSTATE
      [(Forward_Static_Path_length + i) + 10];
  }

  for (case_0 = 0; case_0 < 13; case_0++) {
    for (i = 0; i < 11; i++) {
      b_Path_dis_data[i] = X_2[11 * case_0 + i] - rtDW.UnitDelay6_DSTATE[10];
    }

    power_b(b_Path_dis_data, X1);
    for (i = 0; i < 11; i++) {
      b_Path_dis_data[i] = Y[11 * case_0 + i] - rtDW.UnitDelay6_DSTATE[21];
    }

    power_b(b_Path_dis_data, K1);
    for (i = 0; i < 11; i++) {
      K_11[i] = X1[i] + K1[i];
    }

    sqrt_f(K_11);
    if (!rtIsNaN(K_11[0])) {
      Forward_Static_Path_length_0 = 0;
    } else {
      Forward_Static_Path_length_0 = -1;
      break_count = 2;
      exitg1 = false;
      while ((!exitg1) && (break_count < 12)) {
        if (!rtIsNaN(K_11[break_count - 1])) {
          Forward_Static_Path_length_0 = break_count - 1;
          exitg1 = true;
        } else {
          break_count++;
        }
      }
    }

    if (Forward_Static_Path_length_0 + 1 == 0) {
      Forward_Static_Path_length_0 = 0;
    } else {
      J_minvalue_diff = K_11[Forward_Static_Path_length_0];
      for (break_count = Forward_Static_Path_length_0 + 1; break_count + 1 < 12;
           break_count++) {
        if (J_minvalue_diff > K_11[break_count]) {
          J_minvalue_diff = K_11[break_count];
          Forward_Static_Path_length_0 = break_count;
        }
      }
    }

    Path_overlap_size[0] = Forward_Static_Path_length_0 + 1;
    if (0 <= Forward_Static_Path_length_0) {
      memcpy(&Path_overlap_data[0], &X_2[case_0 * 11],
             (Forward_Static_Path_length_0 + 1) * sizeof(real_T));
    }

    for (i = 0; i <= Forward_Static_Path_length_0; i++) {
      Path_overlap_data[i + Path_overlap_size[0]] = Y[11 * case_0 + i];
    }

    if (12 - Forward_Static_Path_length >= Path_overlap_size[0]) {
      break_count = 13 - (Forward_Static_Path_length + Path_overlap_size[0]);
      if (break_count > 12 - Forward_Static_Path_length) {
        break_count = 1;
        Forward_Static_Path_length_0 = 0;
      } else {
        Forward_Static_Path_length_0 = 12 - Forward_Static_Path_length;
      }

      i = break_count - 1;
      Forward_Static_Path_length_0 -= i;
      LastPath_overlap_size_2[0] = Forward_Static_Path_length_0;
      LastPath_overlap_size_2[1] = 2;
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 <
           Forward_Static_Path_length_0; Path_RES_0_size_idx_1++) {
        LastPath_overlap_data_0[Path_RES_0_size_idx_1] = LastPath_overlap_data[i
          + Path_RES_0_size_idx_1] - Path_overlap_data[Path_RES_0_size_idx_1];
      }

      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 <
           Forward_Static_Path_length_0; Path_RES_0_size_idx_1++) {
        LastPath_overlap_data_0[Path_RES_0_size_idx_1 +
          Forward_Static_Path_length_0] = LastPath_overlap_data[(i +
          Path_RES_0_size_idx_1) + end_ind_0] -
          Path_overlap_data[Path_RES_0_size_idx_1 + Path_overlap_size[0]];
      }

      power_pcxfb(LastPath_overlap_data_0, LastPath_overlap_size_2,
                  Path_overlap_data, Path_overlap_size);
      Path_overlap_size_3[0] = 2;
      Path_overlap_size_3[1] = Path_overlap_size[0];
      loop_ub = Path_overlap_size[0];
      for (i = 0; i < loop_ub; i++) {
        LastPath_overlap_data_0[i << 1] = Path_overlap_data[i];
        LastPath_overlap_data_0[1 + (i << 1)] = Path_overlap_data[i +
          Path_overlap_size[0]];
      }

      sum_h1(LastPath_overlap_data_0, Path_overlap_size_3, b_Path_dis_data,
             b_Path_dis_size);
      sqrt_fh(b_Path_dis_data, b_Path_dis_size);
      loop_ub = b_Path_dis_size[1];
      for (i = 0; i < loop_ub; i++) {
        K_11[i] = b_Path_dis_data[b_Path_dis_size[0] * i];
      }

      i = b_Path_dis_size[1];
      c = b_Path_dis_size[1];
      if (0 <= i - 1) {
        memcpy(&Path_dis_data[0], &K_11[0], i * sizeof(real_T));
      }
    } else {
      Forward_Static_Path_length_0 = 12 - Forward_Static_Path_length;
      LastPath_overlap_size_1[0] = Forward_Static_Path_length_0;
      LastPath_overlap_size_1[1] = 2;
      for (i = 0; i < Forward_Static_Path_length_0; i++) {
        LastPath_overlap_data_0[i] = LastPath_overlap_data[i] -
          Path_overlap_data[i];
      }

      for (i = 0; i < Forward_Static_Path_length_0; i++) {
        LastPath_overlap_data_0[i + Forward_Static_Path_length_0] =
          LastPath_overlap_data[i + end_ind_0] - Path_overlap_data[i +
          Path_overlap_size[0]];
      }

      power_pcxfb(LastPath_overlap_data_0, LastPath_overlap_size_1,
                  Path_overlap_data, Path_overlap_size);
      Path_overlap_size_2[0] = 2;
      Path_overlap_size_2[1] = Path_overlap_size[0];
      loop_ub = Path_overlap_size[0];
      for (i = 0; i < loop_ub; i++) {
        LastPath_overlap_data_0[i << 1] = Path_overlap_data[i];
        LastPath_overlap_data_0[1 + (i << 1)] = Path_overlap_data[i +
          Path_overlap_size[0]];
      }

      sum_h1(LastPath_overlap_data_0, Path_overlap_size_2, b_Path_dis_data,
             b_Path_dis_size);
      sqrt_fh(b_Path_dis_data, b_Path_dis_size);
      loop_ub = b_Path_dis_size[1];
      for (i = 0; i < loop_ub; i++) {
        b_Path_dis_data_0[i] = b_Path_dis_data[b_Path_dis_size[0] * i];
      }

      i = b_Path_dis_size[1];
      c = b_Path_dis_size[1];
      if (0 <= i - 1) {
        memcpy(&Path_dis_data[0], &b_Path_dis_data_0[0], i * sizeof(real_T));
      }
    }

    if (c > 1) {
      i = c;
    } else {
      i = 1;
    }

    if (mod((real_T)i) == 0.0) {
      if (c > 1) {
        Forward_Static_Path_length_0 = c - 1;
      } else {
        Forward_Static_Path_length_0 = 0;
      }

      b_Path_dis_size[1] = Forward_Static_Path_length_0;
      loop_ub = Forward_Static_Path_length_0 - 1;
      for (i = 0; i <= loop_ub; i++) {
        b_Path_dis_data[i] = 4.0;
      }
    } else {
      if (c > 1) {
        Forward_Static_Path_length_0 = c;
      } else {
        Forward_Static_Path_length_0 = 1;
      }

      b_Path_dis_size[1] = Forward_Static_Path_length_0;
      loop_ub = Forward_Static_Path_length_0 - 1;
      for (i = 0; i <= loop_ub; i++) {
        b_Path_dis_data[i] = 4.0;
      }
    }

    b_Path_dis_data[0] = 1.0;
    b_Path_dis_data[b_Path_dis_size[1] - 1] = 1.0;
    if (3 > b_Path_dis_size[1] - 2) {
      Static_PathCycle = 1;
      break_count = 1;
      Forward_Static_Path_length_0 = 0;
    } else {
      Static_PathCycle = 3;
      break_count = 2;
      Forward_Static_Path_length_0 = b_Path_dis_size[1] - 2;
    }

    Forward_Static_Path_length_0 = div_nde_s32_floor((int8_T)
      Forward_Static_Path_length_0 - Static_PathCycle, break_count);
    for (i = 0; i <= Forward_Static_Path_length_0; i++) {
      p_data[i] = (int8_T)((int8_T)((int8_T)(break_count * (int8_T)i) +
        Static_PathCycle) - 1);
    }

    for (i = 0; i <= Forward_Static_Path_length_0; i++) {
      b_Path_dis_data[p_data[i]] = 2.0;
    }

    offset_5 = 0.0;
    for (i = 0; i < b_Path_dis_size[1]; i++) {
      offset_5 += b_Path_dis_data[i] * Path_dis_data[i];
    }

    if (!(c > 1)) {
      c = 1;
    }

    Cobs_0[case_0] = Cobs[case_0] / 11.0 * offset_5 / 3.0 / (Cobs[case_0] *
      (real_T)c / 11.0);
  }

  for (i = 0; i < 13; i++) {
    Cobs[i] = 1.0;
    Cc_0[i] = 0.0;
    rtb_forward_length_free[i] = 0.0;
  }

  if ((rtU.Freespace_mode == 1.0) || (rtU.Freespace_mode == 2.0)) {
    for (Forward_Static_Path_length_0 = 0; Forward_Static_Path_length_0 < 13;
         Forward_Static_Path_length_0++) {
      FreespaceDetectCollision_m(rtU.Freespace, &rtb_XP[6 *
        Forward_Static_Path_length_0], &rtb_YP[6 * Forward_Static_Path_length_0],
        rtb_TmpSignalConversionAtSFun_n, count, rtU.safe_range, rtConstP.pooled5,
        rtConstP.pooled4, &Cobs[Forward_Static_Path_length_0],
        &Cc_0[Forward_Static_Path_length_0],
        &rtb_forward_length_free[Forward_Static_Path_length_0]);
    }
  }

  ex = rtIsNaN(Cobs_0[0]);
  if (!ex) {
    Forward_Static_Path_length = 1;
  } else {
    Forward_Static_Path_length = 0;
    case_0 = 2;
    exitg1 = false;
    while ((!exitg1) && (case_0 < 14)) {
      if (!rtIsNaN(Cobs_0[case_0 - 1])) {
        Forward_Static_Path_length = case_0;
        exitg1 = true;
      } else {
        case_0++;
      }
    }
  }

  if (Forward_Static_Path_length == 0) {
    ang_1 = Cobs_0[0];
  } else {
    ang_1 = Cobs_0[Forward_Static_Path_length - 1];
    while (Forward_Static_Path_length + 1 < 14) {
      if (ang_1 < Cobs_0[Forward_Static_Path_length]) {
        ang_1 = Cobs_0[Forward_Static_Path_length];
      }

      Forward_Static_Path_length++;
    }
  }

  if (!(ang_1 == 0.0)) {
    if (!ex) {
      Forward_Static_Path_length = 1;
    } else {
      Forward_Static_Path_length = 0;
      case_0 = 2;
      exitg1 = false;
      while ((!exitg1) && (case_0 < 14)) {
        if (!rtIsNaN(Cobs_0[case_0 - 1])) {
          Forward_Static_Path_length = case_0;
          exitg1 = true;
        } else {
          case_0++;
        }
      }
    }

    if (Forward_Static_Path_length == 0) {
      Length_1 = Cobs_0[0];
    } else {
      Length_1 = Cobs_0[Forward_Static_Path_length - 1];
      while (Forward_Static_Path_length + 1 < 14) {
        if (Length_1 < Cobs_0[Forward_Static_Path_length]) {
          Length_1 = Cobs_0[Forward_Static_Path_length];
        }

        Forward_Static_Path_length++;
      }
    }

    for (i = 0; i < 13; i++) {
      Cobs_0[i] /= Length_1;
    }
  }

  for (i = 0; i < 13; i++) {
    offset[i] = (rtU.W_2[0] * offset[i] + rtU.W_2[1] * Cobs_0[i]) + rtU.W_2[2] *
      Clane[i];
  }

  for (Forward_Static_Path_length_0 = 0; Forward_Static_Path_length_0 < 13;
       Forward_Static_Path_length_0++) {
    // MATLAB Function: '<S1>/J_fsc_design' incorporates:
    //   Inport: '<Root>/w_fs'

    if (rtb_U_c_n[Forward_Static_Path_length_0] == 1.0) {
      x_endpoint1 = rtb_forward_length_free_f[Forward_Static_Path_length_0] +
        rtb_forward_length_free[Forward_Static_Path_length_0];
      if (Cobs[Forward_Static_Path_length_0] == 1.0) {
        ang_1_tmp = 0.0;
      } else {
        ang_1_tmp = 2.0 - Cobs[Forward_Static_Path_length_0];
      }

      ang_1_tmp = ang_1_tmp * rtU.w_fs + Cc_0[Forward_Static_Path_length_0];
    } else {
      x_endpoint1 = rtb_forward_length_free_f[Forward_Static_Path_length_0];
      if (rtb_U_c_n[Forward_Static_Path_length_0] == 1.0) {
        ang_1_tmp = 0.0;
      } else {
        ang_1_tmp = 3.0 - rtb_U_c_n[Forward_Static_Path_length_0];
      }

      ang_1_tmp = ang_1_tmp * rtU.w_fs +
        rtb_safety_level_all_b[Forward_Static_Path_length_0];
    }

    if (x_endpoint1 > End_x_tmp_tmp) {
      ang_1_tmp = 0.0;
    }

    rtb_forward_length_free_2[Forward_Static_Path_length_0] = x_endpoint1;
    Clane[Forward_Static_Path_length_0] = ang_1_tmp;

    // End of MATLAB Function: '<S1>/J_fsc_design'

    // MATLAB Function: '<S1>/Fianl_Path_Decision'
    rtb_J_out_a[Forward_Static_Path_length_0] =
      (rtb_J_out_a[Forward_Static_Path_length_0] +
       offset[Forward_Static_Path_length_0]) + ang_1_tmp;
  }

  // MATLAB Function: '<S1>/Fianl_Path_Decision' incorporates:
  //   Inport: '<Root>/Freespace_mode'
  //   Inport: '<Root>/J_minvalue_diff_min'
  //   Inport: '<Root>/J_minvalue_index'
  //   Inport: '<Root>/Path_flag'
  //   Inport: '<Root>/W_1'
  //   Inport: '<Root>/w_fs'
  //   UnitDelay: '<S1>/Unit Delay11'
  //   UnitDelay: '<S1>/Unit Delay13'
  //   UnitDelay: '<S1>/Unit Delay7'

  if (!rtIsNaN(rtb_J_out_a[0])) {
    Forward_Static_Path_length = 1;
  } else {
    Forward_Static_Path_length = 0;
    break_count = 2;
    exitg1 = false;
    while ((!exitg1) && (break_count < 14)) {
      if (!rtIsNaN(rtb_J_out_a[break_count - 1])) {
        Forward_Static_Path_length = break_count;
        exitg1 = true;
      } else {
        break_count++;
      }
    }
  }

  if (Forward_Static_Path_length == 0) {
    ang_1 = rtb_J_out_a[0];
    Forward_Static_Path_length = 1;
  } else {
    ang_1 = rtb_J_out_a[Forward_Static_Path_length - 1];
    for (case_0 = Forward_Static_Path_length; case_0 + 1 < 14; case_0++) {
      if (ang_1 > rtb_J_out_a[case_0]) {
        ang_1 = rtb_J_out_a[case_0];
        Forward_Static_Path_length = case_0 + 1;
      }
    }
  }

  J_minvalue_diff = std::abs(rtDW.UnitDelay11_DSTATE - ang_1);
  Length_1 = 0.0;
  if (rtU.Path_flag == 1.0) {
    if ((rtU.Freespace_mode == 0.0) || (rtU.Freespace_mode == 2.0)) {
      if (ang_1 >= rtU.W_1[3]) {
        J_minvalue_diff = rtDW.UnitDelay7_DSTATE;
      } else if ((J_minvalue_diff < rtU.J_minvalue_diff_min) &&
                 (rtDW.UnitDelay13_DSTATE < rtU.J_minvalue_index)) {
        J_minvalue_diff = rtDW.UnitDelay7_DSTATE;
        Length_1 = rtDW.UnitDelay13_DSTATE + 1.0;
      } else {
        J_minvalue_diff = Forward_Static_Path_length;
      }
    } else if (ang_1 >= rtU.w_fs) {
      J_minvalue_diff = rtDW.UnitDelay7_DSTATE;
    } else if ((J_minvalue_diff < rtU.J_minvalue_diff_min) &&
               (rtDW.UnitDelay13_DSTATE < rtU.J_minvalue_index)) {
      J_minvalue_diff = rtDW.UnitDelay7_DSTATE;
      Length_1 = rtDW.UnitDelay13_DSTATE + 1.0;
    } else {
      J_minvalue_diff = Forward_Static_Path_length;
    }
  } else {
    J_minvalue_diff = 7.0;
  }

  break_count = (int32_T)J_minvalue_diff;
  case_0 = (int32_T)J_minvalue_diff;
  Static_PathCycle = (int32_T)J_minvalue_diff;
  loop_ub = (int32_T)J_minvalue_diff;
  end_ind_0 = (int32_T)J_minvalue_diff;
  c = (int32_T)J_minvalue_diff;
  Path_RES_0_size_idx_1 = (int32_T)J_minvalue_diff;
  xy_ends_POS_size_idx_0 = (int32_T)J_minvalue_diff;
  Path_RES_1_size_idx_0 = (int32_T)J_minvalue_diff;
  J_minvalue_diff_0 = (int32_T)J_minvalue_diff;
  J_minvalue_diff_1 = (int32_T)J_minvalue_diff;
  J_minvalue_diff_2 = (int32_T)J_minvalue_diff;
  for (Forward_Static_Path_length_0 = 0; Forward_Static_Path_length_0 < 11;
       Forward_Static_Path_length_0++) {
    ang_1_tmp = a[Forward_Static_Path_length_0] * a[Forward_Static_Path_length_0];

    // Update for UnitDelay: '<S1>/Unit Delay5'
    rtDW.UnitDelay5_DSTATE[Forward_Static_Path_length_0] = ((((rtb_XP_i[(case_0
      - 1) * 6 + 1] * a[Forward_Static_Path_length_0] + rtb_XP_i[(break_count -
      1) * 6]) + rtb_XP_i[(Static_PathCycle - 1) * 6 + 2] * ang_1_tmp) +
      rtb_XP_i[(loop_ub - 1) * 6 + 3] * rt_powd_snf
      (a[Forward_Static_Path_length_0], 3.0)) + rtb_XP_i[(end_ind_0 - 1) * 6 + 4]
      * rt_powd_snf(a[Forward_Static_Path_length_0], 4.0)) + rtb_XP_i[(c - 1) *
      6 + 5] * rt_powd_snf(a[Forward_Static_Path_length_0], 5.0);
    rtDW.UnitDelay5_DSTATE[Forward_Static_Path_length_0 + 11] = ((((rtb_YP_p
      [(xy_ends_POS_size_idx_0 - 1) * 6 + 1] * a[Forward_Static_Path_length_0] +
      rtb_YP_p[(Path_RES_0_size_idx_1 - 1) * 6]) + rtb_YP_p
      [(Path_RES_1_size_idx_0 - 1) * 6 + 2] * ang_1_tmp) + rtb_YP_p
      [(J_minvalue_diff_0 - 1) * 6 + 3] * rt_powd_snf
      (a[Forward_Static_Path_length_0], 3.0)) + rtb_YP_p[(J_minvalue_diff_1 - 1)
      * 6 + 4] * rt_powd_snf(a[Forward_Static_Path_length_0], 4.0)) + rtb_YP_p
      [(J_minvalue_diff_2 - 1) * 6 + 5] * rt_powd_snf
      (a[Forward_Static_Path_length_0], 5.0);
    X1[Forward_Static_Path_length_0] = ang_1_tmp;
    b_Path_dis_data[Forward_Static_Path_length_0] = rt_powd_snf
      (a[Forward_Static_Path_length_0], 3.0);
    K1[Forward_Static_Path_length_0] = rt_powd_snf
      (a[Forward_Static_Path_length_0], 4.0);
    K_11[Forward_Static_Path_length_0] = rt_powd_snf
      (a[Forward_Static_Path_length_0], 5.0);
    X2[Forward_Static_Path_length_0] = ang_1_tmp;
    Y2[Forward_Static_Path_length_0] = rt_powd_snf
      (a[Forward_Static_Path_length_0], 3.0);
    K2[Forward_Static_Path_length_0] = rt_powd_snf
      (a[Forward_Static_Path_length_0], 4.0);
    K_12[Forward_Static_Path_length_0] = rt_powd_snf
      (a[Forward_Static_Path_length_0], 5.0);
  }

  // Outport: '<Root>/J'
  memcpy(&rtY.J[0], &rtb_J_out_a[0], 13U * sizeof(real_T));

  // Outport: '<Root>/J_fsc'
  memcpy(&rtY.J_fsc[0], &Clane[0], 13U * sizeof(real_T));

  // Outport: '<Root>/U_c_1'
  memcpy(&rtY.U_c_1[0], &Cobs[0], 13U * sizeof(real_T));

  // Outport: '<Root>/safety_level_all_1'
  memcpy(&rtY.safety_level_all_1[0], &Cc_0[0], 13U * sizeof(real_T));

  // Outport: '<Root>/U_c'
  memcpy(&rtY.U_c[0], &rtb_U_c_n[0], 13U * sizeof(real_T));

  // Outport: '<Root>/safety_level_all'
  memcpy(&rtY.safety_level_all[0], &rtb_safety_level_all_b[0], 13U * sizeof
         (real_T));

  // MATLAB Function: '<S1>/Fianl_Path_Decision'
  break_count = (int32_T)J_minvalue_diff;
  case_0 = (int32_T)J_minvalue_diff;
  Static_PathCycle = (int32_T)J_minvalue_diff;
  loop_ub = (int32_T)J_minvalue_diff;
  for (i = 0; i < 6; i++) {
    // Outport: '<Root>/XP_final' incorporates:
    //   MATLAB Function: '<S1>/Fianl_Path_Decision'

    rtY.XP_final[i] = rtb_XP_i[(break_count - 1) * 6 + i];

    // Outport: '<Root>/YP_final' incorporates:
    //   MATLAB Function: '<S1>/Fianl_Path_Decision'

    rtY.YP_final[i] = rtb_YP_p[(case_0 - 1) * 6 + i];

    // Outport: '<Root>/XP_final_1' incorporates:
    //   MATLAB Function: '<S1>/Fianl_Path_Decision'

    rtY.XP_final_1[i] = rtb_XP[(Static_PathCycle - 1) * 6 + i];

    // Outport: '<Root>/YP_final_1' incorporates:
    //   MATLAB Function: '<S1>/Fianl_Path_Decision'

    rtY.YP_final_1[i] = rtb_YP[(loop_ub - 1) * 6 + i];
  }

  // SignalConversion: '<S17>/TmpSignal ConversionAt SFunction Inport1' incorporates:
  //   Gain: '<S1>/Gain2'
  //   MATLAB Function: '<S1>/Target_Point_Decision'

  rtb_TmpSignalConversionAtSFun_n[2] = 0.017453292519943295 * vehicle_heading;

  // MATLAB Function: '<S1>/Target_Point_Decision' incorporates:
  //   Inport: '<Root>/Look_ahead_S0'
  //   Inport: '<Root>/Speed_mps1'
  //   Inport: '<Root>/X_UKF_SLAM_i1'
  //   MATLAB Function: '<S1>/EndPointDecision'
  //   MATLAB Function: '<S1>/EndPointDecision1'
  //   MATLAB Function: '<S1>/Fianl_Path_Decision'
  //   MATLAB Function: '<S1>/MATLAB Function'

  offset_5 = rtU.Speed_mps1 * rtb_Look_ahead_time + rtU.Look_ahead_S0;
  if (offset_5 <= x_target) {
    vehicle_heading = offset_5 / x_target;
    x_target = ((((rtb_XP_i[((int32_T)J_minvalue_diff - 1) * 6 + 1] *
                   vehicle_heading + rtb_XP_i[((int32_T)J_minvalue_diff - 1) * 6])
                  + rtb_XP_i[((int32_T)J_minvalue_diff - 1) * 6 + 2] *
                  (vehicle_heading * vehicle_heading)) + rtb_XP_i[((int32_T)
      J_minvalue_diff - 1) * 6 + 3] * rt_powd_snf(vehicle_heading, 3.0)) +
                rtb_XP_i[((int32_T)J_minvalue_diff - 1) * 6 + 4] * rt_powd_snf
                (vehicle_heading, 4.0)) + rtb_XP_i[((int32_T)J_minvalue_diff - 1)
      * 6 + 5] * rt_powd_snf(vehicle_heading, 5.0);
    vehicle_heading = ((((rtb_YP_p[((int32_T)J_minvalue_diff - 1) * 6 + 1] *
                          vehicle_heading + rtb_YP_p[((int32_T)J_minvalue_diff -
      1) * 6]) + rtb_YP_p[((int32_T)J_minvalue_diff - 1) * 6 + 2] *
                         (vehicle_heading * vehicle_heading)) + rtb_YP_p
                        [((int32_T)J_minvalue_diff - 1) * 6 + 3] * rt_powd_snf
                        (vehicle_heading, 3.0)) + rtb_YP_p[((int32_T)
      J_minvalue_diff - 1) * 6 + 4] * rt_powd_snf(vehicle_heading, 4.0)) +
      rtb_YP_p[((int32_T)J_minvalue_diff - 1) * 6 + 5] * rt_powd_snf
      (vehicle_heading, 5.0);
  } else if ((offset_5 > x_target) && (offset_5 <= x_target + count)) {
    vehicle_heading = (offset_5 - x_target) / count;
    count = vehicle_heading * vehicle_heading;
    x_target = ((((rtb_XP[((int32_T)J_minvalue_diff - 1) * 6 + 1] *
                   vehicle_heading + rtb_XP[((int32_T)J_minvalue_diff - 1) * 6])
                  + rtb_XP[((int32_T)J_minvalue_diff - 1) * 6 + 2] * count) +
                 rtb_XP[((int32_T)J_minvalue_diff - 1) * 6 + 3] * rt_powd_snf
                 (vehicle_heading, 3.0)) + rtb_XP[((int32_T)J_minvalue_diff - 1)
                * 6 + 4] * rt_powd_snf(vehicle_heading, 4.0)) + rtb_XP[((int32_T)
      J_minvalue_diff - 1) * 6 + 5] * rt_powd_snf(vehicle_heading, 5.0);
    vehicle_heading = ((((rtb_YP[((int32_T)J_minvalue_diff - 1) * 6 + 1] *
                          vehicle_heading + rtb_YP[((int32_T)J_minvalue_diff - 1)
                          * 6]) + rtb_YP[((int32_T)J_minvalue_diff - 1) * 6 + 2]
                         * count) + rtb_YP[((int32_T)J_minvalue_diff - 1) * 6 +
                        3] * rt_powd_snf(vehicle_heading, 3.0)) + rtb_YP
                       [((int32_T)J_minvalue_diff - 1) * 6 + 4] * rt_powd_snf
                       (vehicle_heading, 4.0)) + rtb_YP[((int32_T)
      J_minvalue_diff - 1) * 6 + 5] * rt_powd_snf(vehicle_heading, 5.0);
  } else {
    x_target = ((((rtb_XP[((int32_T)J_minvalue_diff - 1) * 6 + 1] + rtb_XP
                   [((int32_T)J_minvalue_diff - 1) * 6]) + rtb_XP[((int32_T)
      J_minvalue_diff - 1) * 6 + 2]) + rtb_XP[((int32_T)J_minvalue_diff - 1) * 6
                 + 3]) + rtb_XP[((int32_T)J_minvalue_diff - 1) * 6 + 4]) +
      rtb_XP[((int32_T)J_minvalue_diff - 1) * 6 + 5];
    vehicle_heading = ((((rtb_YP[((int32_T)J_minvalue_diff - 1) * 6 + 1] +
                          rtb_YP[((int32_T)J_minvalue_diff - 1) * 6]) + rtb_YP
                         [((int32_T)J_minvalue_diff - 1) * 6 + 2]) + rtb_YP
                        [((int32_T)J_minvalue_diff - 1) * 6 + 3]) + rtb_YP
                       [((int32_T)J_minvalue_diff - 1) * 6 + 4]) + rtb_YP
      [((int32_T)J_minvalue_diff - 1) * 6 + 5];
  }

  x_target -= rtU.X_UKF_SLAM_i1[0];
  vehicle_heading -= rtU.X_UKF_SLAM_i1[1];
  count = std::sin(-rtb_TmpSignalConversionAtSFun_n[2]);
  ang_1_tmp = std::cos(-rtb_TmpSignalConversionAtSFun_n[2]);

  // Outport: '<Root>/Vehicle_Target_x' incorporates:
  //   MATLAB Function: '<S1>/Target_Point_Decision'

  rtY.Vehicle_Target_x = ang_1_tmp * x_target + -count * vehicle_heading;

  // Outport: '<Root>/Vehicle_Target_y' incorporates:
  //   MATLAB Function: '<S1>/Target_Point_Decision'

  rtY.Vehicle_Target_y = count * x_target + ang_1_tmp * vehicle_heading;

  // Outport: '<Root>/J_minind' incorporates:
  //   MATLAB Function: '<S1>/Fianl_Path_Decision'

  rtY.J_minind = Forward_Static_Path_length;

  // Outport: '<Root>/J_finalind' incorporates:
  //   MATLAB Function: '<S1>/Fianl_Path_Decision'

  rtY.J_finalind = J_minvalue_diff;

  // Outport: '<Root>/forward_length_free' incorporates:
  //   MATLAB Function: '<S1>/Fianl_Path_Decision'

  rtY.forward_length_free = rtb_forward_length_free_2[(int32_T)J_minvalue_diff -
    1];

  // Outport: '<Root>/takeover_length'
  rtY.takeover_length = End_x_tmp_tmp;

  // Outport: '<Root>/takeoverlength_ind' incorporates:
  //   MATLAB Function: '<S1>/DynamicPathPlanning1'

  rtY.takeoverlength_ind = target_k;

  // Outport: '<Root>/OB_enlargescale' incorporates:
  //   MATLAB Function: '<S1>/DynamicPathPlanning1'

  rtY.OB_enlargescale = 0.15;

  // Outport: '<Root>/OB_enlargescale_frontbehind' incorporates:
  //   MATLAB Function: '<S1>/DynamicPathPlanning1'

  rtY.OB_enlargescale_frontbehind = 0.3;

  // Outport: '<Root>/avoidance_mode' incorporates:
  //   MATLAB Function: '<S1>/DangerousArea1'

  rtY.avoidance_mode = total_length;

  // Outport: '<Root>/Target_seg_id' incorporates:
  //   MATLAB Function: '<S1>/target_seg_id_search'

  rtY.Target_seg_id = rtb_Forward_Static_Path_id[rtDW.SFunction_DIMS4 - 1];

  // Outport: '<Root>/End_x' incorporates:
  //   MATLAB Function: '<S1>/EndPointDecision'

  rtY.End_x = End_x;

  // Outport: '<Root>/End_y' incorporates:
  //   MATLAB Function: '<S1>/EndPointDecision'

  rtY.End_y = End_y;

  // Outport: '<Root>/forward_length'
  rtY.forward_length = rtb_Forward_length_final;

  // Outport: '<Root>/Look_ahead_time'
  rtY.Look_ahead_time = rtb_Look_ahead_time;

  // Update for UnitDelay: '<S1>/Unit Delay14' incorporates:
  //   MATLAB Function: '<S1>/DangerousArea1'

  rtDW.UnitDelay14_DSTATE = total_length;

  // Update for UnitDelay: '<S1>/Unit Delay16'
  rtDW.UnitDelay16_DSTATE = rtb_Forward_length_final;

  // Update for UnitDelay: '<S1>/Unit Delay18' incorporates:
  //   MATLAB Function: '<S1>/DangerousArea1'

  rtDW.UnitDelay18_DSTATE = total_length;

  // Update for UnitDelay: '<S1>/Unit Delay17' incorporates:
  //   MATLAB Function: '<S1>/DangerousArea1'

  rtDW.UnitDelay17_DSTATE = rtb_UnitDelay18;

  // Update for UnitDelay: '<S1>/Unit Delay19'
  rtDW.UnitDelay19_DSTATE[0] = rtb_H_x_out[0];

  // Update for UnitDelay: '<S1>/Unit Delay15'
  rtDW.UnitDelay15_DSTATE[0] = rtb_H_y_out[0];

  // Update for UnitDelay: '<S1>/Unit Delay19'
  rtDW.UnitDelay19_DSTATE[1] = rtb_H_x_out[1];

  // Update for UnitDelay: '<S1>/Unit Delay15'
  rtDW.UnitDelay15_DSTATE[1] = rtb_H_y_out[1];

  // Update for UnitDelay: '<S1>/Unit Delay19'
  rtDW.UnitDelay19_DSTATE[2] = rtb_H_x_out[2];

  // Update for UnitDelay: '<S1>/Unit Delay15'
  rtDW.UnitDelay15_DSTATE[2] = rtb_H_y_out[2];

  // Update for UnitDelay: '<S1>/Unit Delay19'
  rtDW.UnitDelay19_DSTATE[3] = rtb_H_x_out[3];

  // Update for UnitDelay: '<S1>/Unit Delay15'
  rtDW.UnitDelay15_DSTATE[3] = rtb_H_y_out[3];

  // MATLAB Function: '<S1>/Fianl_Path_Decision'
  break_count = (int32_T)J_minvalue_diff;
  case_0 = (int32_T)J_minvalue_diff;
  Static_PathCycle = (int32_T)J_minvalue_diff;
  loop_ub = (int32_T)J_minvalue_diff;
  end_ind_0 = (int32_T)J_minvalue_diff;
  c = (int32_T)J_minvalue_diff;
  Path_RES_0_size_idx_1 = (int32_T)J_minvalue_diff;
  xy_ends_POS_size_idx_0 = (int32_T)J_minvalue_diff;
  Path_RES_1_size_idx_0 = (int32_T)J_minvalue_diff;
  J_minvalue_diff_0 = (int32_T)J_minvalue_diff;
  J_minvalue_diff_1 = (int32_T)J_minvalue_diff;
  J_minvalue_diff_2 = (int32_T)J_minvalue_diff;
  for (i = 0; i < 11; i++) {
    // Update for UnitDelay: '<S1>/Unit Delay6'
    rtDW.UnitDelay6_DSTATE[i] = ((((rtb_XP[(case_0 - 1) * 6 + 1] * a[i] +
      rtb_XP[(break_count - 1) * 6]) + rtb_XP[(Static_PathCycle - 1) * 6 + 2] *
      X1[i]) + rtb_XP[(loop_ub - 1) * 6 + 3] * b_Path_dis_data[i]) + rtb_XP
      [(end_ind_0 - 1) * 6 + 4] * K1[i]) + rtb_XP[(c - 1) * 6 + 5] * K_11[i];
    rtDW.UnitDelay6_DSTATE[i + 11] = ((((rtb_YP[(xy_ends_POS_size_idx_0 - 1) * 6
      + 1] * a[i] + rtb_YP[(Path_RES_0_size_idx_1 - 1) * 6]) + rtb_YP
      [(Path_RES_1_size_idx_0 - 1) * 6 + 2] * X2[i]) + rtb_YP[(J_minvalue_diff_0
      - 1) * 6 + 3] * Y2[i]) + rtb_YP[(J_minvalue_diff_1 - 1) * 6 + 4] * K2[i])
      + rtb_YP[(J_minvalue_diff_2 - 1) * 6 + 5] * K_12[i];
  }

  // Update for UnitDelay: '<S1>/Unit Delay7' incorporates:
  //   MATLAB Function: '<S1>/Fianl_Path_Decision'

  rtDW.UnitDelay7_DSTATE = J_minvalue_diff;

  // Update for UnitDelay: '<S1>/Unit Delay11' incorporates:
  //   MATLAB Function: '<S1>/Fianl_Path_Decision'

  rtDW.UnitDelay11_DSTATE = ang_1;

  // Update for UnitDelay: '<S1>/Unit Delay13'
  rtDW.UnitDelay13_DSTATE = Length_1;
}

// Model initialize function
void TPModelClass::initialize()
{
  // Registration code

  // initialize non-finites
  rt_InitInfAndNaN(sizeof(real_T));

  // InitializeConditions for UnitDelay: '<S1>/Unit Delay16'
  rtDW.UnitDelay16_DSTATE = 3.0;

  // InitializeConditions for UnitDelay: '<S1>/Unit Delay7'
  rtDW.UnitDelay7_DSTATE = 7.0;
}

// Constructor
TPModelClass::TPModelClass()
{
}

// Destructor
TPModelClass::~TPModelClass()
{
  // Currently there is no destructor body generated.
}

// Real-Time Model get method
RT_MODEL * TPModelClass::getRTM()
{
  return (&rtM);
}

//
// File trailer for generated code.
//
// [EOF]
//
