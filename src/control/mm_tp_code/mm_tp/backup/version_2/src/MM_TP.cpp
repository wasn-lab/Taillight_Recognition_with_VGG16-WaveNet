//
// File: MM_TP.cpp
//
// Code generated for Simulink model 'MM_TP'.
//
// Model version                  : 1.76
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Wed Sep 11 15:25:57 2019
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
// Code generation objectives:
//    1. Execution efficiency
//    2. RAM efficiency
// Validation result: Not run
//
#include "MM_TP.h"
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

// Function for MATLAB Function: '<S3>/SLAM_UKF'
real_T MM_DPP_1ModelClass::sum(const real_T x[10])
{
  real_T y;
  int32_T k;
  y = x[0];
  for (k = 0; k < 9; k++) {
    y += x[k + 1];
  }

  return y;
}

// Function for MATLAB Function: '<S3>/SLAM_UKF'
void MM_DPP_1ModelClass::invNxN(const real_T x[25], real_T y[25])
{
  int8_T p[5];
  real_T A[25];
  int8_T ipiv[5];
  int32_T b_j;
  int32_T ix;
  real_T smax;
  real_T s;
  int32_T b_k;
  int32_T iy;
  int32_T c_ix;
  int32_T d;
  int32_T ijA;
  int32_T pipk;
  for (b_k = 0; b_k < 25; b_k++) {
    y[b_k] = 0.0;
    A[b_k] = x[b_k];
  }

  for (b_k = 0; b_k < 5; b_k++) {
    ipiv[b_k] = (int8_T)(1 + b_k);
  }

  for (b_j = 0; b_j < 4; b_j++) {
    pipk = b_j * 6;
    iy = 0;
    ix = pipk;
    smax = std::abs(A[pipk]);
    for (b_k = 2; b_k <= 5 - b_j; b_k++) {
      ix++;
      s = std::abs(A[ix]);
      if (s > smax) {
        iy = b_k - 1;
        smax = s;
      }
    }

    if (A[pipk + iy] != 0.0) {
      if (iy != 0) {
        iy += b_j;
        ipiv[b_j] = (int8_T)(iy + 1);
        ix = b_j;
        for (b_k = 0; b_k < 5; b_k++) {
          smax = A[ix];
          A[ix] = A[iy];
          A[iy] = smax;
          ix += 5;
          iy += 5;
        }
      }

      iy = (pipk - b_j) + 5;
      for (ix = pipk + 1; ix < iy; ix++) {
        A[ix] /= A[pipk];
      }
    }

    iy = pipk;
    ix = pipk + 5;
    for (b_k = 1; b_k <= 4 - b_j; b_k++) {
      smax = A[ix];
      if (A[ix] != 0.0) {
        c_ix = pipk + 1;
        d = (iy - b_j) + 10;
        for (ijA = 6 + iy; ijA < d; ijA++) {
          A[ijA] += A[c_ix] * -smax;
          c_ix++;
        }
      }

      ix += 5;
      iy += 5;
    }
  }

  for (b_k = 0; b_k < 5; b_k++) {
    p[b_k] = (int8_T)(1 + b_k);
  }

  if (ipiv[0] > 1) {
    b_k = ipiv[0] - 1;
    pipk = p[b_k];
    p[b_k] = p[0];
    p[0] = (int8_T)pipk;
  }

  if (ipiv[1] > 2) {
    b_k = ipiv[1] - 1;
    pipk = p[b_k];
    p[b_k] = p[1];
    p[1] = (int8_T)pipk;
  }

  if (ipiv[2] > 3) {
    b_k = ipiv[2] - 1;
    pipk = p[b_k];
    p[b_k] = p[2];
    p[2] = (int8_T)pipk;
  }

  if (ipiv[3] > 4) {
    b_k = ipiv[3] - 1;
    pipk = p[b_k];
    p[b_k] = p[3];
    p[3] = (int8_T)pipk;
  }

  for (b_j = 0; b_j < 5; b_j++) {
    b_k = p[b_j] - 1;
    y[b_j + 5 * b_k] = 1.0;
    for (iy = b_j; iy + 1 < 6; iy++) {
      if (y[5 * b_k + iy] != 0.0) {
        for (ix = iy + 1; ix + 1 < 6; ix++) {
          y[ix + 5 * b_k] -= y[5 * b_k + iy] * A[5 * iy + ix];
        }
      }
    }
  }

  for (b_j = 0; b_j < 5; b_j++) {
    pipk = 5 * b_j;
    for (iy = 4; iy >= 0; iy--) {
      ix = 5 * iy;
      b_k = iy + pipk;
      if (y[b_k] != 0.0) {
        y[b_k] = y[iy + pipk] / A[iy + ix];
        for (b_k = 0; b_k < iy; b_k++) {
          c_ix = b_k + pipk;
          y[c_ix] -= y[iy + pipk] * A[b_k + ix];
        }
      }
    }
  }
}

// Function for MATLAB Function: '<S3>/SLAM_UKF_MM'
void MM_DPP_1ModelClass::merge(int32_T idx[188], real_T x[188], int32_T offset,
  int32_T np, int32_T nq, int32_T iwork[188], real_T xwork[188])
{
  int32_T n;
  int32_T q;
  int32_T iout;
  int32_T n_tmp;
  int32_T exitg1;
  if (nq != 0) {
    n_tmp = np + nq;
    for (q = 0; q < n_tmp; q++) {
      iout = offset + q;
      iwork[q] = idx[iout];
      xwork[q] = x[iout];
    }

    n = 0;
    q = np;
    iout = offset - 1;
    do {
      exitg1 = 0;
      iout++;
      if (xwork[n] <= xwork[q]) {
        idx[iout] = iwork[n];
        x[iout] = xwork[n];
        if (n + 1 < np) {
          n++;
        } else {
          exitg1 = 1;
        }
      } else {
        idx[iout] = iwork[q];
        x[iout] = xwork[q];
        if (q + 1 < n_tmp) {
          q++;
        } else {
          q = (iout - n) + 1;
          while (n + 1 <= np) {
            iout = q + n;
            idx[iout] = iwork[n];
            x[iout] = xwork[n];
            n++;
          }

          exitg1 = 1;
        }
      }
    } while (exitg1 == 0);
  }
}

// Function for MATLAB Function: '<S3>/SLAM_UKF_MM'
void MM_DPP_1ModelClass::sort(real_T x[188], int32_T idx[188])
{
  int32_T nNaNs;
  real_T xwork[188];
  int32_T iwork[188];
  real_T x4[4];
  uint8_T idx4[4];
  int32_T ib;
  int32_T m;
  int8_T perm[4];
  int32_T i1;
  int32_T i2;
  int32_T i3;
  int32_T i4;
  x4[0] = 0.0;
  idx4[0] = 0U;
  x4[1] = 0.0;
  idx4[1] = 0U;
  x4[2] = 0.0;
  idx4[2] = 0U;
  x4[3] = 0.0;
  idx4[3] = 0U;
  memset(&idx[0], 0, 188U * sizeof(int32_T));
  memset(&xwork[0], 0, 188U * sizeof(real_T));
  nNaNs = 0;
  ib = 0;
  for (m = 0; m < 188; m++) {
    if (rtIsNaN(x[m])) {
      idx[187 - nNaNs] = m + 1;
      xwork[187 - nNaNs] = x[m];
      nNaNs++;
    } else {
      ib++;
      idx4[ib - 1] = (uint8_T)(m + 1);
      x4[ib - 1] = x[m];
      if (ib == 4) {
        ib = m - nNaNs;
        if (x4[0] <= x4[1]) {
          i1 = 1;
          i2 = 2;
        } else {
          i1 = 2;
          i2 = 1;
        }

        if (x4[2] <= x4[3]) {
          i3 = 3;
          i4 = 4;
        } else {
          i3 = 4;
          i4 = 3;
        }

        if (x4[i1 - 1] <= x4[i3 - 1]) {
          if (x4[i2 - 1] <= x4[i3 - 1]) {
            perm[0] = (int8_T)i1;
            perm[1] = (int8_T)i2;
            perm[2] = (int8_T)i3;
            perm[3] = (int8_T)i4;
          } else if (x4[i2 - 1] <= x4[i4 - 1]) {
            perm[0] = (int8_T)i1;
            perm[1] = (int8_T)i3;
            perm[2] = (int8_T)i2;
            perm[3] = (int8_T)i4;
          } else {
            perm[0] = (int8_T)i1;
            perm[1] = (int8_T)i3;
            perm[2] = (int8_T)i4;
            perm[3] = (int8_T)i2;
          }
        } else if (x4[i1 - 1] <= x4[i4 - 1]) {
          if (x4[i2 - 1] <= x4[i4 - 1]) {
            perm[0] = (int8_T)i3;
            perm[1] = (int8_T)i1;
            perm[2] = (int8_T)i2;
            perm[3] = (int8_T)i4;
          } else {
            perm[0] = (int8_T)i3;
            perm[1] = (int8_T)i1;
            perm[2] = (int8_T)i4;
            perm[3] = (int8_T)i2;
          }
        } else {
          perm[0] = (int8_T)i3;
          perm[1] = (int8_T)i4;
          perm[2] = (int8_T)i1;
          perm[3] = (int8_T)i2;
        }

        i1 = perm[0] - 1;
        idx[ib - 3] = idx4[i1];
        i2 = perm[1] - 1;
        idx[ib - 2] = idx4[i2];
        i3 = perm[2] - 1;
        idx[ib - 1] = idx4[i3];
        i4 = perm[3] - 1;
        idx[ib] = idx4[i4];
        x[ib - 3] = x4[i1];
        x[ib - 2] = x4[i2];
        x[ib - 1] = x4[i3];
        x[ib] = x4[i4];
        ib = 0;
      }
    }
  }

  if (ib > 0) {
    perm[1] = 0;
    perm[2] = 0;
    perm[3] = 0;
    switch (ib) {
     case 1:
      perm[0] = 1;
      break;

     case 2:
      if (x4[0] <= x4[1]) {
        perm[0] = 1;
        perm[1] = 2;
      } else {
        perm[0] = 2;
        perm[1] = 1;
      }
      break;

     default:
      if (x4[0] <= x4[1]) {
        if (x4[1] <= x4[2]) {
          perm[0] = 1;
          perm[1] = 2;
          perm[2] = 3;
        } else if (x4[0] <= x4[2]) {
          perm[0] = 1;
          perm[1] = 3;
          perm[2] = 2;
        } else {
          perm[0] = 3;
          perm[1] = 1;
          perm[2] = 2;
        }
      } else if (x4[0] <= x4[2]) {
        perm[0] = 2;
        perm[1] = 1;
        perm[2] = 3;
      } else if (x4[1] <= x4[2]) {
        perm[0] = 2;
        perm[1] = 3;
        perm[2] = 1;
      } else {
        perm[0] = 3;
        perm[1] = 2;
        perm[2] = 1;
      }
      break;
    }

    for (m = 188; m - 187 <= ib; m++) {
      i1 = perm[m - 188] - 1;
      i2 = (m - nNaNs) - ib;
      idx[i2] = idx4[i1];
      x[i2] = x4[i1];
    }
  }

  m = nNaNs >> 1;
  for (ib = 1; ib <= m; ib++) {
    i2 = (ib - nNaNs) + 187;
    i1 = idx[i2];
    idx[i2] = idx[188 - ib];
    idx[188 - ib] = i1;
    x[i2] = xwork[188 - ib];
    x[188 - ib] = xwork[i2];
  }

  if ((nNaNs & 1U) != 0U) {
    x[(m - nNaNs) + 188] = xwork[(m - nNaNs) + 188];
  }

  if (188 - nNaNs > 1) {
    memset(&iwork[0], 0, 188U * sizeof(int32_T));
    ib = (188 - nNaNs) >> 2;
    m = 4;
    while (ib > 1) {
      if ((ib & 1U) != 0U) {
        ib--;
        i1 = m * ib;
        i2 = 188 - (nNaNs + i1);
        if (i2 > m) {
          merge(idx, x, i1, m, i2 - m, iwork, xwork);
        }
      }

      i1 = m << 1;
      ib >>= 1;
      for (i2 = 1; i2 <= ib; i2++) {
        merge(idx, x, (i2 - 1) * i1, m, m, iwork, xwork);
      }

      m = i1;
    }

    if (188 - nNaNs > m) {
      merge(idx, x, 0, m, 188 - (nNaNs + m), iwork, xwork);
    }
  }
}

// Function for MATLAB Function: '<S3>/SLAM_UKF_MM'
void MM_DPP_1ModelClass::power(const real_T a[188], real_T y[188])
{
  int32_T k;
  for (k = 0; k < 188; k++) {
    y[k] = a[k] * a[k];
  }
}

// Function for MATLAB Function: '<S3>/SLAM_UKF_MM'
void MM_DPP_1ModelClass::rel_dist_xy(const real_T ref_xy[2], const real_T pt_xy
  [376], real_T dist[188])
{
  real_T pt_xy_0[188];
  real_T tmp[188];
  real_T tmp_0[188];
  int32_T k;
  for (k = 0; k < 188; k++) {
    pt_xy_0[k] = pt_xy[k] - ref_xy[0];
  }

  power(pt_xy_0, tmp);
  for (k = 0; k < 188; k++) {
    pt_xy_0[k] = pt_xy[188 + k] - ref_xy[1];
  }

  power(pt_xy_0, tmp_0);
  for (k = 0; k < 188; k++) {
    dist[k] = std::sqrt(tmp[k] + tmp_0[k]);
  }
}

// Function for MATLAB Function: '<S3>/SLAM_UKF_MM'
real_T MM_DPP_1ModelClass::rel_dist_xy_c(const real_T ref_xy[2], const real_T
  pt_xy[2])
{
  real_T a;
  real_T b_a;
  a = pt_xy[0] - ref_xy[0];
  b_a = pt_xy[1] - ref_xy[1];
  return std::sqrt(a * a + b_a * b_a);
}

// Function for MATLAB Function: '<S3>/SLAM_UKF_MM'
void MM_DPP_1ModelClass::MM(real_T heading, const real_T X_pos[2], const real_T
  oi_xy[376], const real_T dist_op[188], const real_T Map_data[4324], real_T
  *seg_id_near, real_T *op_distance, real_T oi_near[2], real_T *note, real_T
  *seg_direction, real_T *head_err, real_T num_lane_direction[4], real_T
  *seg_heading)
{
  real_T op_distance_n;
  real_T C;
  int32_T b_index[376];
  real_T SEG_GPS_HEAD[376];
  real_T dist_ini[188];
  real_T dist_end[188];
  int32_T iidx[188];
  boolean_T x[188];
  uint8_T ii_data[188];
  int32_T idx;
  int32_T b_ii;
  boolean_T ex;
  int32_T d_k;
  real_T c_a;
  real_T d_a;
  real_T oi_xy_0[2];
  boolean_T exitg1;
  memcpy(&dist_ini[0], &dist_op[0], 188U * sizeof(real_T));
  sort(dist_ini, iidx);
  for (b_ii = 0; b_ii < 188; b_ii++) {
    b_index[b_ii] = iidx[b_ii];
    b_index[b_ii + 188] = 0;
    SEG_GPS_HEAD[b_ii] = Map_data[b_ii];
    SEG_GPS_HEAD[188 + b_ii] = Map_data[1316 + b_ii];
  }

  for (idx = 0; idx < 188; idx++) {
    op_distance_n = Map_data[b_index[idx] + 187] - oi_xy[b_index[idx] - 1];
    C = Map_data[b_index[idx] + 375] - oi_xy[b_index[idx] + 187];
    c_a = Map_data[b_index[idx] + 563] - oi_xy[b_index[idx] - 1];
    d_a = Map_data[b_index[idx] + 751] - oi_xy[b_index[idx] + 187];
    if (std::sqrt(op_distance_n * op_distance_n + C * C) <= Map_data[b_index[idx]
        + 1503]) {
      b_index[188 + idx] = (std::sqrt(c_a * c_a + d_a * d_a) <=
                            Map_data[b_index[idx] + 1503]);
    } else {
      b_index[188 + idx] = 0;
    }

    x[idx] = (b_index[188 + idx] == 1);
  }

  idx = 0;
  b_ii = 1;
  exitg1 = false;
  while ((!exitg1) && (b_ii < 189)) {
    if (x[b_ii - 1]) {
      idx++;
      ii_data[idx - 1] = (uint8_T)b_ii;
      if (idx >= 188) {
        exitg1 = true;
      } else {
        b_ii++;
      }
    } else {
      b_ii++;
    }
  }

  if (1 > idx) {
    *note = 1.0;
    rel_dist_xy(X_pos, &Map_data[188], dist_ini);
    rel_dist_xy(X_pos, &Map_data[564], dist_end);
    if (!rtIsNaN(dist_ini[0])) {
      idx = 0;
    } else {
      idx = -1;
      b_ii = 2;
      exitg1 = false;
      while ((!exitg1) && (b_ii < 189)) {
        if (!rtIsNaN(dist_ini[b_ii - 1])) {
          idx = b_ii - 1;
          exitg1 = true;
        } else {
          b_ii++;
        }
      }
    }

    if (idx + 1 == 0) {
      op_distance_n = dist_ini[0];
      idx = 0;
    } else {
      op_distance_n = dist_ini[idx];
      for (b_ii = idx + 1; b_ii + 1 < 189; b_ii++) {
        if (op_distance_n > dist_ini[b_ii]) {
          op_distance_n = dist_ini[b_ii];
          idx = b_ii;
        }
      }
    }

    if (!rtIsNaN(dist_end[0])) {
      b_ii = 0;
    } else {
      b_ii = -1;
      d_k = 2;
      exitg1 = false;
      while ((!exitg1) && (d_k < 189)) {
        if (!rtIsNaN(dist_end[d_k - 1])) {
          b_ii = d_k - 1;
          exitg1 = true;
        } else {
          d_k++;
        }
      }
    }

    if (b_ii + 1 == 0) {
      C = dist_end[0];
      b_ii = 0;
    } else {
      C = dist_end[b_ii];
      for (d_k = b_ii + 1; d_k + 1 < 189; d_k++) {
        if (C > dist_end[d_k]) {
          C = dist_end[d_k];
          b_ii = d_k;
        }
      }
    }

    if (!(op_distance_n <= C)) {
      idx = b_ii;
    }

    *seg_id_near = Map_data[idx];
    *op_distance = dist_op[idx];
    oi_near[0] = oi_xy[idx];
    oi_near[1] = oi_xy[idx + 188];
    if ((idx + 1 > 1) && (idx + 1 < 188)) {
      oi_xy_0[0] = oi_xy[idx];
      oi_xy_0[1] = oi_xy[idx + 188];
      op_distance_n = rel_dist_xy_c(X_pos, oi_xy_0);
      oi_xy_0[0] = Map_data[idx + 188];
      oi_xy_0[1] = Map_data[idx + 376];
      if (op_distance_n < rel_dist_xy_c(X_pos, oi_xy_0)) {
        oi_xy_0[0] = Map_data[idx + 564];
        oi_xy_0[1] = Map_data[idx + 752];
        if (op_distance_n < rel_dist_xy_c(X_pos, oi_xy_0)) {
          *note = 0.0;
        }
      }
    }

    for (b_ii = 0; b_ii < 188; b_ii++) {
      x[b_ii] = (SEG_GPS_HEAD[b_ii] == Map_data[idx]);
    }

    idx = -1;
    ex = x[0];
    for (b_ii = 0; b_ii < 187; b_ii++) {
      if ((int32_T)ex < (int32_T)x[b_ii + 1]) {
        ex = x[b_ii + 1];
        idx = b_ii;
      }
    }

    *head_err = std::abs(heading - SEG_GPS_HEAD[idx + 189]);
    if (*head_err <= 0.78539816339744828) {
      *seg_direction = 1.0;
    } else if ((*head_err >= 0.78539816339744828) && (*head_err <= 90.0)) {
      *seg_direction = 0.0;
    } else {
      *seg_direction = 2.0;
    }
  } else {
    *note = 0.0;
    idx = b_index[ii_data[0] - 1] - 1;
    *seg_id_near = Map_data[idx];
    *op_distance = dist_op[idx];
    oi_near[0] = oi_xy[idx];
    oi_near[1] = oi_xy[b_index[ii_data[0] - 1] + 187];
    for (b_ii = 0; b_ii < 188; b_ii++) {
      x[b_ii] = (Map_data[b_index[ii_data[0] - 1] - 1] == SEG_GPS_HEAD[b_ii]);
    }

    idx = -1;
    ex = x[0];
    for (b_ii = 0; b_ii < 187; b_ii++) {
      if ((int32_T)ex < (int32_T)x[b_ii + 1]) {
        ex = x[b_ii + 1];
        idx = b_ii;
      }
    }

    *head_err = std::abs(heading - SEG_GPS_HEAD[idx + 189]);
    if (*head_err <= 0.78539816339744828) {
      *seg_direction = 1.0;
    } else if ((*head_err >= 0.78539816339744828) && (*head_err <= 90.0)) {
      *seg_direction = 0.0;
    } else {
      *seg_direction = 2.0;
    }

    rel_dist_xy(X_pos, &Map_data[188], dist_ini);
    rel_dist_xy(X_pos, &Map_data[564], dist_end);
    if (!rtIsNaN(dist_ini[0])) {
      idx = 0;
    } else {
      idx = -1;
      b_ii = 2;
      exitg1 = false;
      while ((!exitg1) && (b_ii < 189)) {
        if (!rtIsNaN(dist_ini[b_ii - 1])) {
          idx = b_ii - 1;
          exitg1 = true;
        } else {
          b_ii++;
        }
      }
    }

    if (idx + 1 == 0) {
      op_distance_n = dist_ini[0];
      idx = 0;
    } else {
      op_distance_n = dist_ini[idx];
      for (b_ii = idx + 1; b_ii + 1 < 189; b_ii++) {
        if (op_distance_n > dist_ini[b_ii]) {
          op_distance_n = dist_ini[b_ii];
          idx = b_ii;
        }
      }
    }

    if (!rtIsNaN(dist_end[0])) {
      b_ii = 0;
    } else {
      b_ii = -1;
      d_k = 2;
      exitg1 = false;
      while ((!exitg1) && (d_k < 189)) {
        if (!rtIsNaN(dist_end[d_k - 1])) {
          b_ii = d_k - 1;
          exitg1 = true;
        } else {
          d_k++;
        }
      }
    }

    if (b_ii + 1 == 0) {
      C = dist_end[0];
      b_ii = 0;
    } else {
      C = dist_end[b_ii];
      for (d_k = b_ii + 1; d_k + 1 < 189; d_k++) {
        if (C > dist_end[d_k]) {
          C = dist_end[d_k];
          b_ii = d_k;
        }
      }
    }

    if ((op_distance_n < C) || rtIsNaN(C)) {
      c_a = op_distance_n;
    } else {
      c_a = C;
    }

    if (c_a < dist_op[b_index[ii_data[0] - 1] - 1]) {
      *note = 2.0;
      if (!(op_distance_n <= C)) {
        idx = b_ii;
      }

      for (b_ii = 0; b_ii < 188; b_ii++) {
        x[b_ii] = (Map_data[b_index[ii_data[0] - 1] - 1] == SEG_GPS_HEAD[b_ii]);
      }

      b_ii = -1;
      ex = x[0];
      for (d_k = 0; d_k < 187; d_k++) {
        if ((int32_T)ex < (int32_T)x[d_k + 1]) {
          ex = x[d_k + 1];
          b_ii = d_k;
        }
      }

      *head_err = std::abs(heading - SEG_GPS_HEAD[b_ii + 189]);
      if (*head_err <= 0.78539816339744828) {
        *seg_direction = 1.0;
      } else if ((*head_err >= 0.78539816339744828) && (*head_err <= 90.0)) {
        *seg_direction = 0.0;
      } else {
        *seg_direction = 2.0;
      }

      *seg_id_near = Map_data[idx];
      *op_distance = dist_op[idx];
      oi_near[0] = oi_xy[idx];
      oi_near[1] = oi_xy[idx + 188];
    }
  }

  for (b_ii = 0; b_ii < 188; b_ii++) {
    dist_ini[b_ii] = Map_data[1316 + b_ii] * 3.1415926535897931 / 180.0;
    x[b_ii] = (Map_data[b_ii] == *seg_id_near);
  }

  idx = 0;
  ex = x[0];
  for (b_ii = 0; b_ii < 187; b_ii++) {
    if ((int32_T)ex < (int32_T)x[b_ii + 1]) {
      ex = x[b_ii + 1];
      idx = b_ii + 1;
    }
  }

  op_distance_n = oi_near[1] - Map_data[940 + idx] * oi_near[0];
  if (Map_data[940 + idx] < 0.0) {
    C = (-Map_data[940 + idx] * X_pos[0] - op_distance_n) + X_pos[1];
    if (dist_ini[idx] > 4.71238898038469) {
      if (!(dist_ini[idx] < 6.2831853071795862)) {
        C = -C;
      }
    } else {
      C = -C;
    }
  } else if (Map_data[940 + idx] == 0.0) {
    if (oi_near[1] < X_pos[1]) {
      C = -1.0;
    } else {
      C = 1.0;
    }
  } else {
    C = (Map_data[940 + idx] * X_pos[0] + op_distance_n) - X_pos[1];
    if (dist_ini[idx] > 3.1415926535897931) {
      if (!(dist_ini[idx] < 4.71238898038469)) {
        C = -C;
      }
    } else {
      C = -C;
    }
  }

  num_lane_direction[0] = Map_data[940 + idx];
  num_lane_direction[1] = op_distance_n;
  num_lane_direction[2] = C;
  if (C < 0.0) {
    num_lane_direction[3] = 1.0;
  } else if (C == 0.0) {
    num_lane_direction[3] = 1.0;
  } else {
    num_lane_direction[3] = -1.0;
  }

  *seg_heading = dist_ini[idx];
}

// Function for MATLAB Function: '<S2>/MM'
void MM_DPP_1ModelClass::merge_e(int32_T idx_data[], real_T x_data[], int32_T
  offset, int32_T np, int32_T nq, int32_T iwork_data[], real_T xwork_data[])
{
  int32_T n;
  int32_T q;
  int32_T iout;
  int32_T n_tmp;
  int32_T exitg1;
  if (nq != 0) {
    n_tmp = np + nq;
    for (q = 0; q < n_tmp; q++) {
      iout = offset + q;
      iwork_data[q] = idx_data[iout];
      xwork_data[q] = x_data[iout];
    }

    n = 0;
    q = np;
    iout = offset - 1;
    do {
      exitg1 = 0;
      iout++;
      if (xwork_data[n] <= xwork_data[q]) {
        idx_data[iout] = iwork_data[n];
        x_data[iout] = xwork_data[n];
        if (n + 1 < np) {
          n++;
        } else {
          exitg1 = 1;
        }
      } else {
        idx_data[iout] = iwork_data[q];
        x_data[iout] = xwork_data[q];
        if (q + 1 < n_tmp) {
          q++;
        } else {
          q = (iout - n) + 1;
          while (n + 1 <= np) {
            iout = q + n;
            idx_data[iout] = iwork_data[n];
            x_data[iout] = xwork_data[n];
            n++;
          }

          exitg1 = 1;
        }
      }
    } while (exitg1 == 0);
  }
}

// Function for MATLAB Function: '<S2>/MM'
void MM_DPP_1ModelClass::merge_block(int32_T idx_data[], real_T x_data[],
  int32_T n, int32_T iwork_data[], real_T xwork_data[])
{
  int32_T bLen;
  int32_T tailOffset;
  int32_T nTail;
  int32_T nPairs;
  nPairs = n >> 2;
  bLen = 4;
  while (nPairs > 1) {
    if ((nPairs & 1U) != 0U) {
      nPairs--;
      tailOffset = bLen * nPairs;
      nTail = n - tailOffset;
      if (nTail > bLen) {
        merge_e(idx_data, x_data, tailOffset, bLen, nTail - bLen, iwork_data,
                xwork_data);
      }
    }

    tailOffset = bLen << 1;
    nPairs >>= 1;
    for (nTail = 1; nTail <= nPairs; nTail++) {
      merge_e(idx_data, x_data, (nTail - 1) * tailOffset, bLen, bLen, iwork_data,
              xwork_data);
    }

    bLen = tailOffset;
  }

  if (n > bLen) {
    merge_e(idx_data, x_data, 0, bLen, n - bLen, iwork_data, xwork_data);
  }
}

// Function for MATLAB Function: '<S2>/MM'
void MM_DPP_1ModelClass::sortIdx(real_T x_data[], int32_T *x_size, int32_T
  idx_data[], int32_T *idx_size)
{
  real_T c_x_data[188];
  real_T xwork_data[188];
  int32_T iwork_data[188];
  int32_T n;
  real_T x4[4];
  uint8_T idx4[4];
  int32_T ib;
  int32_T wOffset;
  int32_T itmp;
  int8_T perm[4];
  int32_T i1;
  int32_T i3;
  int32_T i4;
  int32_T c_x_size;
  uint8_T b_x_idx_0;
  uint8_T b_idx_0;
  b_x_idx_0 = (uint8_T)*x_size;
  b_idx_0 = (uint8_T)*x_size;
  *idx_size = b_x_idx_0;
  if (0 <= b_x_idx_0 - 1) {
    memset(&idx_data[0], 0, b_x_idx_0 * sizeof(int32_T));
  }

  if (*x_size != 0) {
    c_x_size = *x_size;
    if (0 <= *x_size - 1) {
      memcpy(&c_x_data[0], &x_data[0], *x_size * sizeof(real_T));
    }

    *idx_size = b_idx_0;
    if (0 <= b_idx_0 - 1) {
      memset(&idx_data[0], 0, b_idx_0 * sizeof(int32_T));
    }

    x4[0] = 0.0;
    idx4[0] = 0U;
    x4[1] = 0.0;
    idx4[1] = 0U;
    x4[2] = 0.0;
    idx4[2] = 0U;
    x4[3] = 0.0;
    idx4[3] = 0U;
    b_idx_0 = (uint8_T)*x_size;
    if (0 <= b_idx_0 - 1) {
      memset(&xwork_data[0], 0, b_idx_0 * sizeof(real_T));
    }

    n = 1;
    ib = 0;
    for (wOffset = 0; wOffset < *x_size; wOffset++) {
      if (rtIsNaN(c_x_data[wOffset])) {
        i3 = *x_size - n;
        idx_data[i3] = wOffset + 1;
        xwork_data[i3] = c_x_data[wOffset];
        n++;
      } else {
        ib++;
        idx4[ib - 1] = (uint8_T)(wOffset + 1);
        x4[ib - 1] = c_x_data[wOffset];
        if (ib == 4) {
          ib = wOffset - n;
          if (x4[0] <= x4[1]) {
            i1 = 1;
            itmp = 2;
          } else {
            i1 = 2;
            itmp = 1;
          }

          if (x4[2] <= x4[3]) {
            i3 = 3;
            i4 = 4;
          } else {
            i3 = 4;
            i4 = 3;
          }

          if (x4[i1 - 1] <= x4[i3 - 1]) {
            if (x4[itmp - 1] <= x4[i3 - 1]) {
              perm[0] = (int8_T)i1;
              perm[1] = (int8_T)itmp;
              perm[2] = (int8_T)i3;
              perm[3] = (int8_T)i4;
            } else if (x4[itmp - 1] <= x4[i4 - 1]) {
              perm[0] = (int8_T)i1;
              perm[1] = (int8_T)i3;
              perm[2] = (int8_T)itmp;
              perm[3] = (int8_T)i4;
            } else {
              perm[0] = (int8_T)i1;
              perm[1] = (int8_T)i3;
              perm[2] = (int8_T)i4;
              perm[3] = (int8_T)itmp;
            }
          } else if (x4[i1 - 1] <= x4[i4 - 1]) {
            if (x4[itmp - 1] <= x4[i4 - 1]) {
              perm[0] = (int8_T)i3;
              perm[1] = (int8_T)i1;
              perm[2] = (int8_T)itmp;
              perm[3] = (int8_T)i4;
            } else {
              perm[0] = (int8_T)i3;
              perm[1] = (int8_T)i1;
              perm[2] = (int8_T)i4;
              perm[3] = (int8_T)itmp;
            }
          } else {
            perm[0] = (int8_T)i3;
            perm[1] = (int8_T)i4;
            perm[2] = (int8_T)i1;
            perm[3] = (int8_T)itmp;
          }

          i3 = perm[0] - 1;
          idx_data[ib - 2] = idx4[i3];
          itmp = perm[1] - 1;
          idx_data[ib - 1] = idx4[itmp];
          i1 = perm[2] - 1;
          idx_data[ib] = idx4[i1];
          i4 = perm[3] - 1;
          idx_data[ib + 1] = idx4[i4];
          c_x_data[ib - 2] = x4[i3];
          c_x_data[ib - 1] = x4[itmp];
          c_x_data[ib] = x4[i1];
          c_x_data[ib + 1] = x4[i4];
          ib = 0;
        }
      }
    }

    wOffset = *x_size - n;
    if (ib > 0) {
      perm[1] = 0;
      perm[2] = 0;
      perm[3] = 0;
      switch (ib) {
       case 1:
        perm[0] = 1;
        break;

       case 2:
        if (x4[0] <= x4[1]) {
          perm[0] = 1;
          perm[1] = 2;
        } else {
          perm[0] = 2;
          perm[1] = 1;
        }
        break;

       default:
        if (x4[0] <= x4[1]) {
          if (x4[1] <= x4[2]) {
            perm[0] = 1;
            perm[1] = 2;
            perm[2] = 3;
          } else if (x4[0] <= x4[2]) {
            perm[0] = 1;
            perm[1] = 3;
            perm[2] = 2;
          } else {
            perm[0] = 3;
            perm[1] = 1;
            perm[2] = 2;
          }
        } else if (x4[0] <= x4[2]) {
          perm[0] = 2;
          perm[1] = 1;
          perm[2] = 3;
        } else if (x4[1] <= x4[2]) {
          perm[0] = 2;
          perm[1] = 3;
          perm[2] = 1;
        } else {
          perm[0] = 3;
          perm[1] = 2;
          perm[2] = 1;
        }
        break;
      }

      for (i1 = 1; i1 <= ib; i1++) {
        i3 = perm[i1 - 1] - 1;
        itmp = (wOffset - ib) + i1;
        idx_data[itmp] = idx4[i3];
        c_x_data[itmp] = x4[i3];
      }
    }

    ib = ((n - 1) >> 1) + 1;
    for (i1 = 1; i1 < ib; i1++) {
      i4 = wOffset + i1;
      itmp = idx_data[i4];
      i3 = *x_size - i1;
      idx_data[i4] = idx_data[i3];
      idx_data[i3] = itmp;
      c_x_data[i4] = xwork_data[i3];
      c_x_data[i3] = xwork_data[i4];
    }

    if (((n - 1) & 1U) != 0U) {
      c_x_data[wOffset + ib] = xwork_data[wOffset + ib];
    }

    n = wOffset + 1;
    if (n > 1) {
      if (0 <= b_x_idx_0 - 1) {
        memset(&iwork_data[0], 0, b_x_idx_0 * sizeof(int32_T));
      }

      merge_block(idx_data, c_x_data, n, iwork_data, xwork_data);
    }

    if (0 <= c_x_size - 1) {
      memcpy(&x_data[0], &c_x_data[0], c_x_size * sizeof(real_T));
    }
  }
}

// Function for MATLAB Function: '<S2>/MM'
void MM_DPP_1ModelClass::sort_g(real_T x_data[], int32_T *x_size, int32_T
  idx_data[], int32_T *idx_size)
{
  int32_T dim;
  real_T vwork_data[188];
  int32_T vstride;
  int32_T iidx_data[188];
  int32_T b;
  int32_T c_k;
  int32_T vwork_size;
  int32_T tmp;
  dim = 2;
  if (*x_size != 1) {
    dim = 1;
  }

  if (dim <= 1) {
    b = *x_size;
  } else {
    b = 1;
  }

  vwork_size = (uint8_T)b;
  *idx_size = (uint8_T)*x_size;
  vstride = 1;
  c_k = 1;
  while (c_k <= dim - 1) {
    vstride *= *x_size;
    c_k = 2;
  }

  for (dim = 0; dim < vstride; dim++) {
    for (c_k = 0; c_k < b; c_k++) {
      vwork_data[c_k] = x_data[c_k * vstride + dim];
    }

    sortIdx(vwork_data, &vwork_size, iidx_data, &c_k);
    for (c_k = 0; c_k < b; c_k++) {
      tmp = dim + c_k * vstride;
      x_data[tmp] = vwork_data[c_k];
      idx_data[tmp] = iidx_data[c_k];
    }
  }
}

// Function for MATLAB Function: '<S2>/MM'
void MM_DPP_1ModelClass::power_l(const real_T a_data[], const int32_T *a_size,
  real_T y_data[], int32_T *y_size)
{
  real_T z1_data[188];
  int32_T loop_ub;
  uint8_T a_idx_0;
  a_idx_0 = (uint8_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&z1_data[0], &y_data[0], a_idx_0 * sizeof(real_T));
  }

  for (loop_ub = 0; loop_ub < a_idx_0; loop_ub++) {
    z1_data[loop_ub] = a_data[loop_ub] * a_data[loop_ub];
  }

  *y_size = (uint8_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&y_data[0], &z1_data[0], a_idx_0 * sizeof(real_T));
  }
}

// Function for MATLAB Function: '<S2>/MM'
void MM_DPP_1ModelClass::power_lz(const real_T a_data[], const int32_T *a_size,
  real_T y_data[], int32_T *y_size)
{
  real_T z1_data[188];
  int32_T loop_ub;
  uint8_T a_idx_0;
  a_idx_0 = (uint8_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&z1_data[0], &y_data[0], a_idx_0 * sizeof(real_T));
  }

  for (loop_ub = 0; loop_ub < a_idx_0; loop_ub++) {
    z1_data[loop_ub] = std::sqrt(a_data[loop_ub]);
  }

  *y_size = (uint8_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&y_data[0], &z1_data[0], a_idx_0 * sizeof(real_T));
  }
}

// Function for MATLAB Function: '<S2>/MM'
void MM_DPP_1ModelClass::rel_dist_xy_d(const real_T ref_xy[2], const real_T
  pt_xy_data[], const int32_T pt_xy_size[2], real_T dist_data[], int32_T
  *dist_size)
{
  real_T pt_xy_data_0[188];
  real_T tmp_data[188];
  real_T tmp_data_0[188];
  int32_T loop_ub;
  int32_T i;
  int32_T pt_xy_size_0;
  int32_T tmp_size;
  int32_T pt_xy_size_1;
  loop_ub = pt_xy_size[0];
  pt_xy_size_0 = pt_xy_size[0];
  for (i = 0; i < loop_ub; i++) {
    pt_xy_data_0[i] = pt_xy_data[i] - ref_xy[0];
  }

  power_l(pt_xy_data_0, &pt_xy_size_0, tmp_data, &tmp_size);
  loop_ub = pt_xy_size[0];
  pt_xy_size_1 = pt_xy_size[0];
  for (i = 0; i < loop_ub; i++) {
    pt_xy_data_0[i] = pt_xy_data[i + pt_xy_size[0]] - ref_xy[1];
  }

  power_l(pt_xy_data_0, &pt_xy_size_1, tmp_data_0, &pt_xy_size_0);
  for (i = 0; i < tmp_size; i++) {
    pt_xy_data_0[i] = tmp_data[i] + tmp_data_0[i];
  }

  power_lz(pt_xy_data_0, &tmp_size, dist_data, dist_size);
}

// Function for MATLAB Function: '<S2>/MM'
void MM_DPP_1ModelClass::MM_f(real_T heading, const real_T X_pos[2], const
  real_T oi_xy_data[], const int32_T oi_xy_size[2], const real_T dist_op_data[],
  const int32_T *dist_op_size, const real_T Map_data_data[], const int32_T
  Map_data_size[2], real_T *seg_id_near, real_T *op_distance, real_T oi_near[2],
  real_T *note, real_T *seg_direction, real_T *head_err, real_T
  num_lane_direction[4], real_T *seg_heading)
{
  real_T xy_ini_data[376];
  real_T xy_end_data[376];
  real_T seg_id_data[188];
  real_T ind_temp_data[188];
  real_T op_distance_n;
  real_T C;
  int32_T b_index_data[376];
  real_T SEG_GPS_HEAD_data[376];
  real_T dist_ini_data[188];
  real_T dist_end_data[188];
  boolean_T x_data[188];
  int32_T ii_data[188];
  int32_T nx;
  int32_T idx;
  int32_T g_idx;
  boolean_T i_ex;
  real_T d_a;
  real_T oi_xy[2];
  int32_T loop_ub;
  int32_T xy_ini_size[2];
  int32_T xy_end_size[2];
  int32_T ii_size;
  real_T op_distance_n_0;
  boolean_T exitg1;
  g_idx = *dist_op_size;
  if (0 <= *dist_op_size - 1) {
    memcpy(&ind_temp_data[0], &dist_op_data[0], *dist_op_size * sizeof(real_T));
  }

  sort_g(ind_temp_data, &g_idx, ii_data, &ii_size);
  for (g_idx = 0; g_idx < ii_size; g_idx++) {
    ind_temp_data[g_idx] = ii_data[g_idx];
  }

  loop_ub = ii_size - 1;
  for (g_idx = 0; g_idx < ii_size; g_idx++) {
    b_index_data[g_idx] = (int32_T)ind_temp_data[g_idx];
  }

  if (0 <= loop_ub) {
    memset(&b_index_data[ii_size], 0, (((loop_ub + ii_size) - ii_size) + 1) *
           sizeof(int32_T));
  }

  loop_ub = Map_data_size[0];
  xy_ini_size[0] = Map_data_size[0];
  xy_ini_size[1] = 2;
  nx = Map_data_size[0];
  xy_end_size[0] = Map_data_size[0];
  xy_end_size[1] = 2;
  for (g_idx = 0; g_idx < loop_ub; g_idx++) {
    xy_ini_data[g_idx] = Map_data_data[g_idx + Map_data_size[0]];
  }

  for (g_idx = 0; g_idx < nx; g_idx++) {
    xy_end_data[g_idx] = Map_data_data[Map_data_size[0] * 3 + g_idx];
  }

  for (g_idx = 0; g_idx < loop_ub; g_idx++) {
    xy_ini_data[g_idx + loop_ub] = Map_data_data[(Map_data_size[0] << 1) + g_idx];
  }

  for (g_idx = 0; g_idx < nx; g_idx++) {
    xy_end_data[g_idx + nx] = Map_data_data[(Map_data_size[0] << 2) + g_idx];
  }

  loop_ub = Map_data_size[0];
  if (0 <= loop_ub - 1) {
    memcpy(&seg_id_data[0], &Map_data_data[0], loop_ub * sizeof(real_T));
    memcpy(&SEG_GPS_HEAD_data[0], &seg_id_data[0], loop_ub * sizeof(real_T));
  }

  nx = Map_data_size[0] - 1;
  for (g_idx = 0; g_idx <= nx; g_idx++) {
    SEG_GPS_HEAD_data[g_idx + loop_ub] = Map_data_data[Map_data_size[0] * 7 +
      g_idx];
  }

  for (nx = 0; nx < Map_data_size[0]; nx++) {
    op_distance_n = Map_data_data[(b_index_data[nx] + Map_data_size[0]) - 1] -
      oi_xy_data[b_index_data[nx] - 1];
    C = Map_data_data[((Map_data_size[0] << 1) + b_index_data[nx]) - 1] -
      oi_xy_data[(b_index_data[nx] + oi_xy_size[0]) - 1];
    op_distance_n_0 = Map_data_data[(Map_data_size[0] * 3 + b_index_data[nx]) -
      1] - oi_xy_data[b_index_data[nx] - 1];
    d_a = Map_data_data[((Map_data_size[0] << 2) + b_index_data[nx]) - 1] -
      oi_xy_data[(b_index_data[nx] + oi_xy_size[0]) - 1];
    if (std::sqrt(op_distance_n * op_distance_n + C * C) <= Map_data_data
        [((Map_data_size[0] << 3) + b_index_data[nx]) - 1]) {
      b_index_data[nx + ii_size] = (std::sqrt(op_distance_n_0 * op_distance_n_0
        + d_a * d_a) <= Map_data_data[((Map_data_size[0] << 3) + b_index_data[nx])
        - 1]);
    } else {
      b_index_data[nx + ii_size] = 0;
    }
  }

  if (1 > ii_size) {
    nx = 0;
  } else {
    nx = ii_size;
  }

  for (g_idx = 0; g_idx < nx; g_idx++) {
    x_data[g_idx] = (b_index_data[g_idx + ii_size] == 1);
  }

  idx = 0;
  ii_size = nx;
  g_idx = 1;
  exitg1 = false;
  while ((!exitg1) && (g_idx <= nx)) {
    if (x_data[g_idx - 1]) {
      idx++;
      ii_data[idx - 1] = g_idx;
      if (idx >= nx) {
        exitg1 = true;
      } else {
        g_idx++;
      }
    } else {
      g_idx++;
    }
  }

  if (nx == 1) {
    if (idx == 0) {
      ii_size = 0;
    }
  } else if (1 > idx) {
    ii_size = 0;
  } else {
    ii_size = idx;
  }

  for (g_idx = 0; g_idx < ii_size; g_idx++) {
    ind_temp_data[g_idx] = ii_data[g_idx];
  }

  if (ii_size == 0) {
    *note = 1.0;
    rel_dist_xy_d(X_pos, xy_ini_data, xy_ini_size, dist_ini_data, &g_idx);
    rel_dist_xy_d(X_pos, xy_end_data, xy_end_size, dist_end_data, &ii_size);
    if (g_idx <= 2) {
      if (g_idx == 1) {
        op_distance_n = dist_ini_data[0];
        nx = 0;
      } else if ((dist_ini_data[0] > dist_ini_data[1]) || (rtIsNaN
                  (dist_ini_data[0]) && (!rtIsNaN(dist_ini_data[1])))) {
        op_distance_n = dist_ini_data[1];
        nx = 1;
      } else {
        op_distance_n = dist_ini_data[0];
        nx = 0;
      }
    } else {
      if (!rtIsNaN(dist_ini_data[0])) {
        nx = 0;
      } else {
        nx = -1;
        idx = 2;
        exitg1 = false;
        while ((!exitg1) && (idx <= g_idx)) {
          if (!rtIsNaN(dist_ini_data[idx - 1])) {
            nx = idx - 1;
            exitg1 = true;
          } else {
            idx++;
          }
        }
      }

      if (nx + 1 == 0) {
        op_distance_n = dist_ini_data[0];
        nx = 0;
      } else {
        op_distance_n = dist_ini_data[nx];
        for (idx = nx + 1; idx < g_idx; idx++) {
          if (op_distance_n > dist_ini_data[idx]) {
            op_distance_n = dist_ini_data[idx];
            nx = idx;
          }
        }
      }
    }

    if (ii_size <= 2) {
      if (ii_size == 1) {
        C = dist_end_data[0];
        idx = 0;
      } else if ((dist_end_data[0] > dist_end_data[1]) || (rtIsNaN
                  (dist_end_data[0]) && (!rtIsNaN(dist_end_data[1])))) {
        C = dist_end_data[1];
        idx = 1;
      } else {
        C = dist_end_data[0];
        idx = 0;
      }
    } else {
      if (!rtIsNaN(dist_end_data[0])) {
        idx = 0;
      } else {
        idx = -1;
        g_idx = 2;
        exitg1 = false;
        while ((!exitg1) && (g_idx <= ii_size)) {
          if (!rtIsNaN(dist_end_data[g_idx - 1])) {
            idx = g_idx - 1;
            exitg1 = true;
          } else {
            g_idx++;
          }
        }
      }

      if (idx + 1 == 0) {
        C = dist_end_data[0];
        idx = 0;
      } else {
        C = dist_end_data[idx];
        for (g_idx = idx + 1; g_idx < ii_size; g_idx++) {
          if (C > dist_end_data[g_idx]) {
            C = dist_end_data[g_idx];
            idx = g_idx;
          }
        }
      }
    }

    if (!(op_distance_n <= C)) {
      nx = idx;
    }

    *seg_id_near = Map_data_data[nx];
    *op_distance = dist_op_data[nx];
    oi_near[0] = oi_xy_data[nx];
    oi_near[1] = oi_xy_data[nx + oi_xy_size[0]];
    if ((nx + 1 > 1) && (nx + 1 < Map_data_size[0])) {
      oi_xy[0] = oi_xy_data[nx];
      oi_xy[1] = oi_xy_data[nx + oi_xy_size[0]];
      op_distance_n = rel_dist_xy_c(X_pos, oi_xy);
      oi_xy[0] = xy_ini_data[nx];
      g_idx = nx + Map_data_size[0];
      oi_xy[1] = xy_ini_data[g_idx];
      if (op_distance_n < rel_dist_xy_c(X_pos, oi_xy)) {
        oi_xy[0] = xy_end_data[nx];
        oi_xy[1] = xy_end_data[g_idx];
        if (op_distance_n < rel_dist_xy_c(X_pos, oi_xy)) {
          *note = 0.0;
        }
      }
    }

    for (g_idx = 0; g_idx < loop_ub; g_idx++) {
      x_data[g_idx] = (SEG_GPS_HEAD_data[g_idx] == Map_data_data[nx]);
    }

    idx = 0;
    g_idx = 1;
    exitg1 = false;
    while ((!exitg1) && (g_idx <= loop_ub)) {
      if (x_data[g_idx - 1]) {
        idx++;
        ii_data[idx - 1] = g_idx;
        if (idx >= loop_ub) {
          exitg1 = true;
        } else {
          g_idx++;
        }
      } else {
        g_idx++;
      }
    }

    *head_err = std::abs(heading - SEG_GPS_HEAD_data[(ii_data[0] +
      Map_data_size[0]) - 1]);
    if (*head_err <= 0.78539816339744828) {
      *seg_direction = 1.0;
    } else if ((*head_err >= 0.78539816339744828) && (*head_err <= 90.0)) {
      *seg_direction = 0.0;
    } else {
      *seg_direction = 2.0;
    }
  } else {
    *note = 0.0;
    nx = b_index_data[(int32_T)ind_temp_data[0] - 1] - 1;
    *seg_id_near = Map_data_data[nx];
    *op_distance = dist_op_data[nx];
    oi_near[0] = oi_xy_data[nx];
    oi_near[1] = oi_xy_data[(b_index_data[(int32_T)ind_temp_data[0] - 1] +
      oi_xy_size[0]) - 1];
    for (g_idx = 0; g_idx < loop_ub; g_idx++) {
      x_data[g_idx] = (Map_data_data[b_index_data[(int32_T)ind_temp_data[0] - 1]
                       - 1] == SEG_GPS_HEAD_data[g_idx]);
    }

    idx = 0;
    g_idx = 1;
    exitg1 = false;
    while ((!exitg1) && (g_idx <= loop_ub)) {
      if (x_data[g_idx - 1]) {
        idx++;
        ii_data[idx - 1] = g_idx;
        if (idx >= loop_ub) {
          exitg1 = true;
        } else {
          g_idx++;
        }
      } else {
        g_idx++;
      }
    }

    *head_err = std::abs(heading - SEG_GPS_HEAD_data[(ii_data[0] +
      Map_data_size[0]) - 1]);
    if (*head_err <= 0.78539816339744828) {
      *seg_direction = 1.0;
    } else if ((*head_err >= 0.78539816339744828) && (*head_err <= 90.0)) {
      *seg_direction = 0.0;
    } else {
      *seg_direction = 2.0;
    }

    rel_dist_xy_d(X_pos, xy_ini_data, xy_ini_size, dist_ini_data, &g_idx);
    rel_dist_xy_d(X_pos, xy_end_data, xy_end_size, dist_end_data, &ii_size);
    if (g_idx <= 2) {
      if (g_idx == 1) {
        op_distance_n = dist_ini_data[0];
        nx = 0;
      } else if ((dist_ini_data[0] > dist_ini_data[1]) || (rtIsNaN
                  (dist_ini_data[0]) && (!rtIsNaN(dist_ini_data[1])))) {
        op_distance_n = dist_ini_data[1];
        nx = 1;
      } else {
        op_distance_n = dist_ini_data[0];
        nx = 0;
      }
    } else {
      if (!rtIsNaN(dist_ini_data[0])) {
        nx = 0;
      } else {
        nx = -1;
        idx = 2;
        exitg1 = false;
        while ((!exitg1) && (idx <= g_idx)) {
          if (!rtIsNaN(dist_ini_data[idx - 1])) {
            nx = idx - 1;
            exitg1 = true;
          } else {
            idx++;
          }
        }
      }

      if (nx + 1 == 0) {
        op_distance_n = dist_ini_data[0];
        nx = 0;
      } else {
        op_distance_n = dist_ini_data[nx];
        for (idx = nx + 1; idx < g_idx; idx++) {
          if (op_distance_n > dist_ini_data[idx]) {
            op_distance_n = dist_ini_data[idx];
            nx = idx;
          }
        }
      }
    }

    if (ii_size <= 2) {
      if (ii_size == 1) {
        C = dist_end_data[0];
        idx = 0;
      } else if ((dist_end_data[0] > dist_end_data[1]) || (rtIsNaN
                  (dist_end_data[0]) && (!rtIsNaN(dist_end_data[1])))) {
        C = dist_end_data[1];
        idx = 1;
      } else {
        C = dist_end_data[0];
        idx = 0;
      }
    } else {
      if (!rtIsNaN(dist_end_data[0])) {
        idx = 0;
      } else {
        idx = -1;
        g_idx = 2;
        exitg1 = false;
        while ((!exitg1) && (g_idx <= ii_size)) {
          if (!rtIsNaN(dist_end_data[g_idx - 1])) {
            idx = g_idx - 1;
            exitg1 = true;
          } else {
            g_idx++;
          }
        }
      }

      if (idx + 1 == 0) {
        C = dist_end_data[0];
        idx = 0;
      } else {
        C = dist_end_data[idx];
        for (g_idx = idx + 1; g_idx < ii_size; g_idx++) {
          if (C > dist_end_data[g_idx]) {
            C = dist_end_data[g_idx];
            idx = g_idx;
          }
        }
      }
    }

    if ((op_distance_n < C) || rtIsNaN(C)) {
      op_distance_n_0 = op_distance_n;
    } else {
      op_distance_n_0 = C;
    }

    if (op_distance_n_0 < dist_op_data[b_index_data[(int32_T)ind_temp_data[0] -
        1] - 1]) {
      *note = 2.0;
      if (!(op_distance_n <= C)) {
        nx = idx;
      }

      for (g_idx = 0; g_idx < loop_ub; g_idx++) {
        x_data[g_idx] = (Map_data_data[b_index_data[(int32_T)ind_temp_data[0] -
                         1] - 1] == SEG_GPS_HEAD_data[g_idx]);
      }

      g_idx = 0;
      idx = 1;
      exitg1 = false;
      while ((!exitg1) && (idx <= loop_ub)) {
        if (x_data[idx - 1]) {
          g_idx++;
          ii_data[g_idx - 1] = idx;
          if (g_idx >= loop_ub) {
            exitg1 = true;
          } else {
            idx++;
          }
        } else {
          idx++;
        }
      }

      *head_err = std::abs(heading - SEG_GPS_HEAD_data[(ii_data[0] +
        Map_data_size[0]) - 1]);
      if (*head_err <= 0.78539816339744828) {
        *seg_direction = 1.0;
      } else if ((*head_err >= 0.78539816339744828) && (*head_err <= 90.0)) {
        *seg_direction = 0.0;
      } else {
        *seg_direction = 2.0;
      }

      *seg_id_near = Map_data_data[nx];
      *op_distance = dist_op_data[nx];
      oi_near[0] = oi_xy_data[nx];
      oi_near[1] = oi_xy_data[nx + oi_xy_size[0]];
    }
  }

  loop_ub = Map_data_size[0];
  for (g_idx = 0; g_idx < loop_ub; g_idx++) {
    ind_temp_data[g_idx] = Map_data_data[Map_data_size[0] * 7 + g_idx] *
      3.1415926535897931 / 180.0;
  }

  loop_ub = Map_data_size[0];
  for (g_idx = 0; g_idx < loop_ub; g_idx++) {
    x_data[g_idx] = (seg_id_data[g_idx] == *seg_id_near);
  }

  nx = 0;
  i_ex = x_data[0];
  for (idx = 1; idx < loop_ub; idx++) {
    if ((int32_T)i_ex < (int32_T)x_data[idx]) {
      i_ex = x_data[idx];
      nx = idx;
    }
  }

  op_distance_n = oi_near[1] - Map_data_data[Map_data_size[0] * 5 + nx] *
    oi_near[0];
  if (Map_data_data[Map_data_size[0] * 5 + nx] < 0.0) {
    C = (-Map_data_data[Map_data_size[0] * 5 + nx] * X_pos[0] - op_distance_n) +
      X_pos[1];
    if (ind_temp_data[nx] > 4.71238898038469) {
      if (!(ind_temp_data[nx] < 6.2831853071795862)) {
        C = -C;
      }
    } else {
      C = -C;
    }
  } else if (Map_data_data[Map_data_size[0] * 5 + nx] == 0.0) {
    if (oi_near[1] < X_pos[1]) {
      C = -1.0;
    } else {
      C = 1.0;
    }
  } else {
    C = (Map_data_data[Map_data_size[0] * 5 + nx] * X_pos[0] + op_distance_n) -
      X_pos[1];
    if (ind_temp_data[nx] > 3.1415926535897931) {
      if (!(ind_temp_data[nx] < 4.71238898038469)) {
        C = -C;
      }
    } else {
      C = -C;
    }
  }

  num_lane_direction[0] = Map_data_data[Map_data_size[0] * 5 + nx];
  num_lane_direction[1] = op_distance_n;
  num_lane_direction[2] = C;
  if (C < 0.0) {
    num_lane_direction[3] = 1.0;
  } else if (C == 0.0) {
    num_lane_direction[3] = 1.0;
  } else {
    num_lane_direction[3] = -1.0;
  }

  *seg_heading = ind_temp_data[nx];
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

// Function for MATLAB Function: '<S2>/EndPointDecision'
void MM_DPP_1ModelClass::power_ec(const real_T a_data[], const int32_T *a_size,
  real_T y_data[], int32_T *y_size)
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

// Function for MATLAB Function: '<S2>/EndPointDecision1'
void MM_DPP_1ModelClass::power_jb(const real_T a_data[], const int32_T *a_size,
  real_T y_data[], int32_T *y_size)
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

// Function for MATLAB Function: '<S2>/DangerousArea'
void MM_DPP_1ModelClass::abs_g(const real_T x[4], real_T y[4])
{
  y[0] = std::abs(x[0]);
  y[1] = std::abs(x[1]);
  y[2] = std::abs(x[2]);
  y[3] = std::abs(x[3]);
}

// Function for MATLAB Function: '<S2>/DangerousArea'
void MM_DPP_1ModelClass::power_j(const real_T a[4], real_T y[4])
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

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
void MM_DPP_1ModelClass::G2splines(real_T xa, real_T ya, real_T thetaa, real_T
  ka, real_T xb, real_T yb, real_T thetab, real_T kb, real_T path_length, real_T
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

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
void MM_DPP_1ModelClass::power_dw(const real_T a[143], real_T y[143])
{
  int32_T k;
  for (k = 0; k < 143; k++) {
    y[k] = a[k] * a[k];
  }
}

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
void MM_DPP_1ModelClass::power_dw3(const real_T a[143], real_T y[143])
{
  int32_T k;
  for (k = 0; k < 143; k++) {
    y[k] = std::sqrt(a[k]);
  }
}

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
real_T MM_DPP_1ModelClass::std(const real_T x[13])
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

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
void MM_DPP_1ModelClass::power_dw3x(const real_T a[13], real_T y[13])
{
  int32_T k;
  for (k = 0; k < 13; k++) {
    y[k] = a[k] * a[k];
  }
}

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
void MM_DPP_1ModelClass::exp_n(real_T x[13])
{
  int32_T k;
  for (k = 0; k < 13; k++) {
    x[k] = std::exp(x[k]);
  }
}

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
real_T MM_DPP_1ModelClass::sum_a(const real_T x[13])
{
  real_T y;
  int32_T k;
  y = x[0];
  for (k = 0; k < 12; k++) {
    y += x[k + 1];
  }

  return y;
}

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
void MM_DPP_1ModelClass::power_d(const real_T a[11], real_T y[11])
{
  int32_T k;
  for (k = 0; k < 11; k++) {
    y[k] = a[k] * a[k];
  }
}

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
void MM_DPP_1ModelClass::sqrt_l(real_T x[11])
{
  int32_T k;
  for (k = 0; k < 11; k++) {
    x[k] = std::sqrt(x[k]);
  }
}

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
void MM_DPP_1ModelClass::power_dw3xd(const real_T a_data[], const int32_T
  a_size[2], real_T y_data[], int32_T y_size[2])
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

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
void MM_DPP_1ModelClass::sum_ae(const real_T x_data[], const int32_T x_size[2],
  real_T y_data[], int32_T y_size[2])
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

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
void MM_DPP_1ModelClass::sqrt_l5(real_T x_data[], int32_T x_size[2])
{
  int32_T k;
  for (k = 0; k < x_size[1]; k++) {
    x_data[k] = std::sqrt(x_data[k]);
  }
}

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
real_T MM_DPP_1ModelClass::mod(real_T x)
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

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
void MM_DPP_1ModelClass::abs_a(const real_T x[143], real_T y[143])
{
  int32_T k;
  for (k = 0; k < 143; k++) {
    y[k] = std::abs(x[k]);
  }
}

// Function for MATLAB Function: '<S2>/DynamicPathPlanning1'
void MM_DPP_1ModelClass::G2splines_e(real_T xa, real_T ya, real_T thetaa, real_T
  ka, real_T xb, real_T yb, real_T thetab, real_T kb, real_T path_length, real_T
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

// Function for MATLAB Function: '<S2>/DynamicPathPlanning1'
void MM_DPP_1ModelClass::power_egqso(const real_T a_data[], const int32_T
  a_size[2], real_T y_data[], int32_T y_size[2])
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

// Function for MATLAB Function: '<S2>/DynamicPathPlanning1'
void MM_DPP_1ModelClass::sum_hx(const real_T x_data[], const int32_T x_size[2],
  real_T y_data[], int32_T y_size[2])
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

// Function for MATLAB Function: '<S13>/Dijkstra'
void MM_DPP_1ModelClass::power_n(const real_T a[2], real_T y[2])
{
  y[0] = a[0] * a[0];
  y[1] = a[1] * a[1];
}

// Function for MATLAB Function: '<S13>/Dijkstra'
real_T MM_DPP_1ModelClass::sum_e(const real_T x[2])
{
  return x[0] + x[1];
}

// Model step function
void MM_DPP_1ModelClass::step()
{
  real_T SLAM_X_out;
  real_T SLAM_Y_out;
  real_T SLAM_Heading_out;
  static const real_T b[5] = { 0.00025, 0.00025, 1.0E-7, 1.0E-5, 0.0001 };

  real_T p0[25];
  real_T p_sqrt_data[25];
  real_T temp_dia[5];
  int32_T jj;
  real_T ajj;
  int32_T iy;
  int8_T ii_data[5];
  int32_T b_colj;
  int32_T d_ix;
  real_T K1[4];
  real_T x[11];
  int8_T I[25];
  real_T head_err;
  real_T oi_xy_data[376];
  real_T dist_op_data[188];
  real_T total_length;
  real_T Forward_Static_Path_id_0_data[188];
  int16_T cb_data[188];
  real_T target_k;
  real_T Length_1;
  real_T ang_1;
  real_T x_0;
  real_T OBXY_m[8];
  real_T c;
  real_T offset_2;
  real_T offset_3;
  real_T offset_4;
  real_T offset_5;
  real_T offset_6;
  real_T offset[13];
  real_T x_endpoint1;
  real_T y_endpoint1;
  real_T x_endpoint2;
  real_T y_endpoint2;
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
  real_T XP1[6];
  real_T YP1[6];
  real_T K1_0[11];
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
  int8_T l_data[9];
  int32_T p_data[188];
  int32_T q_data[188];
  int32_T s_data[188];
  real_T f_maxval[13];
  real_T end_heading_0_data[188];
  real_T end_heading;
  int32_T p_data_0[188];
  int32_T q_data_0[188];
  int32_T s_data_0[188];
  static const real_T a[11] = { 0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0 };

  real_T table[376];
  real_T shortest_distance[188];
  int8_T settled[188];
  uint8_T pidx_data[188];
  uint8_T zz_data[188];
  real_T tmp_path_data[188];
  uint8_T nidx_data[188];
  uint8_T c_data[188];
  int32_T ii_data_0[188];
  int32_T idx;
  int32_T b_idx;
  int32_T n;
  int32_T d_idx;
  boolean_T x_data[188];
  boolean_T b_x[188];
  uint8_T f_ii_data[188];
  real_T rtb_Add;
  real_T rtb_Gain1;
  real_T rtb_Gain_p;
  real_T rtb_X[5];
  real_T rtb_UnitDelay34[5];
  real_T rtb_X_state[5];
  real_T rtb_Oi_near_l[2];
  real_T rtb_num_lane_direction_f[4];
  real_T rtb_Look_ahead_S0;
  real_T rtb_H_y_out[4];
  real_T rtb_J_out_d[13];
  real_T rtb_Q_last_o[25];
  real_T rtb_R_last_o[25];
  real_T rtb_X_AUG[55];
  real_T rtb_K[25];
  real_T rtb_V_boundingbox[400];
  real_T rtb_Forward_Static_Path_x_h[188];
  real_T rtb_Forward_Static_Path_y_p[188];
  real_T rtb_Forward_Static_Path_id[188];
  real_T rtb_XP_i[78];
  real_T rtb_YP_g[78];
  real_T rtb_XP[78];
  real_T rtb_YP[78];
  int32_T i;
  real_T Path_col_0[13];
  real_T tmp[4];
  real_T ex[4];
  real_T rtb_X_AUG_0[10];
  real_T LastPath_overlap_data_0[22];
  real_T p_sqrt_data_0[25];
  real_T rtb_X_state_0[2];
  real_T rtb_num_lane_direction_b[4];
  int16_T tmp_0;
  real_T rtb_Oi_near_o[8];
  real_T rtb_Oi_near_o_0[8];
  real_T OBXY_EL_0[8];
  real_T b_Path_dis_data_0[121];
  int32_T oi_xy_size[2];
  int32_T Path_overlap_size[2];
  int32_T Path_overlap_size_0[2];
  int32_T LastPath_overlap_size[2];
  int32_T Path_overlap_size_1[2];
  int32_T LastPath_overlap_size_0[2];
  int32_T Path_overlap_size_2[2];
  int32_T LastPath_overlap_size_1[2];
  int32_T Path_overlap_size_3[2];
  int32_T LastPath_overlap_size_2[2];
  boolean_T b_x_0;
  int32_T Path_RES_1_size_idx_0;
  int32_T xy_ends_POS_size_idx_0;
  real_T rtb_TmpSignalConversionAtSFun_0;
  int32_T Path_RES_0_size_idx_1;
  int32_T shortest_distance_tmp;
  real_T table_tmp;
  boolean_T exitg1;
  int32_T exitg2;
  boolean_T exitg3;
  boolean_T exitg4;
  boolean_T guard1 = false;

  // Sum: '<S2>/Add' incorporates:
  //   Abs: '<S2>/Abs'
  //   Abs: '<S2>/Abs1'
  //   Constant: '<S2>/Constant7'
  //   Constant: '<S2>/Constant8'
  //   Memory: '<S2>/Memory'
  //   Memory: '<S2>/Memory1'
  //   Sum: '<S2>/Sum'
  //   Sum: '<S2>/Sum1'

  rtb_Add = std::abs(1.0 - rtDW.Memory1_PreviousInput) + std::abs(188.0 -
    rtDW.Memory_PreviousInput);

  // Outputs for Enabled SubSystem: '<S2>/Enabled Subsystem' incorporates:
  //   EnablePort: '<S13>/Enable'

  if (rtb_Add > 0.0) {
    // MATLAB Function: '<S13>/Dijkstra' incorporates:
    //   Constant: '<S2>/Constant7'

    memset(&table[0], 0, 376U * sizeof(real_T));
    for (i = 0; i < 188; i++) {
      shortest_distance[i] = (rtInf);
      settled[i] = 0;
    }

    memset(&rtDW.path[0], 0, 35344U * sizeof(real_T));
    idx = 0;
    b_idx = 1;
    exitg1 = false;
    while ((!exitg1) && (b_idx < 189)) {
      if (rtConstP.Constant3_Value[b_idx - 1] == 1.0) {
        idx++;
        ii_data_0[idx - 1] = b_idx;
        if (idx >= 188) {
          exitg1 = true;
        } else {
          b_idx++;
        }
      } else {
        b_idx++;
      }
    }

    if (1 > idx) {
      idx = 0;
    }

    for (shortest_distance_tmp = 0; shortest_distance_tmp < idx;
         shortest_distance_tmp++) {
      pidx_data[shortest_distance_tmp] = (uint8_T)
        ii_data_0[shortest_distance_tmp];
    }

    shortest_distance[ii_data_0[0] - 1] = 0.0;
    table[ii_data_0[0] + 187] = 0.0;
    settled[ii_data_0[0] - 1] = 1;
    rtDW.path[ii_data_0[0] - 1] = 1.0;
    b_idx = 0;
    idx = 1;
    exitg1 = false;
    while ((!exitg1) && (idx < 189)) {
      if (rtConstP.Constant3_Value[idx - 1] == 188.0) {
        b_idx++;
        ii_data_0[b_idx - 1] = idx;
        if (b_idx >= 188) {
          exitg1 = true;
        } else {
          idx++;
        }
      } else {
        idx++;
      }
    }

    if (1 > b_idx) {
      idx = 0;
    } else {
      idx = b_idx;
    }

    for (shortest_distance_tmp = 0; shortest_distance_tmp < idx;
         shortest_distance_tmp++) {
      zz_data[shortest_distance_tmp] = (uint8_T)ii_data_0[shortest_distance_tmp];
    }

    do {
      exitg2 = 0;
      shortest_distance_tmp = zz_data[0] - 1;
      if (settled[shortest_distance_tmp] == 0) {
        for (iy = 0; iy < 188; iy++) {
          table[iy] = table[188 + iy];
        }

        n = pidx_data[0] + 187;
        table[n] = 0.0;
        d_idx = 0;
        for (i = 0; i < 188; i++) {
          b_x_0 = (rtConstP.Constant3_Value[pidx_data[0] - 1] ==
                   rtConstP.Constant5_Value[188 + i]);
          if (b_x_0) {
            d_idx++;
          }

          b_x[i] = b_x_0;
        }

        i = d_idx;
        d_idx = 0;
        for (idx = 0; idx < 188; idx++) {
          if (b_x[idx]) {
            c_data[d_idx] = (uint8_T)(idx + 1);
            d_idx++;
          }
        }

        for (b_idx = 0; b_idx < i; b_idx++) {
          for (iy = 0; iy < 188; iy++) {
            b_x[iy] = (rtConstP.Constant5_Value[c_data[b_idx] + 375] ==
                       rtConstP.Constant3_Value[iy]);
          }

          d_idx = -1;
          idx = 1;
          exitg1 = false;
          while ((!exitg1) && (idx < 189)) {
            if (b_x[idx - 1]) {
              d_idx++;
              ii_data_0[d_idx] = idx;
              if (d_idx + 1 >= 188) {
                exitg1 = true;
              } else {
                idx++;
              }
            } else {
              idx++;
            }
          }

          if (!(settled[ii_data_0[0] - 1] != 0)) {
            rtb_X_state_0[0] = rtConstP.Constant3_Value[n] -
              rtConstP.Constant3_Value[ii_data_0[0] + 187];
            rtb_X_state_0[1] = rtConstP.Constant3_Value[pidx_data[0] + 375] -
              rtConstP.Constant3_Value[ii_data_0[0] + 375];
            power_n(rtb_X_state_0, rtb_Oi_near_l);
            ajj = std::sqrt(sum_e(rtb_Oi_near_l));
            if ((table[ii_data_0[0] - 1] == 0.0) || (table[ii_data_0[0] - 1] >
                 table[pidx_data[0] - 1] + ajj)) {
              table[ii_data_0[0] + 187] = table[pidx_data[0] - 1] + ajj;
              for (iy = 0; iy < 188; iy++) {
                b_x[iy] = (rtDW.path[(188 * iy + pidx_data[0]) - 1] != 0.0);
              }

              idx = 0;
              d_idx = 1;
              exitg1 = false;
              while ((!exitg1) && (d_idx < 189)) {
                if (b_x[d_idx - 1]) {
                  idx++;
                  f_ii_data[idx - 1] = (uint8_T)d_idx;
                  if (idx >= 188) {
                    exitg1 = true;
                  } else {
                    d_idx++;
                  }
                } else {
                  d_idx++;
                }
              }

              if (1 > idx) {
                b_colj = 0;
              } else {
                b_colj = idx;
              }

              d_ix = b_colj - 1;
              if (0 <= d_ix) {
                memset(&tmp_path_data[0], 0, (d_ix + 1) * sizeof(real_T));
              }

              for (d_idx = 0; d_idx < b_colj; d_idx++) {
                tmp_path_data[d_idx] = rtDW.path[((f_ii_data[d_idx] - 1) * 188 +
                  pidx_data[0]) - 1];
              }

              idx = ii_data_0[0] - 1;
              for (iy = 0; iy < b_colj; iy++) {
                rtDW.path[idx + 188 * iy] = tmp_path_data[iy];
              }

              rtDW.path[idx + 188 * b_colj] =
                rtConstP.Constant5_Value[c_data[b_idx] + 375];
            } else {
              table[ii_data_0[0] + 187] = table[ii_data_0[0] - 1];
            }
          }
        }

        b_idx = 0;
        idx = 1;
        exitg1 = false;
        while ((!exitg1) && (idx < 189)) {
          if (table[idx + 187] != 0.0) {
            b_idx++;
            ii_data_0[b_idx - 1] = idx;
            if (b_idx >= 188) {
              exitg1 = true;
            } else {
              idx++;
            }
          } else {
            idx++;
          }
        }

        if (1 > b_idx) {
          idx = 0;
        } else {
          idx = b_idx;
        }

        for (iy = 0; iy < idx; iy++) {
          nidx_data[iy] = (uint8_T)ii_data_0[iy];
        }

        if (idx <= 2) {
          if (idx == 1) {
            offset_2 = table[ii_data_0[0] + 187];
          } else if (table[ii_data_0[0] + 187] > table[ii_data_0[1] + 187]) {
            offset_2 = table[ii_data_0[1] + 187];
          } else if (rtIsNaN(table[ii_data_0[0] + 187])) {
            if (!rtIsNaN(table[ii_data_0[1] + 187])) {
              offset_2 = table[ii_data_0[1] + 187];
            } else {
              offset_2 = table[ii_data_0[0] + 187];
            }
          } else {
            offset_2 = table[ii_data_0[0] + 187];
          }
        } else {
          if (!rtIsNaN(table[ii_data_0[0] + 187])) {
            d_idx = 1;
          } else {
            d_idx = 0;
            b_idx = 2;
            exitg1 = false;
            while ((!exitg1) && (b_idx <= idx)) {
              if (!rtIsNaN(table[ii_data_0[b_idx - 1] + 187])) {
                d_idx = b_idx;
                exitg1 = true;
              } else {
                b_idx++;
              }
            }
          }

          if (d_idx == 0) {
            offset_2 = table[ii_data_0[0] + 187];
          } else {
            offset_2 = table[ii_data_0[d_idx - 1] + 187];
            while (d_idx + 1 <= idx) {
              if (offset_2 > table[ii_data_0[d_idx] + 187]) {
                offset_2 = table[ii_data_0[d_idx] + 187];
              }

              d_idx++;
            }
          }
        }

        for (iy = 0; iy < idx; iy++) {
          x_data[iy] = (table[ii_data_0[iy] + 187] == offset_2);
        }

        d_idx = 0;
        b_idx = 1;
        exitg1 = false;
        while ((!exitg1) && (b_idx <= idx)) {
          if (x_data[b_idx - 1]) {
            d_idx++;
            ii_data_0[d_idx - 1] = b_idx;
            if (d_idx >= idx) {
              exitg1 = true;
            } else {
              b_idx++;
            }
          } else {
            b_idx++;
          }
        }

        if (idx == 1) {
          if (d_idx == 0) {
            idx = 0;
          }
        } else if (1 > d_idx) {
          idx = 0;
        } else {
          idx = d_idx;
        }

        if (idx == 0) {
          exitg2 = 1;
        } else {
          pidx_data[0] = nidx_data[ii_data_0[0] - 1];
          shortest_distance_tmp = nidx_data[ii_data_0[0] - 1] - 1;
          shortest_distance[shortest_distance_tmp] = table[nidx_data[ii_data_0[0]
            - 1] + 187];
          settled[shortest_distance_tmp] = 1;
        }
      } else {
        exitg2 = 1;
      }
    } while (exitg2 == 0);

    for (iy = 0; iy < 188; iy++) {
      b_x[iy] = (rtDW.path[(188 * iy + zz_data[0]) - 1] != 0.0);
    }

    b_idx = 0;
    idx = 0;
    exitg1 = false;
    while ((!exitg1) && (idx + 1 < 189)) {
      if (b_x[idx]) {
        b_idx++;
        if (b_idx >= 188) {
          exitg1 = true;
        } else {
          idx++;
        }
      } else {
        idx++;
      }
    }

    if (1 > b_idx) {
      b_colj = 0;
    } else {
      b_colj = b_idx;
    }

    if (1 > b_colj) {
      b_colj = 0;
    }

    rtDW.dist = shortest_distance[shortest_distance_tmp];
    rtDW.SFunction_DIMS3_c = b_colj;
    for (shortest_distance_tmp = 0; shortest_distance_tmp < b_colj;
         shortest_distance_tmp++) {
      rtDW.path_2[shortest_distance_tmp] = rtDW.path[(188 *
        shortest_distance_tmp + zz_data[0]) - 1];
    }

    // End of MATLAB Function: '<S13>/Dijkstra'
  }

  // End of Outputs for SubSystem: '<S2>/Enabled Subsystem'

  // MATLAB Function: '<S2>/Final_Static_Path' incorporates:
  //   Constant: '<S2>/Constant6'

  if (!rtDW.path_out1_not_empty) {
    if (rtb_Add > 0.0) {
      rtDW.path_out1.size = rtDW.SFunction_DIMS3_c;
      for (shortest_distance_tmp = 0; shortest_distance_tmp <
           rtDW.SFunction_DIMS3_c; shortest_distance_tmp++) {
        rtDW.path_out1.data[shortest_distance_tmp] =
          rtDW.path_2[shortest_distance_tmp];
      }

      rtDW.path_out1_not_empty = !(rtDW.path_out1.size == 0);
    } else {
      rtDW.path_out1.size = 2;
      rtDW.path_out1.data[0] = 0.0;
      rtDW.path_out1.data[1] = 0.0;
      rtDW.path_out1_not_empty = true;
    }
  }

  if (rtb_Add > 0.0) {
    rtDW.path_out1.size = rtDW.SFunction_DIMS3_c;
    for (shortest_distance_tmp = 0; shortest_distance_tmp <
         rtDW.SFunction_DIMS3_c; shortest_distance_tmp++) {
      rtDW.path_out1.data[shortest_distance_tmp] =
        rtDW.path_2[shortest_distance_tmp];
    }

    rtDW.path_out1_not_empty = !(rtDW.path_out1.size == 0);
  }

  rtDW.SFunction_DIMS2_m = rtDW.path_out1.size;
  rtDW.SFunction_DIMS3_l = rtDW.path_out1.size;
  rtDW.SFunction_DIMS4_h[0] = 188;
  rtDW.SFunction_DIMS4_h[1] = 23;
  memcpy(&rtDW.Static_Path_0[0], &rtConstP.pooled2[0], 4324U * sizeof(real_T));
  rtDW.SFunction_DIMS6_c = rtDW.path_out1.size;

  // Gain: '<S1>/Gain' incorporates:
  //   Gain: '<Root>/Gain'
  //   Inport: '<Root>/angular_vz'

  rtb_Gain_p = -(0.017453292519943295 * rtU.angular_vz);

  // MATLAB Function: '<S3>/SLAM_Check' incorporates:
  //   Gain: '<S1>/Gain1'
  //   Inport: '<Root>/SLAM_counter'
  //   Inport: '<Root>/SLAM_fault'
  //   Inport: '<Root>/SLAM_heading'
  //   Inport: '<Root>/SLAM_x'
  //   Inport: '<Root>/SLAM_y'
  //   Inport: '<Root>/Speed_mps'
  //   SignalConversion: '<S4>/TmpSignal ConversionAt SFunction Inport6'
  //   UnitDelay: '<S3>/Unit Delay1'
  //   UnitDelay: '<S3>/Unit Delay35'
  //   UnitDelay: '<S3>/Unit Delay36'
  //   UnitDelay: '<S3>/Unit Delay37'

  d_idx = 0;
  if (rtU.SLAM_counter != rtDW.UnitDelay35_DSTATE[3]) {
    d_idx = 1;

    // Update for UnitDelay: '<S3>/Unit Delay38'
    rtDW.UnitDelay38_DSTATE = 0.0;
  } else {
    // Update for UnitDelay: '<S3>/Unit Delay38'
    rtDW.UnitDelay38_DSTATE++;
  }

  if (rtU.SLAM_fault == 1.0) {
    d_idx = 0;
  }

  if (d_idx == 0) {
    SLAM_X_out = rtDW.UnitDelay35_DSTATE[0];
    SLAM_Y_out = rtDW.UnitDelay35_DSTATE[1];
    SLAM_Heading_out = rtDW.UnitDelay35_DSTATE[2];
    memcpy(&rtb_Q_last_o[0], &rtDW.UnitDelay37_DSTATE[0], 25U * sizeof(real_T));
    memcpy(&rtb_R_last_o[0], &rtDW.UnitDelay36_DSTATE[0], 25U * sizeof(real_T));
  } else {
    SLAM_X_out = rtU.SLAM_x;
    SLAM_Y_out = rtU.SLAM_y;
    SLAM_Heading_out = rtU.SLAM_heading;
    memset(&rtb_Q_last_o[0], 0, 25U * sizeof(real_T));
    for (b_colj = 0; b_colj < 5; b_colj++) {
      rtb_Q_last_o[b_colj + 5 * b_colj] = b[b_colj];
    }

    rtb_X_state[0] = 0.0001;
    rtb_X_state[1] = 0.0001;
    if (std::abs(rtU.SLAM_heading - rtDW.UnitDelay1_DSTATE[2]) > 5.5) {
      rtb_X_state[2] = 100.0;
    } else {
      rtb_X_state[2] = 0.1;
    }

    rtb_X_state[3] = 0.1;
    rtb_X_state[4] = 0.0001;
    memset(&rtb_R_last_o[0], 0, 25U * sizeof(real_T));
    for (b_colj = 0; b_colj < 5; b_colj++) {
      rtb_R_last_o[b_colj + 5 * b_colj] = rtb_X_state[b_colj];
    }
  }

  rtb_X_state[0] = SLAM_X_out;
  rtb_X_state[1] = SLAM_Y_out;
  rtb_X_state[2] = SLAM_Heading_out;
  rtb_X_state[3] = rtb_Gain_p;
  rtb_X_state[4] = 1.025 * rtU.Speed_mps;

  // UnitDelay: '<S3>/Unit Delay34'
  for (i = 0; i < 5; i++) {
    rtb_UnitDelay34[i] = rtDW.UnitDelay34_DSTATE[i];
  }

  // End of UnitDelay: '<S3>/Unit Delay34'

  // MATLAB Function: '<S3>/SLAM_Generate_sigma_pt_UKF' incorporates:
  //   Constant: '<S1>/Constant28'
  //   UnitDelay: '<S3>/Unit Delay33'

  memcpy(&p_sqrt_data[0], &rtDW.UnitDelay33_DSTATE[0], 25U * sizeof(real_T));
  idx = 0;
  i = 0;
  b_colj = 1;
  exitg1 = false;
  while ((!exitg1) && (b_colj <= 5)) {
    jj = (i + b_colj) - 1;
    ajj = 0.0;
    if (!(b_colj - 1 < 1)) {
      n = i;
      iy = i;
      for (b_idx = 1; b_idx < b_colj; b_idx++) {
        ajj += p_sqrt_data[n] * p_sqrt_data[iy];
        n++;
        iy++;
      }
    }

    ajj = p_sqrt_data[jj] - ajj;
    if (ajj > 0.0) {
      ajj = std::sqrt(ajj);
      p_sqrt_data[jj] = ajj;
      if (b_colj < 5) {
        if (b_colj - 1 != 0) {
          b_idx = jj + 5;
          n = ((4 - b_colj) * 5 + i) + 6;
          for (shortest_distance_tmp = i + 6; shortest_distance_tmp <= n;
               shortest_distance_tmp += 5) {
            d_ix = i;
            end_heading = 0.0;
            iy = (shortest_distance_tmp + b_colj) - 2;
            for (Path_RES_0_size_idx_1 = shortest_distance_tmp;
                 Path_RES_0_size_idx_1 <= iy; Path_RES_0_size_idx_1++) {
              end_heading += p_sqrt_data[Path_RES_0_size_idx_1 - 1] *
                p_sqrt_data[d_ix];
              d_ix++;
            }

            p_sqrt_data[b_idx] += -end_heading;
            b_idx += 5;
          }
        }

        ajj = 1.0 / ajj;
        n = ((4 - b_colj) * 5 + jj) + 6;
        for (b_idx = jj + 5; b_idx + 1 <= n; b_idx += 5) {
          p_sqrt_data[b_idx] *= ajj;
        }

        i += 5;
      }

      b_colj++;
    } else {
      p_sqrt_data[jj] = ajj;
      idx = b_colj;
      exitg1 = true;
    }
  }

  if (idx == 0) {
    b_idx = 5;
  } else {
    b_idx = idx - 1;
  }

  for (b_colj = 1; b_colj <= b_idx; b_colj++) {
    for (i = b_colj; i < b_idx; i++) {
      p_sqrt_data[i + 5 * (b_colj - 1)] = 0.0;
    }
  }

  if (1 > b_idx) {
    b_idx = 0;
    d_ix = 0;
  } else {
    d_ix = b_idx;
  }

  for (shortest_distance_tmp = 0; shortest_distance_tmp < d_ix;
       shortest_distance_tmp++) {
    for (iy = 0; iy < b_idx; iy++) {
      p_sqrt_data_0[iy + b_idx * shortest_distance_tmp] = p_sqrt_data[5 *
        shortest_distance_tmp + iy];
    }
  }

  for (shortest_distance_tmp = 0; shortest_distance_tmp < d_ix;
       shortest_distance_tmp++) {
    for (iy = 0; iy < b_idx; iy++) {
      i = b_idx * shortest_distance_tmp;
      p_sqrt_data[iy + i] = p_sqrt_data_0[i + iy];
    }
  }

  memset(&rtb_X_AUG[0], 0, 55U * sizeof(real_T));
  if (idx != 0) {
    for (i = 0; i < 5; i++) {
      temp_dia[i] = std::abs(rtDW.UnitDelay33_DSTATE[5 * i + i]);
    }

    idx = 0;
    b_idx = 1;
    exitg1 = false;
    while ((!exitg1) && (b_idx < 6)) {
      if (temp_dia[b_idx - 1] < 1.0E-10) {
        idx++;
        ii_data[idx - 1] = (int8_T)b_idx;
        if (idx >= 5) {
          exitg1 = true;
        } else {
          b_idx++;
        }
      } else {
        b_idx++;
      }
    }

    if (!(1 > idx)) {
      for (shortest_distance_tmp = 0; shortest_distance_tmp < idx;
           shortest_distance_tmp++) {
        temp_dia[ii_data[shortest_distance_tmp] - 1] = 1.0E-10;
      }
    }

    memset(&p0[0], 0, 25U * sizeof(real_T));
    for (idx = 0; idx < 5; idx++) {
      p0[idx + 5 * idx] = temp_dia[idx];
    }

    b_idx = 0;
    b_colj = 0;
    idx = 1;
    exitg1 = false;
    while ((!exitg1) && (idx < 6)) {
      i = (b_colj + idx) - 1;
      ajj = 0.0;
      if (!(idx - 1 < 1)) {
        n = b_colj;
        jj = b_colj;
        for (iy = 1; iy < idx; iy++) {
          ajj += p0[n] * p0[jj];
          n++;
          jj++;
        }
      }

      ajj = p0[i] - ajj;
      if (ajj > 0.0) {
        ajj = std::sqrt(ajj);
        p0[i] = ajj;
        if (idx < 5) {
          if (idx - 1 != 0) {
            iy = i + 5;
            jj = ((4 - idx) * 5 + b_colj) + 6;
            for (shortest_distance_tmp = b_colj + 6; shortest_distance_tmp <= jj;
                 shortest_distance_tmp += 5) {
              d_ix = b_colj;
              end_heading = 0.0;
              n = (shortest_distance_tmp + idx) - 2;
              for (Path_RES_0_size_idx_1 = shortest_distance_tmp;
                   Path_RES_0_size_idx_1 <= n; Path_RES_0_size_idx_1++) {
                end_heading += p0[Path_RES_0_size_idx_1 - 1] * p0[d_ix];
                d_ix++;
              }

              p0[iy] += -end_heading;
              iy += 5;
            }
          }

          ajj = 1.0 / ajj;
          n = ((4 - idx) * 5 + i) + 6;
          for (i += 5; i + 1 <= n; i += 5) {
            p0[i] *= ajj;
          }

          b_colj += 5;
        }

        idx++;
      } else {
        p0[i] = ajj;
        b_idx = idx;
        exitg1 = true;
      }
    }

    if (b_idx == 0) {
      b_idx = 5;
    } else {
      b_idx--;
    }

    for (idx = 0; idx < b_idx; idx++) {
      for (b_colj = idx + 1; b_colj < b_idx; b_colj++) {
        p0[b_colj + 5 * idx] = 0.0;
      }
    }

    b_idx = 5;
    d_ix = 5;
    memcpy(&p_sqrt_data[0], &p0[0], 25U * sizeof(real_T));
  }

  for (shortest_distance_tmp = 0; shortest_distance_tmp < b_idx;
       shortest_distance_tmp++) {
    for (iy = 0; iy < d_ix; iy++) {
      p_sqrt_data_0[iy + d_ix * shortest_distance_tmp] = p_sqrt_data[b_idx * iy
        + shortest_distance_tmp] * 2.23606797749979;
    }
  }

  for (shortest_distance_tmp = 0; shortest_distance_tmp < b_idx;
       shortest_distance_tmp++) {
    for (iy = 0; iy < d_ix; iy++) {
      i = d_ix * shortest_distance_tmp;
      p_sqrt_data[iy + i] = p_sqrt_data_0[i + iy];
    }
  }

  for (shortest_distance_tmp = 0; shortest_distance_tmp < 5;
       shortest_distance_tmp++) {
    rtb_X_AUG[shortest_distance_tmp] = rtb_UnitDelay34[shortest_distance_tmp];
  }

  for (i = 0; i < 5; i++) {
    b_colj = d_ix - 1;
    for (shortest_distance_tmp = 0; shortest_distance_tmp <= b_colj;
         shortest_distance_tmp++) {
      temp_dia[shortest_distance_tmp] = p_sqrt_data[d_ix * i +
        shortest_distance_tmp];
    }

    shortest_distance_tmp = i + 2;
    for (iy = 0; iy < 5; iy++) {
      rtb_X_AUG[iy + 5 * (shortest_distance_tmp - 1)] = rtb_UnitDelay34[iy] +
        temp_dia[iy];
    }
  }

  for (idx = 0; idx < 5; idx++) {
    b_colj = d_ix - 1;
    for (shortest_distance_tmp = 0; shortest_distance_tmp <= b_colj;
         shortest_distance_tmp++) {
      temp_dia[shortest_distance_tmp] = p_sqrt_data[d_ix * idx +
        shortest_distance_tmp];
    }

    shortest_distance_tmp = idx + 7;
    for (iy = 0; iy < 5; iy++) {
      rtb_X_AUG[iy + 5 * (shortest_distance_tmp - 1)] = rtb_UnitDelay34[iy] -
        temp_dia[iy];
    }
  }

  // End of MATLAB Function: '<S3>/SLAM_Generate_sigma_pt_UKF'

  // MATLAB Function: '<S3>/SLAM_UKF' incorporates:
  //   Constant: '<Root>/[Para] D_GC'
  //   Constant: '<S1>/Constant25'
  //   MATLAB Function: '<S3>/SLAM_Check'
  //   SignalConversion: '<S6>/TmpSignal ConversionAt SFunction Inport5'

  end_heading = 0.01 * rtb_Gain_p * 3.8;
  for (b_idx = 0; b_idx < 11; b_idx++) {
    rtb_X_AUG[5 * b_idx] = (rtb_X_AUG[5 * b_idx + 4] * 0.01 * std::cos
      (rtb_X_AUG[5 * b_idx + 2]) + rtb_X_AUG[5 * b_idx]) + std::cos(rtb_X_AUG[5 *
      b_idx + 2] + 1.5707963267948966) * end_heading;
    rtb_X_AUG[1 + 5 * b_idx] = (rtb_X_AUG[5 * b_idx + 4] * 0.01 * std::sin
      (rtb_X_AUG[5 * b_idx + 2]) + rtb_X_AUG[5 * b_idx + 1]) + std::sin
      (rtb_X_AUG[5 * b_idx + 2] + 1.5707963267948966) * end_heading;
    rtb_X_AUG[2 + 5 * b_idx] += rtb_X_AUG[5 * b_idx + 3] * 0.01;
  }

  for (shortest_distance_tmp = 0; shortest_distance_tmp < 10;
       shortest_distance_tmp++) {
    rtb_X_AUG_0[shortest_distance_tmp] = rtb_X_AUG[(1 + shortest_distance_tmp) *
      5];
  }

  rtb_X[0] = rtb_X_AUG[0] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (shortest_distance_tmp = 0; shortest_distance_tmp < 10;
       shortest_distance_tmp++) {
    rtb_X_AUG_0[shortest_distance_tmp] = rtb_X_AUG[(1 + shortest_distance_tmp) *
      5 + 1];
  }

  rtb_X[1] = rtb_X_AUG[1] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (shortest_distance_tmp = 0; shortest_distance_tmp < 10;
       shortest_distance_tmp++) {
    rtb_X_AUG_0[shortest_distance_tmp] = rtb_X_AUG[(1 + shortest_distance_tmp) *
      5 + 2];
  }

  rtb_X[2] = rtb_X_AUG[2] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (shortest_distance_tmp = 0; shortest_distance_tmp < 10;
       shortest_distance_tmp++) {
    rtb_X_AUG_0[shortest_distance_tmp] = rtb_X_AUG[(1 + shortest_distance_tmp) *
      5 + 3];
  }

  rtb_X[3] = rtb_X_AUG[3] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (shortest_distance_tmp = 0; shortest_distance_tmp < 10;
       shortest_distance_tmp++) {
    rtb_X_AUG_0[shortest_distance_tmp] = rtb_X_AUG[(1 + shortest_distance_tmp) *
      5 + 4];
  }

  rtb_X[4] = rtb_X_AUG[4] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (shortest_distance_tmp = 0; shortest_distance_tmp < 5;
       shortest_distance_tmp++) {
    ajj = rtb_X_AUG[shortest_distance_tmp] - rtb_X[shortest_distance_tmp];
    rtb_UnitDelay34[shortest_distance_tmp] = ajj;
    temp_dia[shortest_distance_tmp] = ajj;
  }

  for (shortest_distance_tmp = 0; shortest_distance_tmp < 5;
       shortest_distance_tmp++) {
    for (iy = 0; iy < 5; iy++) {
      p_sqrt_data[shortest_distance_tmp + 5 * iy] =
        rtb_UnitDelay34[shortest_distance_tmp] * temp_dia[iy];
    }
  }

  for (shortest_distance_tmp = 0; shortest_distance_tmp < 5;
       shortest_distance_tmp++) {
    for (iy = 0; iy < 5; iy++) {
      p0[iy + 5 * shortest_distance_tmp] = p_sqrt_data[5 * shortest_distance_tmp
        + iy] * 2.0;
    }
  }

  for (i = 0; i < 10; i++) {
    for (shortest_distance_tmp = 0; shortest_distance_tmp < 5;
         shortest_distance_tmp++) {
      ajj = rtb_X_AUG[(i + 1) * 5 + shortest_distance_tmp] -
        rtb_X[shortest_distance_tmp];
      rtb_UnitDelay34[shortest_distance_tmp] = ajj;
      temp_dia[shortest_distance_tmp] = ajj;
    }

    for (shortest_distance_tmp = 0; shortest_distance_tmp < 5;
         shortest_distance_tmp++) {
      for (iy = 0; iy < 5; iy++) {
        p_sqrt_data[shortest_distance_tmp + 5 * iy] =
          rtb_UnitDelay34[shortest_distance_tmp] * temp_dia[iy];
      }
    }

    for (shortest_distance_tmp = 0; shortest_distance_tmp < 5;
         shortest_distance_tmp++) {
      for (iy = 0; iy < 5; iy++) {
        b_idx = 5 * shortest_distance_tmp + iy;
        p0[iy + 5 * shortest_distance_tmp] = p_sqrt_data[b_idx] * 0.1 + p0[b_idx];
      }
    }
  }

  for (shortest_distance_tmp = 0; shortest_distance_tmp < 25;
       shortest_distance_tmp++) {
    p0[shortest_distance_tmp] += rtb_Q_last_o[shortest_distance_tmp];
  }

  if (rtb_X[2] < 0.0) {
    rtb_X[2] += 6.2831853071795862;
  } else {
    if (rtb_X[2] >= 6.2831853071795862) {
      rtb_X[2] -= 6.2831853071795862;
    }
  }

  if (d_idx > 0) {
    for (shortest_distance_tmp = 0; shortest_distance_tmp < 25;
         shortest_distance_tmp++) {
      p_sqrt_data[shortest_distance_tmp] = p0[shortest_distance_tmp] +
        rtb_R_last_o[shortest_distance_tmp];
    }

    invNxN(p_sqrt_data, p_sqrt_data_0);
    for (shortest_distance_tmp = 0; shortest_distance_tmp < 5;
         shortest_distance_tmp++) {
      for (iy = 0; iy < 5; iy++) {
        b_idx = shortest_distance_tmp + 5 * iy;
        rtb_K[b_idx] = 0.0;
        for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 5;
             Path_RES_0_size_idx_1++) {
          rtb_K[b_idx] = p0[5 * Path_RES_0_size_idx_1 + shortest_distance_tmp] *
            p_sqrt_data_0[5 * iy + Path_RES_0_size_idx_1] + rtb_K[5 * iy +
            shortest_distance_tmp];
        }
      }

      rtb_UnitDelay34[shortest_distance_tmp] = rtb_X_state[shortest_distance_tmp]
        - rtb_X[shortest_distance_tmp];
    }

    for (shortest_distance_tmp = 0; shortest_distance_tmp < 5;
         shortest_distance_tmp++) {
      end_heading = 0.0;
      for (iy = 0; iy < 5; iy++) {
        end_heading += rtb_K[5 * iy + shortest_distance_tmp] *
          rtb_UnitDelay34[iy];
      }

      rtb_X[shortest_distance_tmp] += end_heading;
    }

    for (shortest_distance_tmp = 0; shortest_distance_tmp < 25;
         shortest_distance_tmp++) {
      I[shortest_distance_tmp] = 0;
    }

    for (d_idx = 0; d_idx < 5; d_idx++) {
      I[d_idx + 5 * d_idx] = 1;
    }

    for (shortest_distance_tmp = 0; shortest_distance_tmp < 5;
         shortest_distance_tmp++) {
      for (iy = 0; iy < 5; iy++) {
        b_idx = 5 * shortest_distance_tmp + iy;
        p_sqrt_data[iy + 5 * shortest_distance_tmp] = (real_T)I[b_idx] -
          rtb_K[b_idx];
      }
    }

    for (shortest_distance_tmp = 0; shortest_distance_tmp < 5;
         shortest_distance_tmp++) {
      for (iy = 0; iy < 5; iy++) {
        b_idx = iy + 5 * shortest_distance_tmp;
        rtb_K[b_idx] = 0.0;
        for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 5;
             Path_RES_0_size_idx_1++) {
          rtb_K[b_idx] = p_sqrt_data[5 * Path_RES_0_size_idx_1 + iy] * p0[5 *
            shortest_distance_tmp + Path_RES_0_size_idx_1] + rtb_K[5 *
            shortest_distance_tmp + iy];
        }
      }
    }

    for (shortest_distance_tmp = 0; shortest_distance_tmp < 5;
         shortest_distance_tmp++) {
      for (iy = 0; iy < 5; iy++) {
        p0[iy + 5 * shortest_distance_tmp] = rtb_K[5 * shortest_distance_tmp +
          iy];
      }
    }
  } else {
    rtb_num_lane_direction_f[0] = p0[18] + rtb_R_last_o[18];
    rtb_num_lane_direction_f[1] = p0[19] + rtb_R_last_o[19];
    rtb_num_lane_direction_f[2] = p0[23] + rtb_R_last_o[23];
    rtb_num_lane_direction_f[3] = p0[24] + rtb_R_last_o[24];
    if (std::abs(rtb_num_lane_direction_f[1]) > std::abs
        (rtb_num_lane_direction_f[0])) {
      rtb_Gain1 = rtb_num_lane_direction_f[0] / rtb_num_lane_direction_f[1];
      rtb_Gain_p = 1.0 / (rtb_Gain1 * rtb_num_lane_direction_f[3] -
                          rtb_num_lane_direction_f[2]);
      rtb_H_y_out[0] = rtb_num_lane_direction_f[3] / rtb_num_lane_direction_f[1]
        * rtb_Gain_p;
      rtb_H_y_out[1] = -rtb_Gain_p;
      rtb_H_y_out[2] = -rtb_num_lane_direction_f[2] / rtb_num_lane_direction_f[1]
        * rtb_Gain_p;
      rtb_H_y_out[3] = rtb_Gain1 * rtb_Gain_p;
    } else {
      rtb_Gain1 = rtb_num_lane_direction_f[1] / rtb_num_lane_direction_f[0];
      rtb_Gain_p = 1.0 / (rtb_num_lane_direction_f[3] - rtb_Gain1 *
                          rtb_num_lane_direction_f[2]);
      rtb_H_y_out[0] = rtb_num_lane_direction_f[3] / rtb_num_lane_direction_f[0]
        * rtb_Gain_p;
      rtb_H_y_out[1] = -rtb_Gain1 * rtb_Gain_p;
      rtb_H_y_out[2] = -rtb_num_lane_direction_f[2] / rtb_num_lane_direction_f[0]
        * rtb_Gain_p;
      rtb_H_y_out[3] = rtb_Gain_p;
    }

    for (shortest_distance_tmp = 0; shortest_distance_tmp < 2;
         shortest_distance_tmp++) {
      K1[shortest_distance_tmp] = 0.0;
      K1[shortest_distance_tmp] += p0[shortest_distance_tmp + 18] * rtb_H_y_out
        [0];
      K1[shortest_distance_tmp] += p0[shortest_distance_tmp + 23] * rtb_H_y_out
        [1];
      K1[shortest_distance_tmp + 2] = 0.0;
      K1[shortest_distance_tmp + 2] += p0[shortest_distance_tmp + 18] *
        rtb_H_y_out[2];
      K1[shortest_distance_tmp + 2] += p0[shortest_distance_tmp + 23] *
        rtb_H_y_out[3];
      rtb_X_state_0[shortest_distance_tmp] = rtb_X_state[3 +
        shortest_distance_tmp] - rtb_X[3 + shortest_distance_tmp];
    }

    rtb_X[3] += K1[0] * rtb_X_state_0[0] + K1[2] * rtb_X_state_0[1];
    rtb_X[4] += K1[1] * rtb_X_state_0[0] + K1[3] * rtb_X_state_0[1];
    rtb_num_lane_direction_f[0] = 1.0 - K1[0];
    rtb_num_lane_direction_f[1] = 0.0 - K1[1];
    rtb_num_lane_direction_f[2] = 0.0 - K1[2];
    rtb_num_lane_direction_f[3] = 1.0 - K1[3];
    for (shortest_distance_tmp = 0; shortest_distance_tmp < 2;
         shortest_distance_tmp++) {
      rtb_num_lane_direction_b[shortest_distance_tmp] = 0.0;
      rtb_num_lane_direction_b[shortest_distance_tmp] +=
        rtb_num_lane_direction_f[shortest_distance_tmp] * p0[18];
      rtb_num_lane_direction_b[shortest_distance_tmp] +=
        rtb_num_lane_direction_f[shortest_distance_tmp + 2] * p0[19];
      rtb_num_lane_direction_b[shortest_distance_tmp + 2] = 0.0;
      rtb_num_lane_direction_b[shortest_distance_tmp + 2] +=
        rtb_num_lane_direction_f[shortest_distance_tmp] * p0[23];
      rtb_num_lane_direction_b[shortest_distance_tmp + 2] +=
        rtb_num_lane_direction_f[shortest_distance_tmp + 2] * p0[24];
    }

    p0[18] = rtb_num_lane_direction_b[0];
    p0[19] = rtb_num_lane_direction_b[1];
    p0[23] = rtb_num_lane_direction_b[2];
    p0[24] = rtb_num_lane_direction_b[3];
  }

  // End of MATLAB Function: '<S3>/SLAM_UKF'

  // MATLAB Function: '<S2>/Boundingbox_trans' incorporates:
  //   Inport: '<Root>/BB_all_XY'

  rtb_Gain1 = rtb_X[2];
  memcpy(&rtb_V_boundingbox[0], &rtU.BB_all_XY[0], 400U * sizeof(real_T));
  n = 0;
  do {
    exitg2 = 0;
    shortest_distance_tmp = (int32_T)rtU.BB_num - 1;
    if (n <= shortest_distance_tmp) {
      end_heading = (1.0 + (real_T)n) * 2.0;
      for (i = 0; i < 4; i++) {
        ajj = std::sin(rtb_Gain1);
        table_tmp = std::cos(rtb_Gain1);
        rtb_V_boundingbox[((int32_T)(end_heading + -1.0) + 100 * i) - 1] =
          (rtU.BB_all_XY[((int32_T)(end_heading + -1.0) + 100 * i) - 1] *
           table_tmp + rtU.BB_all_XY[(100 * i + (int32_T)end_heading) - 1] *
           -ajj) + rtb_X[0];
        rtb_V_boundingbox[((int32_T)end_heading + 100 * i) - 1] =
          (rtU.BB_all_XY[((int32_T)(end_heading + -1.0) + 100 * i) - 1] * ajj +
           rtU.BB_all_XY[(100 * i + (int32_T)end_heading) - 1] * table_tmp) +
          rtb_X[1];
      }

      n++;
    } else {
      exitg2 = 1;
    }
  } while (exitg2 == 0);

  // MATLAB Function: '<S3>/SLAM_UKF_MM' incorporates:
  //   Constant: '<S3>/Constant4'

  for (i = 0; i < 188; i++) {
    if (rtConstP.pooled2[940 + i] == (rtInf)) {
      table[i] = rtConstP.pooled2[188 + i];
      table[188 + i] = rtb_X[1];
    } else if (rtConstP.pooled2[940 + i] == 0.0) {
      table[i] = rtb_X[0];
      table[188 + i] = rtConstP.pooled2[376 + i];
    } else {
      rtb_Gain_p = -1.0 / rtConstP.pooled2[940 + i];
      ajj = rtb_X[1] - rtb_Gain_p * rtb_X[0];
      table_tmp = rtConstP.pooled2[940 + i] - rtb_Gain_p;
      table[i] = (ajj - rtConstP.pooled2[1128 + i]) / table_tmp;
      table[188 + i] = (rtConstP.pooled2[940 + i] * ajj - rtConstP.pooled2[1128
                        + i] * rtb_Gain_p) / table_tmp;
    }

    offset_2 = table[i] - rtb_X[0];
    offset_3 = table[188 + i] - rtb_X[1];
    shortest_distance[i] = std::sqrt(offset_2 * offset_2 + offset_3 * offset_3);
  }

  rtb_X_state_0[0] = rtb_X[0];
  rtb_X_state_0[1] = rtb_X[1];
  MM(rtb_X[2] * 180.0 / 3.1415926535897931, rtb_X_state_0, table,
     shortest_distance, rtConstP.pooled2, &ajj, &rtb_Gain_p, rtb_Oi_near_l,
     &table_tmp, &rtb_Look_ahead_S0, &head_err, rtb_num_lane_direction_f,
     &total_length);

  // End of MATLAB Function: '<S3>/SLAM_UKF_MM'

  // Gain: '<S2>/Gain3'
  rtb_Gain1 = 57.295779513082323 * rtb_X[2];

  // MATLAB Function: '<S2>/MM' incorporates:
  //   Gain: '<S2>/Gain'

  idx = rtDW.SFunction_DIMS4_h[0];
  oi_xy_size[0] = rtDW.SFunction_DIMS4_h[0];
  oi_xy_size[1] = 2;
  d_ix = (rtDW.SFunction_DIMS4_h[0] << 1) - 1;
  if (0 <= d_ix) {
    memset(&oi_xy_data[0], 0, (d_ix + 1) * sizeof(real_T));
  }

  b_colj = rtDW.SFunction_DIMS4_h[0];
  if (0 <= idx - 1) {
    memset(&dist_op_data[0], 0, idx * sizeof(real_T));
  }

  for (i = 0; i < rtDW.SFunction_DIMS4_h[0]; i++) {
    if (rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 5 + i] == (rtInf)) {
      oi_xy_data[i] = rtDW.Static_Path_0[i + rtDW.SFunction_DIMS4_h[0]];
      oi_xy_data[i + idx] = rtb_X[1];
    } else if (rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 5 + i] == 0.0) {
      oi_xy_data[i] = rtb_X[0];
      oi_xy_data[i + idx] = rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] << 1)
        + i];
    } else {
      rtb_Gain_p = -1.0 / rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 5 + i];
      ajj = rtb_X[1] - rtb_Gain_p * rtb_X[0];
      table_tmp = rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 5 + i] -
        rtb_Gain_p;
      oi_xy_data[i] = (ajj - rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 6 +
                       i]) / table_tmp;
      oi_xy_data[i + idx] = (rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 5 +
        i] * ajj - rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 6 + i] *
        rtb_Gain_p) / table_tmp;
    }
  }

  for (idx = 0; idx < oi_xy_size[0]; idx++) {
    offset_2 = oi_xy_data[idx] - rtb_X[0];
    offset_3 = oi_xy_data[idx + oi_xy_size[0]] - rtb_X[1];
    dist_op_data[idx] = std::sqrt(offset_2 * offset_2 + offset_3 * offset_3);
  }

  rtb_X_state_0[0] = rtb_X[0];
  rtb_X_state_0[1] = rtb_X[1];
  MM_f(0.017453292519943295 * rtb_Gain1 * 180.0 / 3.1415926535897931,
       rtb_X_state_0, oi_xy_data, oi_xy_size, dist_op_data, &b_colj,
       rtDW.Static_Path_0, rtDW.SFunction_DIMS4_h, &ajj, &rtb_Gain_p,
       rtb_Oi_near_l, &table_tmp, &rtb_Look_ahead_S0, &head_err,
       rtb_num_lane_direction_f, &total_length);
  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    x_data[iy] = (rtDW.Static_Path_0[iy] == ajj);
  }

  idx = 0;
  b_x_0 = x_data[0];
  for (b_idx = 1; b_idx < rtDW.SFunction_DIMS4_h[0]; b_idx++) {
    if ((int32_T)b_x_0 < (int32_T)x_data[b_idx]) {
      b_x_0 = x_data[b_idx];
      idx = b_idx;
    }
  }

  // MATLAB Function: '<S2>/Forward_Length_Decision' incorporates:
  //   Inport: '<Root>/Look_ahead_time'
  //   Inport: '<Root>/Speed_mps'
  //   MATLAB Function: '<S2>/MATLAB Function'

  rtb_Gain_p = rtU.Speed_mps * rtU.Look_ahead_time + 3.0;

  // MATLAB Function: '<S2>/Forward_Seg' incorporates:
  //   MATLAB Function: '<S2>/MM'

  xy_ends_POS_size_idx_0 = rtDW.SFunction_DIMS4_h[0];
  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    rtDW.xy_ends_POS_data[iy] = rtDW.Static_Path_0[iy + rtDW.SFunction_DIMS4_h[0]];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    rtDW.xy_ends_POS_data[iy + xy_ends_POS_size_idx_0] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS4_h[0] << 1) + iy];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    rtDW.xy_ends_POS_data[iy + (xy_ends_POS_size_idx_0 << 1)] =
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 3 + iy];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    rtDW.xy_ends_POS_data[iy + xy_ends_POS_size_idx_0 * 3] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS4_h[0] << 2) + iy];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  if (0 <= d_ix - 1) {
    memcpy(&dist_op_data[0], &rtDW.Static_Path_0[0], d_ix * sizeof(real_T));
  }

  if (rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 3 +
                          rtDW.SFunction_DIMS4_h[0]) - 1] ==
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0]]) {
    b_colj = (rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] << 2) +
               rtDW.SFunction_DIMS4_h[0]) - 1] ==
              rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] << 1]);
  } else {
    b_colj = 0;
  }

  i = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < i; iy++) {
    x_data[iy] = (dist_op_data[iy] == ajj);
  }

  d_idx = 1;
  b_x_0 = x_data[0];
  for (b_idx = 2; b_idx <= i; b_idx++) {
    if ((int32_T)b_x_0 < (int32_T)x_data[b_idx - 1]) {
      b_x_0 = x_data[b_idx - 1];
      d_idx = b_idx;
    }
  }

  offset_2 = rtb_Oi_near_l[0] - rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] *
    3 + d_idx) - 1];
  offset_3 = rtb_Oi_near_l[1] - rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] <<
    2) + d_idx) - 1];
  total_length = std::sqrt(offset_2 * offset_2 + offset_3 * offset_3);
  jj = d_idx;
  n = 0;
  b_idx = 0;
  i = 0;
  exitg1 = false;
  while ((!exitg1) && (i <= rtDW.SFunction_DIMS4_h[0] - 1)) {
    if (total_length > rtb_Gain_p) {
      b_idx = jj;
      exitg1 = true;
    } else {
      iy = d_idx + i;
      Path_RES_0_size_idx_1 = iy + 1;
      if (Path_RES_0_size_idx_1 <= rtDW.SFunction_DIMS4_h[0]) {
        total_length += rtDW.Static_Path_0[iy + (rtDW.SFunction_DIMS4_h[0] << 3)];
        jj = Path_RES_0_size_idx_1;
        n = 1;
        i++;
      } else if (b_colj == 1) {
        iy -= rtDW.SFunction_DIMS4_h[0];
        total_length += rtDW.Static_Path_0[iy + (rtDW.SFunction_DIMS4_h[0] << 3)];
        jj = iy + 1;
        n = 2;
        i++;
      } else {
        b_idx = jj;
        n = 3;
        exitg1 = true;
      }
    }
  }

  i = rtDW.SFunction_DIMS4_h[0];
  if (0 <= i - 1) {
    memset(&Forward_Static_Path_id_0_data[0], 0, i * sizeof(real_T));
  }

  if ((n == 1) || (n == 0)) {
    if (d_idx > b_idx) {
      jj = 0;
      b_colj = 0;
    } else {
      jj = d_idx - 1;
      b_colj = b_idx;
    }

    Path_RES_0_size_idx_1 = b_colj - jj;
    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy] = rtDW.xy_ends_POS_data[jj + iy];
    }

    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(jj + iy) + xy_ends_POS_size_idx_0];
    }

    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + (Path_RES_0_size_idx_1 << 1)] =
        rtDW.xy_ends_POS_data[(jj + iy) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[(jj + iy) + xy_ends_POS_size_idx_0 * 3];
    }

    if (d_idx > b_idx) {
      n = 1;
      iy = 0;
    } else {
      n = d_idx;
      iy = b_idx;
    }

    d_ix = iy - n;
    for (iy = 0; iy <= d_ix; iy++) {
      Forward_Static_Path_id_0_data[iy] = dist_op_data[(n + iy) - 1];
    }

    if (d_idx > b_idx) {
      d_idx = 1;
      b_idx = 0;
    }

    b_idx = (b_idx - d_idx) + 1;
  } else if (n == 2) {
    if (d_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_colj = 0;
      i = 0;
    } else {
      b_colj = d_idx - 1;
      i = rtDW.SFunction_DIMS4_h[0];
    }

    if (1 > b_idx) {
      d_ix = 0;
    } else {
      d_ix = b_idx;
    }

    n = i - b_colj;
    Path_RES_0_size_idx_1 = n + d_ix;
    for (iy = 0; iy < n; iy++) {
      rtDW.Static_Path_ends_POS_data[iy] = rtDW.xy_ends_POS_data[b_colj + iy];
    }

    for (iy = 0; iy < n; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(b_colj + iy) + xy_ends_POS_size_idx_0];
    }

    for (iy = 0; iy < n; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + (Path_RES_0_size_idx_1 << 1)] =
        rtDW.xy_ends_POS_data[(b_colj + iy) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (iy = 0; iy < n; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[(b_colj + iy) + xy_ends_POS_size_idx_0 * 3];
    }

    for (iy = 0; iy < d_ix; iy++) {
      rtDW.Static_Path_ends_POS_data[(iy + i) - b_colj] =
        rtDW.xy_ends_POS_data[iy];
    }

    for (iy = 0; iy < d_ix; iy++) {
      rtDW.Static_Path_ends_POS_data[((iy + i) - b_colj) + Path_RES_0_size_idx_1]
        = rtDW.xy_ends_POS_data[iy + xy_ends_POS_size_idx_0];
    }

    for (iy = 0; iy < d_ix; iy++) {
      rtDW.Static_Path_ends_POS_data[((iy + i) - b_colj) +
        (Path_RES_0_size_idx_1 << 1)] = rtDW.xy_ends_POS_data
        [(xy_ends_POS_size_idx_0 << 1) + iy];
    }

    for (iy = 0; iy < d_ix; iy++) {
      rtDW.Static_Path_ends_POS_data[((iy + i) - b_colj) + Path_RES_0_size_idx_1
        * 3] = rtDW.xy_ends_POS_data[xy_ends_POS_size_idx_0 * 3 + iy];
    }

    if (d_idx > rtDW.SFunction_DIMS4_h[0]) {
      n = 0;
      jj = 0;
    } else {
      n = d_idx - 1;
      jj = rtDW.SFunction_DIMS4_h[0];
    }

    b_colj = ((rtDW.SFunction_DIMS4_h[0] - d_idx) + b_idx) + 1;
    if (1 > b_colj) {
      tmp_0 = 0;
    } else {
      tmp_0 = (int16_T)b_colj;
    }

    i = tmp_0;
    d_ix = tmp_0 - 1;
    for (iy = 0; iy <= d_ix; iy++) {
      cb_data[iy] = (int16_T)iy;
    }

    if (1 > b_idx) {
      iy = 0;
    } else {
      iy = b_idx;
    }

    d_ix = iy - 1;
    b_colj = jj - n;
    for (iy = 0; iy < b_colj; iy++) {
      table[iy] = dist_op_data[n + iy];
    }

    for (iy = 0; iy <= d_ix; iy++) {
      table[(iy + jj) - n] = dist_op_data[iy];
    }

    for (iy = 0; iy < i; iy++) {
      Forward_Static_Path_id_0_data[cb_data[iy]] = table[iy];
    }

    if (d_idx > rtDW.SFunction_DIMS4_h[0]) {
      d_idx = 1;
      b_colj = 1;
    } else {
      b_colj = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    if (1 > b_idx) {
      b_idx = 0;
    }

    b_idx += b_colj - d_idx;
  } else {
    if (d_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_colj = 0;
      n = 0;
    } else {
      b_colj = d_idx - 1;
      n = rtDW.SFunction_DIMS4_h[0];
    }

    Path_RES_0_size_idx_1 = n - b_colj;
    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy] = rtDW.xy_ends_POS_data[b_colj + iy];
    }

    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(b_colj + iy) + xy_ends_POS_size_idx_0];
    }

    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + (Path_RES_0_size_idx_1 << 1)] =
        rtDW.xy_ends_POS_data[(b_colj + iy) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[(b_colj + iy) + xy_ends_POS_size_idx_0 * 3];
    }

    if (d_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_colj = 1;
      i = 0;
    } else {
      b_colj = d_idx;
      i = rtDW.SFunction_DIMS4_h[0];
    }

    d_ix = i - b_colj;
    for (iy = 0; iy <= d_ix; iy++) {
      Forward_Static_Path_id_0_data[iy] = dist_op_data[(b_colj + iy) - 1];
    }

    if (d_idx > rtDW.SFunction_DIMS4_h[0]) {
      d_idx = 1;
      b_idx = 1;
    } else {
      b_idx = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    b_idx -= d_idx;
  }

  if (1 > b_idx) {
    b_idx = 0;
  }

  rtDW.SFunction_DIMS4_f = b_idx;
  if (0 <= b_idx - 1) {
    memcpy(&shortest_distance[0], &Forward_Static_Path_id_0_data[0], b_idx *
           sizeof(real_T));
  }

  b_idx = Path_RES_0_size_idx_1 + 1;
  d_ix = (b_idx << 1) - 1;
  if (0 <= d_ix) {
    memset(&rtDW.Forward_Static_Path_data[0], 0, (d_ix + 1) * sizeof(real_T));
  }

  d_ix = Path_RES_0_size_idx_1 - 1;
  if (0 <= d_ix) {
    memcpy(&rtDW.Forward_Static_Path_data[0], &rtDW.Static_Path_ends_POS_data[0],
           (d_ix + 1) * sizeof(real_T));
  }

  for (iy = 0; iy <= d_ix; iy++) {
    rtDW.Forward_Static_Path_data[iy + b_idx] =
      rtDW.Static_Path_ends_POS_data[iy + Path_RES_0_size_idx_1];
  }

  iy = Path_RES_0_size_idx_1 - 1;
  rtDW.Forward_Static_Path_data[Path_RES_0_size_idx_1] =
    rtDW.Static_Path_ends_POS_data[(Path_RES_0_size_idx_1 << 1) + iy];
  rtDW.Forward_Static_Path_data[Path_RES_0_size_idx_1 + b_idx] =
    rtDW.Static_Path_ends_POS_data[Path_RES_0_size_idx_1 * 3 + iy];
  rtDW.SFunction_DIMS2_h = b_idx;
  d_ix = b_idx - 1;
  if (0 <= d_ix) {
    memcpy(&rtb_Forward_Static_Path_x_h[0], &rtDW.Forward_Static_Path_data[0],
           (d_ix + 1) * sizeof(real_T));
  }

  rtDW.SFunction_DIMS3_k = b_idx;
  d_ix = b_idx - 1;
  for (iy = 0; iy <= d_ix; iy++) {
    rtb_Forward_Static_Path_y_p[iy] = rtDW.Forward_Static_Path_data[iy + b_idx];
  }

  rtDW.SFunction_DIMS6[0] = rtDW.SFunction_DIMS4_h[0];
  rtDW.SFunction_DIMS6[1] = 1;

  // MATLAB Function: '<S2>/EndPointDecision' incorporates:
  //   MATLAB Function: '<S2>/Forward_Length_Decision'

  xy_ends_POS_size_idx_0 = 20000;
  Path_RES_0_size_idx_1 = 2;
  memset(&rtDW.Path_RES_0_data[0], 0, 40000U * sizeof(real_T));
  memset(&rtDW.Path_RES_0_1[0], 0, 40000U * sizeof(real_T));
  rtb_Look_ahead_S0 = 0.0;
  total_length = 0.0;
  n = 0;
  target_k = std::floor(rtb_Gain_p / 0.1);
  offset_2 = rtb_Forward_Static_Path_x_h[1] - rtb_Forward_Static_Path_x_h[0];
  offset_3 = rtb_Forward_Static_Path_y_p[1] - rtb_Forward_Static_Path_y_p[0];
  Length_1 = std::sqrt(offset_2 * offset_2 + offset_3 * offset_3);
  ang_1 = rt_atan2d_snf(rtb_Forward_Static_Path_y_p[1] -
                        rtb_Forward_Static_Path_y_p[0],
                        rtb_Forward_Static_Path_x_h[1] -
                        rtb_Forward_Static_Path_x_h[0]);
  if (Length_1 > 0.1) {
    Length_1 = rt_roundd_snf(Length_1 / 0.1);
    for (b_colj = 0; b_colj < (int32_T)Length_1; b_colj++) {
      total_length = ((1.0 + (real_T)b_colj) - 1.0) * 0.1;
      rtDW.Path_RES_0_1[b_colj] = total_length * std::cos(ang_1) +
        rtb_Forward_Static_Path_x_h[0];
      rtDW.Path_RES_0_1[20000 + b_colj] = total_length * std::sin(ang_1) +
        rtb_Forward_Static_Path_y_p[0];
      total_length = 1.0 + (real_T)b_colj;
    }
  } else {
    rtDW.Path_RES_0_1[0] = rtb_Forward_Static_Path_x_h[0];
    rtDW.Path_RES_0_1[20000] = rtb_Forward_Static_Path_y_p[0];
    total_length = 1.0;
  }

  if (1.0 > total_length) {
    jj = 0;
  } else {
    jj = (int32_T)total_length;
  }

  Path_RES_1_size_idx_0 = jj;
  if (0 <= jj - 1) {
    memcpy(&rtDW.Path_RES_1_data[0], &rtDW.Path_RES_0_1[0], jj * sizeof(real_T));
  }

  for (iy = 0; iy < jj; iy++) {
    rtDW.Path_RES_1_data[iy + jj] = rtDW.Path_RES_0_1[iy + 20000];
  }

  i = jj;
  for (iy = 0; iy < jj; iy++) {
    rtDW.rtb_X_data[iy] = rtb_X[0] - rtDW.Path_RES_1_data[iy];
  }

  power_ec(rtDW.rtb_X_data, &jj, rtDW.tmp_data, &b_colj);
  for (iy = 0; iy < jj; iy++) {
    rtDW.rtb_X_data[iy] = rtb_X[1] - rtDW.Path_RES_1_data[iy + jj];
  }

  power_ec(rtDW.rtb_X_data, &jj, rtDW.tmp_data_c, &i);
  for (iy = 0; iy < b_colj; iy++) {
    rtDW.ob_distance_data[iy] = rtDW.tmp_data[iy] + rtDW.tmp_data_c[iy];
  }

  if (b_colj <= 2) {
    if (b_colj == 1) {
      d_idx = 0;
    } else if (rtDW.ob_distance_data[0] > rtDW.ob_distance_data[1]) {
      d_idx = 1;
    } else if (rtIsNaN(rtDW.ob_distance_data[0])) {
      if (!rtIsNaN(rtDW.ob_distance_data[1])) {
        iy = 2;
      } else {
        iy = 1;
      }

      d_idx = iy - 1;
    } else {
      d_idx = 0;
    }
  } else {
    if (!rtIsNaN(rtDW.ob_distance_data[0])) {
      d_idx = 0;
    } else {
      d_idx = -1;
      b_idx = 2;
      exitg1 = false;
      while ((!exitg1) && (b_idx <= b_colj)) {
        if (!rtIsNaN(rtDW.ob_distance_data[b_idx - 1])) {
          d_idx = b_idx - 1;
          exitg1 = true;
        } else {
          b_idx++;
        }
      }
    }

    if (d_idx + 1 == 0) {
      d_idx = 0;
    } else {
      offset_2 = rtDW.ob_distance_data[d_idx];
      for (i = d_idx + 1; i < b_colj; i++) {
        if (offset_2 > rtDW.ob_distance_data[i]) {
          offset_2 = rtDW.ob_distance_data[i];
          d_idx = i;
        }
      }
    }
  }

  Length_1 = total_length - (real_T)(d_idx + 1);
  if (rtDW.SFunction_DIMS2_h - 2 >= 1) {
    for (i = 1; i - 1 <= rtDW.SFunction_DIMS2_h - 3; i++) {
      if (n == 0) {
        offset_3 = rtb_Forward_Static_Path_x_h[i + 1] -
          rtb_Forward_Static_Path_x_h[i];
        x_0 = rtb_Forward_Static_Path_y_p[i + 1] - rtb_Forward_Static_Path_y_p[i];
        ang_1 = std::sqrt(offset_3 * offset_3 + x_0 * x_0);
        total_length = rt_atan2d_snf(rtb_Forward_Static_Path_y_p[i + 1] -
          rtb_Forward_Static_Path_y_p[i], rtb_Forward_Static_Path_x_h[i + 1] -
          rtb_Forward_Static_Path_x_h[i]);
        if (ang_1 >= 0.1) {
          ang_1 = rt_roundd_snf(ang_1 / 0.1);
          for (b_colj = 0; b_colj < (int32_T)ang_1; b_colj++) {
            c = ((1.0 + (real_T)b_colj) - 1.0) * 0.1;
            b_idx = (int32_T)((1.0 + (real_T)b_colj) + rtb_Look_ahead_S0);
            rtDW.Path_RES_0_data[b_idx - 1] = c * std::cos(total_length) +
              rtb_Forward_Static_Path_x_h[i];
            rtDW.Path_RES_0_data[b_idx + 19999] = c * std::sin(total_length) +
              rtb_Forward_Static_Path_y_p[i];
          }

          rtb_Look_ahead_S0 += ang_1;
        } else {
          rtDW.Path_RES_0_data[(int32_T)(1.0 + rtb_Look_ahead_S0) - 1] =
            rtb_Forward_Static_Path_x_h[i];
          rtDW.Path_RES_0_data[(int32_T)(1.0 + rtb_Look_ahead_S0) + 19999] =
            rtb_Forward_Static_Path_y_p[i];
          rtb_Look_ahead_S0++;
        }

        if (rtb_Look_ahead_S0 > target_k - Length_1) {
          n = 1;
        }
      }
    }
  } else {
    xy_ends_POS_size_idx_0 = 0;
    Path_RES_0_size_idx_1 = 0;
  }

  total_length = (real_T)(d_idx + 1) + target_k;
  if ((xy_ends_POS_size_idx_0 == 0) || (Path_RES_0_size_idx_1 == 0)) {
    if (total_length <= jj) {
      if (d_idx + 1 > total_length) {
        d_idx = 0;
      }

      d_idx += (int32_T)target_k;
      ajj = rtDW.Path_RES_1_data[d_idx - 1];
      table_tmp = rtDW.Path_RES_1_data[(d_idx + jj) - 1];
      head_err = target_k * 0.1;
    } else {
      if (d_idx + 1 > jj) {
        d_idx = 0;
        n = 0;
      } else {
        n = jj;
      }

      n -= d_idx;
      d_idx += n;
      ajj = rtDW.Path_RES_1_data[d_idx - 1];
      table_tmp = rtDW.Path_RES_1_data[(d_idx + jj) - 1];
      if (n == 0) {
        n = 0;
      } else {
        if (!(n > 2)) {
          n = 2;
        }
      }

      head_err = (real_T)n * 0.1;
    }
  } else {
    if (d_idx + 1 > jj) {
      d_idx = 0;
      b_idx = 0;
    } else {
      b_idx = jj;
    }

    if (1.0 > rtb_Look_ahead_S0) {
      n = 0;
    } else {
      n = (int32_T)rtb_Look_ahead_S0;
    }

    d_ix = b_idx - d_idx;
    if (!(d_ix == 0)) {
      b_colj = 2;
      i = d_ix;
    } else {
      if (!(n == 0)) {
        b_colj = Path_RES_0_size_idx_1;
      } else {
        b_colj = 2;
      }

      i = 0;
    }

    if (!(n == 0)) {
      jj = n;
    } else {
      jj = 0;
    }

    for (iy = 0; iy < d_ix; iy++) {
      rtDW.Path_RES_0_1[iy] = rtDW.Path_RES_1_data[d_idx + iy];
    }

    for (iy = 0; iy < d_ix; iy++) {
      rtDW.Path_RES_0_1[iy + d_ix] = rtDW.Path_RES_1_data[(d_idx + iy) +
        Path_RES_1_size_idx_0];
    }

    d_ix = Path_RES_0_size_idx_1 - 1;
    for (iy = 0; iy <= d_ix; iy++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < n;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_0_data_k[Path_RES_0_size_idx_1 + n * iy] =
          rtDW.Path_RES_0_data[xy_ends_POS_size_idx_0 * iy +
          Path_RES_0_size_idx_1];
      }
    }

    b_idx = i + jj;
    for (iy = 0; iy < b_colj; iy++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < i;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_data[Path_RES_0_size_idx_1 + b_idx * iy] =
          rtDW.Path_RES_0_1[i * iy + Path_RES_0_size_idx_1];
      }
    }

    for (iy = 0; iy < b_colj; iy++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < jj;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_data[(Path_RES_0_size_idx_1 + i) + b_idx * iy] =
          rtDW.Path_RES_0_data_k[jj * iy + Path_RES_0_size_idx_1];
      }
    }

    if (target_k - Length_1 <= rtb_Look_ahead_S0) {
      ajj = rtDW.Path_RES_data[(int32_T)target_k - 1];
      table_tmp = rtDW.Path_RES_data[((int32_T)target_k + b_idx) - 1];
      head_err = target_k * 0.1;
    } else {
      target_k = rtb_Look_ahead_S0 + Length_1;
      d_idx = (int32_T)target_k;
      ajj = rtDW.Path_RES_data[d_idx - 1];
      table_tmp = rtDW.Path_RES_data[(d_idx + b_idx) - 1];
      head_err = target_k * 0.1;
    }
  }

  // MATLAB Function: '<S2>/Forward_Seg1' incorporates:
  //   MATLAB Function: '<S2>/EndPointDecision'
  //   MATLAB Function: '<S2>/Forward_Seg'

  xy_ends_POS_size_idx_0 = rtDW.SFunction_DIMS4_h[0];
  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    rtDW.xy_ends_POS_data[iy] = rtDW.Static_Path_0[iy + rtDW.SFunction_DIMS4_h[0]];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    rtDW.xy_ends_POS_data[iy + xy_ends_POS_size_idx_0] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS4_h[0] << 1) + iy];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    rtDW.xy_ends_POS_data[iy + (xy_ends_POS_size_idx_0 << 1)] =
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 3 + iy];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    rtDW.xy_ends_POS_data[iy + xy_ends_POS_size_idx_0 * 3] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS4_h[0] << 2) + iy];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  if (0 <= d_ix - 1) {
    memcpy(&dist_op_data[0], &rtDW.Static_Path_0[0], d_ix * sizeof(real_T));
  }

  if (rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 3 +
                          rtDW.SFunction_DIMS4_h[0]) - 1] ==
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0]]) {
    b_colj = (rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] << 2) +
               rtDW.SFunction_DIMS4_h[0]) - 1] ==
              rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] << 1]);
  } else {
    b_colj = 0;
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    x_data[iy] = (shortest_distance[rtDW.SFunction_DIMS4_f - 1] ==
                  dist_op_data[iy]);
  }

  d_idx = 1;
  b_x_0 = x_data[0];
  for (b_idx = 2; b_idx <= rtDW.SFunction_DIMS4_h[0]; b_idx++) {
    if ((int32_T)b_x_0 < (int32_T)x_data[b_idx - 1]) {
      b_x_0 = x_data[b_idx - 1];
      d_idx = b_idx;
    }
  }

  offset_2 = ajj - rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 3 + d_idx) -
    1];
  offset_3 = table_tmp - rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] << 2) +
    d_idx) - 1];
  total_length = std::sqrt(offset_2 * offset_2 + offset_3 * offset_3);
  jj = d_idx;
  n = 0;
  b_idx = 0;
  i = 0;
  exitg1 = false;
  while ((!exitg1) && (i <= rtDW.SFunction_DIMS4_h[0] - 1)) {
    if (total_length > 15.0) {
      b_idx = jj;
      exitg1 = true;
    } else {
      iy = d_idx + i;
      Path_RES_0_size_idx_1 = iy + 1;
      if (Path_RES_0_size_idx_1 <= rtDW.SFunction_DIMS4_h[0]) {
        total_length += rtDW.Static_Path_0[iy + (rtDW.SFunction_DIMS4_h[0] << 3)];
        jj = Path_RES_0_size_idx_1;
        n = 1;
        i++;
      } else if (b_colj == 1) {
        iy -= rtDW.SFunction_DIMS4_h[0];
        total_length += rtDW.Static_Path_0[iy + (rtDW.SFunction_DIMS4_h[0] << 3)];
        jj = iy + 1;
        n = 2;
        i++;
      } else {
        b_idx = jj;
        n = 3;
        exitg1 = true;
      }
    }
  }

  b_colj = rtDW.SFunction_DIMS4_h[0] - 1;
  if (0 <= b_colj) {
    memset(&Forward_Static_Path_id_0_data[0], 0, (b_colj + 1) * sizeof(real_T));
  }

  if ((n == 1) || (n == 0)) {
    if (d_idx > b_idx) {
      jj = 0;
      b_colj = 0;
    } else {
      jj = d_idx - 1;
      b_colj = b_idx;
    }

    Path_RES_0_size_idx_1 = b_colj - jj;
    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy] = rtDW.xy_ends_POS_data[jj + iy];
    }

    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(jj + iy) + xy_ends_POS_size_idx_0];
    }

    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + (Path_RES_0_size_idx_1 << 1)] =
        rtDW.xy_ends_POS_data[(jj + iy) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[(jj + iy) + xy_ends_POS_size_idx_0 * 3];
    }

    if (d_idx > b_idx) {
      n = 1;
      iy = 0;
    } else {
      n = d_idx;
      iy = b_idx;
    }

    d_ix = iy - n;
    for (iy = 0; iy <= d_ix; iy++) {
      Forward_Static_Path_id_0_data[iy] = dist_op_data[(n + iy) - 1];
    }

    if (d_idx > b_idx) {
      d_idx = 1;
      b_idx = 0;
    }

    b_idx = (b_idx - d_idx) + 1;
  } else if (n == 2) {
    if (d_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_colj = 0;
      i = 0;
    } else {
      b_colj = d_idx - 1;
      i = rtDW.SFunction_DIMS4_h[0];
    }

    if (1 > b_idx) {
      d_ix = 0;
    } else {
      d_ix = b_idx;
    }

    n = i - b_colj;
    Path_RES_0_size_idx_1 = n + d_ix;
    for (iy = 0; iy < n; iy++) {
      rtDW.Static_Path_ends_POS_data[iy] = rtDW.xy_ends_POS_data[b_colj + iy];
    }

    for (iy = 0; iy < n; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(b_colj + iy) + xy_ends_POS_size_idx_0];
    }

    for (iy = 0; iy < n; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + (Path_RES_0_size_idx_1 << 1)] =
        rtDW.xy_ends_POS_data[(b_colj + iy) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (iy = 0; iy < n; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[(b_colj + iy) + xy_ends_POS_size_idx_0 * 3];
    }

    for (iy = 0; iy < d_ix; iy++) {
      rtDW.Static_Path_ends_POS_data[(iy + i) - b_colj] =
        rtDW.xy_ends_POS_data[iy];
    }

    for (iy = 0; iy < d_ix; iy++) {
      rtDW.Static_Path_ends_POS_data[((iy + i) - b_colj) + Path_RES_0_size_idx_1]
        = rtDW.xy_ends_POS_data[iy + xy_ends_POS_size_idx_0];
    }

    for (iy = 0; iy < d_ix; iy++) {
      rtDW.Static_Path_ends_POS_data[((iy + i) - b_colj) +
        (Path_RES_0_size_idx_1 << 1)] = rtDW.xy_ends_POS_data
        [(xy_ends_POS_size_idx_0 << 1) + iy];
    }

    for (iy = 0; iy < d_ix; iy++) {
      rtDW.Static_Path_ends_POS_data[((iy + i) - b_colj) + Path_RES_0_size_idx_1
        * 3] = rtDW.xy_ends_POS_data[xy_ends_POS_size_idx_0 * 3 + iy];
    }

    if (d_idx > rtDW.SFunction_DIMS4_h[0]) {
      n = 0;
      jj = 0;
    } else {
      n = d_idx - 1;
      jj = rtDW.SFunction_DIMS4_h[0];
    }

    b_colj = ((rtDW.SFunction_DIMS4_h[0] - d_idx) + b_idx) + 1;
    if (1 > b_colj) {
      tmp_0 = 0;
    } else {
      tmp_0 = (int16_T)b_colj;
    }

    i = tmp_0;
    d_ix = tmp_0 - 1;
    for (iy = 0; iy <= d_ix; iy++) {
      cb_data[iy] = (int16_T)iy;
    }

    if (1 > b_idx) {
      iy = 0;
    } else {
      iy = b_idx;
    }

    d_ix = iy - 1;
    b_colj = jj - n;
    for (iy = 0; iy < b_colj; iy++) {
      table[iy] = dist_op_data[n + iy];
    }

    for (iy = 0; iy <= d_ix; iy++) {
      table[(iy + jj) - n] = dist_op_data[iy];
    }

    for (iy = 0; iy < i; iy++) {
      Forward_Static_Path_id_0_data[cb_data[iy]] = table[iy];
    }

    if (d_idx > rtDW.SFunction_DIMS4_h[0]) {
      d_idx = 1;
      b_colj = 1;
    } else {
      b_colj = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    if (1 > b_idx) {
      b_idx = 0;
    }

    b_idx += b_colj - d_idx;
  } else {
    if (d_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_colj = 0;
      n = 0;
    } else {
      b_colj = d_idx - 1;
      n = rtDW.SFunction_DIMS4_h[0];
    }

    Path_RES_0_size_idx_1 = n - b_colj;
    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy] = rtDW.xy_ends_POS_data[b_colj + iy];
    }

    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(b_colj + iy) + xy_ends_POS_size_idx_0];
    }

    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + (Path_RES_0_size_idx_1 << 1)] =
        rtDW.xy_ends_POS_data[(b_colj + iy) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (iy = 0; iy < Path_RES_0_size_idx_1; iy++) {
      rtDW.Static_Path_ends_POS_data[iy + Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[(b_colj + iy) + xy_ends_POS_size_idx_0 * 3];
    }

    if (d_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_colj = 1;
      i = 0;
    } else {
      b_colj = d_idx;
      i = rtDW.SFunction_DIMS4_h[0];
    }

    d_ix = i - b_colj;
    for (iy = 0; iy <= d_ix; iy++) {
      Forward_Static_Path_id_0_data[iy] = dist_op_data[(b_colj + iy) - 1];
    }

    if (d_idx > rtDW.SFunction_DIMS4_h[0]) {
      d_idx = 1;
      b_idx = 1;
    } else {
      b_idx = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    b_idx -= d_idx;
  }

  if (1 > b_idx) {
    b_idx = 0;
  }

  d_idx = Path_RES_0_size_idx_1 + 1;
  d_ix = (d_idx << 1) - 1;
  if (0 <= d_ix) {
    memset(&rtDW.Forward_Static_Path_data_m[0], 0, (d_ix + 1) * sizeof(real_T));
  }

  d_ix = Path_RES_0_size_idx_1 - 1;
  if (0 <= d_ix) {
    memcpy(&rtDW.Forward_Static_Path_data_m[0], &rtDW.Static_Path_ends_POS_data
           [0], (d_ix + 1) * sizeof(real_T));
  }

  for (iy = 0; iy <= d_ix; iy++) {
    rtDW.Forward_Static_Path_data_m[iy + d_idx] =
      rtDW.Static_Path_ends_POS_data[iy + Path_RES_0_size_idx_1];
  }

  iy = Path_RES_0_size_idx_1 - 1;
  rtDW.Forward_Static_Path_data_m[Path_RES_0_size_idx_1] =
    rtDW.Static_Path_ends_POS_data[(Path_RES_0_size_idx_1 << 1) + iy];
  rtDW.Forward_Static_Path_data_m[Path_RES_0_size_idx_1 + d_idx] =
    rtDW.Static_Path_ends_POS_data[Path_RES_0_size_idx_1 * 3 + iy];
  rtDW.SFunction_DIMS2 = d_idx;
  d_ix = d_idx - 1;
  if (0 <= d_ix) {
    memcpy(&rtb_Forward_Static_Path_x_h[0], &rtDW.Forward_Static_Path_data_m[0],
           (d_ix + 1) * sizeof(real_T));
  }

  rtDW.SFunction_DIMS3 = d_idx;
  d_ix = d_idx - 1;
  for (iy = 0; iy <= d_ix; iy++) {
    rtb_Forward_Static_Path_y_p[iy] = rtDW.Forward_Static_Path_data_m[iy + d_idx];
  }

  rtDW.SFunction_DIMS4 = b_idx;
  if (0 <= b_idx - 1) {
    memcpy(&rtb_Forward_Static_Path_id[0], &Forward_Static_Path_id_0_data[0],
           b_idx * sizeof(real_T));
  }

  // End of MATLAB Function: '<S2>/Forward_Seg1'

  // MATLAB Function: '<S2>/EndPointDecision1' incorporates:
  //   MATLAB Function: '<S2>/EndPointDecision'

  xy_ends_POS_size_idx_0 = 20000;
  Path_RES_0_size_idx_1 = 2;
  memset(&rtDW.Path_RES_0_data[0], 0, 40000U * sizeof(real_T));
  memset(&rtDW.Path_RES_0_1[0], 0, 40000U * sizeof(real_T));
  rtb_Look_ahead_S0 = 0.0;
  total_length = 0.0;
  n = 0;
  offset_2 = rtb_Forward_Static_Path_x_h[1] - rtb_Forward_Static_Path_x_h[0];
  offset_3 = rtb_Forward_Static_Path_y_p[1] - rtb_Forward_Static_Path_y_p[0];
  Length_1 = std::sqrt(offset_2 * offset_2 + offset_3 * offset_3);
  ang_1 = rt_atan2d_snf(rtb_Forward_Static_Path_y_p[1] -
                        rtb_Forward_Static_Path_y_p[0],
                        rtb_Forward_Static_Path_x_h[1] -
                        rtb_Forward_Static_Path_x_h[0]);
  if (Length_1 > 0.1) {
    Length_1 = rt_roundd_snf(Length_1 / 0.1);
    for (b_colj = 0; b_colj < (int32_T)Length_1; b_colj++) {
      total_length = ((1.0 + (real_T)b_colj) - 1.0) * 0.1;
      rtDW.Path_RES_0_1[b_colj] = total_length * std::cos(ang_1) +
        rtb_Forward_Static_Path_x_h[0];
      rtDW.Path_RES_0_1[20000 + b_colj] = total_length * std::sin(ang_1) +
        rtb_Forward_Static_Path_y_p[0];
      total_length = 1.0 + (real_T)b_colj;
    }
  } else {
    rtDW.Path_RES_0_1[0] = rtb_Forward_Static_Path_x_h[0];
    rtDW.Path_RES_0_1[20000] = rtb_Forward_Static_Path_y_p[0];
    total_length = 1.0;
  }

  if (1.0 > total_length) {
    jj = 0;
  } else {
    jj = (int32_T)total_length;
  }

  Path_RES_1_size_idx_0 = jj;
  if (0 <= jj - 1) {
    memcpy(&rtDW.Path_RES_1_data[0], &rtDW.Path_RES_0_1[0], jj * sizeof(real_T));
  }

  for (iy = 0; iy < jj; iy++) {
    rtDW.Path_RES_1_data[iy + jj] = rtDW.Path_RES_0_1[iy + 20000];
  }

  for (iy = 0; iy < jj; iy++) {
    rtDW.tmp_data_c[iy] = ajj - rtDW.Path_RES_1_data[iy];
  }

  power_jb(rtDW.tmp_data_c, &jj, rtDW.tmp_data, &b_colj);
  for (iy = 0; iy < jj; iy++) {
    rtDW.rtb_X_data[iy] = table_tmp - rtDW.Path_RES_1_data[iy + jj];
  }

  power_jb(rtDW.rtb_X_data, &jj, rtDW.tmp_data_c, &i);
  for (iy = 0; iy < b_colj; iy++) {
    rtDW.ob_distance_data[iy] = rtDW.tmp_data[iy] + rtDW.tmp_data_c[iy];
  }

  if (b_colj <= 2) {
    if (b_colj == 1) {
      d_idx = 0;
    } else if (rtDW.ob_distance_data[0] > rtDW.ob_distance_data[1]) {
      d_idx = 1;
    } else if (rtIsNaN(rtDW.ob_distance_data[0])) {
      if (!rtIsNaN(rtDW.ob_distance_data[1])) {
        iy = 2;
      } else {
        iy = 1;
      }

      d_idx = iy - 1;
    } else {
      d_idx = 0;
    }
  } else {
    if (!rtIsNaN(rtDW.ob_distance_data[0])) {
      d_idx = 0;
    } else {
      d_idx = -1;
      b_idx = 2;
      exitg1 = false;
      while ((!exitg1) && (b_idx <= b_colj)) {
        if (!rtIsNaN(rtDW.ob_distance_data[b_idx - 1])) {
          d_idx = b_idx - 1;
          exitg1 = true;
        } else {
          b_idx++;
        }
      }
    }

    if (d_idx + 1 == 0) {
      d_idx = 0;
    } else {
      offset_2 = rtDW.ob_distance_data[d_idx];
      for (i = d_idx + 1; i < b_colj; i++) {
        if (offset_2 > rtDW.ob_distance_data[i]) {
          offset_2 = rtDW.ob_distance_data[i];
          d_idx = i;
        }
      }
    }
  }

  Length_1 = total_length - (real_T)(d_idx + 1);
  if (rtDW.SFunction_DIMS2 - 2 >= 1) {
    for (i = 1; i - 1 <= rtDW.SFunction_DIMS2 - 3; i++) {
      if (n == 0) {
        offset_3 = rtb_Forward_Static_Path_x_h[i + 1] -
          rtb_Forward_Static_Path_x_h[i];
        x_0 = rtb_Forward_Static_Path_y_p[i + 1] - rtb_Forward_Static_Path_y_p[i];
        ang_1 = std::sqrt(offset_3 * offset_3 + x_0 * x_0);
        total_length = rt_atan2d_snf(rtb_Forward_Static_Path_y_p[i + 1] -
          rtb_Forward_Static_Path_y_p[i], rtb_Forward_Static_Path_x_h[i + 1] -
          rtb_Forward_Static_Path_x_h[i]);
        if (ang_1 >= 0.1) {
          ang_1 = rt_roundd_snf(ang_1 / 0.1);
          for (b_colj = 0; b_colj < (int32_T)ang_1; b_colj++) {
            c = ((1.0 + (real_T)b_colj) - 1.0) * 0.1;
            b_idx = (int32_T)((1.0 + (real_T)b_colj) + rtb_Look_ahead_S0);
            rtDW.Path_RES_0_data[b_idx - 1] = c * std::cos(total_length) +
              rtb_Forward_Static_Path_x_h[i];
            rtDW.Path_RES_0_data[b_idx + 19999] = c * std::sin(total_length) +
              rtb_Forward_Static_Path_y_p[i];
          }

          rtb_Look_ahead_S0 += ang_1;
        } else {
          rtDW.Path_RES_0_data[(int32_T)(1.0 + rtb_Look_ahead_S0) - 1] =
            rtb_Forward_Static_Path_x_h[i];
          rtDW.Path_RES_0_data[(int32_T)(1.0 + rtb_Look_ahead_S0) + 19999] =
            rtb_Forward_Static_Path_y_p[i];
          rtb_Look_ahead_S0++;
        }

        if (rtb_Look_ahead_S0 > 150.0 - Length_1) {
          n = 1;
        }
      }
    }
  } else {
    xy_ends_POS_size_idx_0 = 0;
    Path_RES_0_size_idx_1 = 0;
  }

  total_length = (real_T)(d_idx + 1) + 150.0;
  if ((xy_ends_POS_size_idx_0 == 0) || (Path_RES_0_size_idx_1 == 0)) {
    if (total_length <= jj) {
      if (d_idx + 1 > (int32_T)total_length) {
        d_idx = 0;
      }

      total_length = rtDW.Path_RES_1_data[d_idx + 149];
      ang_1 = rtDW.Path_RES_1_data[(d_idx + jj) + 149];
      rtb_Look_ahead_S0 = 15.0;
    } else {
      if (d_idx + 1 > jj) {
        d_idx = 0;
        n = 0;
      } else {
        n = jj;
      }

      n -= d_idx;
      iy = n + d_idx;
      total_length = rtDW.Path_RES_1_data[iy - 1];
      ang_1 = rtDW.Path_RES_1_data[(iy + jj) - 1];
      if (n == 0) {
        n = 0;
      } else {
        if (!(n > 2)) {
          n = 2;
        }
      }

      rtb_Look_ahead_S0 = (real_T)n * 0.1;
    }
  } else {
    if (d_idx + 1 > jj) {
      d_idx = 0;
      b_idx = 0;
    } else {
      b_idx = jj;
    }

    if (1.0 > rtb_Look_ahead_S0) {
      n = 0;
    } else {
      n = (int32_T)rtb_Look_ahead_S0;
    }

    d_ix = b_idx - d_idx;
    if (!(d_ix == 0)) {
      b_colj = 2;
      i = d_ix;
    } else {
      if (!(n == 0)) {
        b_colj = Path_RES_0_size_idx_1;
      } else {
        b_colj = 2;
      }

      i = 0;
    }

    if (!(n == 0)) {
      jj = n;
    } else {
      jj = 0;
    }

    for (iy = 0; iy < d_ix; iy++) {
      rtDW.Path_RES_0_1[iy] = rtDW.Path_RES_1_data[d_idx + iy];
    }

    for (iy = 0; iy < d_ix; iy++) {
      rtDW.Path_RES_0_1[iy + d_ix] = rtDW.Path_RES_1_data[(d_idx + iy) +
        Path_RES_1_size_idx_0];
    }

    d_ix = Path_RES_0_size_idx_1 - 1;
    for (iy = 0; iy <= d_ix; iy++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < n;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_0_data_k[Path_RES_0_size_idx_1 + n * iy] =
          rtDW.Path_RES_0_data[xy_ends_POS_size_idx_0 * iy +
          Path_RES_0_size_idx_1];
      }
    }

    b_idx = i + jj;
    for (iy = 0; iy < b_colj; iy++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < i;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_data[Path_RES_0_size_idx_1 + b_idx * iy] =
          rtDW.Path_RES_0_1[i * iy + Path_RES_0_size_idx_1];
      }
    }

    for (iy = 0; iy < b_colj; iy++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < jj;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_data[(Path_RES_0_size_idx_1 + i) + b_idx * iy] =
          rtDW.Path_RES_0_data_k[jj * iy + Path_RES_0_size_idx_1];
      }
    }

    if (150.0 - Length_1 <= rtb_Look_ahead_S0) {
      total_length = rtDW.Path_RES_data[149];
      ang_1 = rtDW.Path_RES_data[b_idx + 149];
      rtb_Look_ahead_S0 = 15.0;
    } else {
      target_k = rtb_Look_ahead_S0 + Length_1;
      iy = (int32_T)target_k;
      total_length = rtDW.Path_RES_data[iy - 1];
      ang_1 = rtDW.Path_RES_data[(iy + b_idx) - 1];
      rtb_Look_ahead_S0 = target_k * 0.1;
    }
  }

  // MATLAB Function: '<S2>/DangerousArea' incorporates:
  //   MATLAB Function: '<S2>/EndPointDecision'
  //   MATLAB Function: '<S2>/Forward_Length_Decision'
  //   UnitDelay: '<S2>/Unit Delay10'
  //   UnitDelay: '<S2>/Unit Delay12'
  //   UnitDelay: '<S2>/Unit Delay8'
  //   UnitDelay: '<S2>/Unit Delay9'

  Length_1 = rtDW.UnitDelay8_DSTATE;
  rtb_num_lane_direction_f[0] = rtDW.UnitDelay9_DSTATE[0];
  rtb_H_y_out[0] = rtDW.UnitDelay10_DSTATE[0];
  rtb_num_lane_direction_f[1] = rtDW.UnitDelay9_DSTATE[1];
  rtb_H_y_out[1] = rtDW.UnitDelay10_DSTATE[1];
  rtb_num_lane_direction_f[2] = rtDW.UnitDelay9_DSTATE[2];
  rtb_H_y_out[2] = rtDW.UnitDelay10_DSTATE[2];
  rtb_num_lane_direction_f[3] = rtDW.UnitDelay9_DSTATE[3];
  rtb_H_y_out[3] = rtDW.UnitDelay10_DSTATE[3];
  jj = 0;
  x_0 = rtb_X[0];
  target_k = rtb_X[1];
  d_ix = rtDW.SFunction_DIMS4_f * rtDW.SFunction_DIMS4_h[1] - 1;
  if (0 <= d_ix) {
    memset(&rtDW.Forward_Static_Path_0_data[0], 0, (d_ix + 1) * sizeof(real_T));
  }

  for (i = 0; i < rtDW.SFunction_DIMS4_f; i++) {
    d_ix = rtDW.SFunction_DIMS4_h[0];
    for (iy = 0; iy < d_ix; iy++) {
      x_data[iy] = (shortest_distance[i] == rtDW.Static_Path_0[iy]);
    }

    d_idx = 0;
    b_x_0 = x_data[0];
    for (b_colj = 1; b_colj < rtDW.SFunction_DIMS4_h[0]; b_colj++) {
      if ((int32_T)b_x_0 < (int32_T)x_data[b_colj]) {
        b_x_0 = x_data[b_colj];
        d_idx = b_colj;
      }
    }

    d_ix = rtDW.SFunction_DIMS4_h[1];
    for (iy = 0; iy < d_ix; iy++) {
      rtDW.Forward_Static_Path_0_data[i + rtDW.SFunction_DIMS4_f * iy] =
        rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * iy + d_idx];
    }
  }

  n = 0;
  exitg1 = false;
  while ((!exitg1) && (n <= shortest_distance_tmp)) {
    end_heading = (1.0 + (real_T)n) * 2.0;
    for (iy = 0; iy < 4; iy++) {
      OBXY_m[iy << 1] = rtb_V_boundingbox[((int32_T)(end_heading + -1.0) + 100 *
        iy) - 1];
      OBXY_m[1 + (iy << 1)] = rtb_V_boundingbox[(100 * iy + (int32_T)end_heading)
        - 1];
    }

    b_colj = 0;
    exitg3 = false;
    while ((!exitg3) && (b_colj <= rtDW.SFunction_DIMS4_f - 1)) {
      offset_2 = rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f << 2) +
        b_colj] - rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f << 1)
        + b_colj];
      Length_1 = rtDW.Forward_Static_Path_0_data[b_colj + rtDW.SFunction_DIMS4_f]
        - rtDW.Forward_Static_Path_0_data[rtDW.SFunction_DIMS4_f * 3 + b_colj];
      c = (rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f << 2) +
           b_colj] - rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f <<
            1) + b_colj]) * -rtDW.Forward_Static_Path_0_data[b_colj +
        rtDW.SFunction_DIMS4_f] +
        (rtDW.Forward_Static_Path_0_data[rtDW.SFunction_DIMS4_f * 3 + b_colj] -
         rtDW.Forward_Static_Path_0_data[b_colj + rtDW.SFunction_DIMS4_f]) *
        rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f << 1) + b_colj];
      offset_4 = Length_1 * Length_1;
      offset_3 = std::sqrt(offset_2 * offset_2 + offset_4);
      ex[0] = (offset_2 * OBXY_m[0] + Length_1 * OBXY_m[1]) + c;
      ex[1] = (offset_2 * OBXY_m[2] + Length_1 * OBXY_m[3]) + c;
      ex[2] = (offset_2 * OBXY_m[4] + Length_1 * OBXY_m[5]) + c;
      ex[3] = (offset_2 * OBXY_m[6] + Length_1 * OBXY_m[7]) + c;
      abs_g(ex, rtb_num_lane_direction_b);
      K1[0] = rtb_num_lane_direction_b[0] / offset_3;
      K1[1] = rtb_num_lane_direction_b[1] / offset_3;
      K1[2] = rtb_num_lane_direction_b[2] / offset_3;
      K1[3] = rtb_num_lane_direction_b[3] / offset_3;
      offset_3 = offset_2 * Length_1;
      x_endpoint1 = offset_2 * offset_2 + offset_4;
      y_endpoint1 = offset_2 * c;
      rtb_num_lane_direction_f[0] = ((offset_4 * OBXY_m[0] - offset_3 * OBXY_m[1])
        - y_endpoint1) / x_endpoint1;
      rtb_num_lane_direction_f[1] = ((offset_4 * OBXY_m[2] - offset_3 * OBXY_m[3])
        - y_endpoint1) / x_endpoint1;
      rtb_num_lane_direction_f[2] = ((offset_4 * OBXY_m[4] - offset_3 * OBXY_m[5])
        - y_endpoint1) / x_endpoint1;
      rtb_num_lane_direction_f[3] = ((offset_4 * OBXY_m[6] - offset_3 * OBXY_m[7])
        - y_endpoint1) / x_endpoint1;
      offset_3 = -offset_2 * Length_1;
      end_heading = offset_2 * offset_2;
      x_endpoint1 = offset_2 * offset_2 + offset_4;
      y_endpoint1 = Length_1 * c;
      rtb_H_y_out[0] = ((offset_3 * OBXY_m[0] + end_heading * OBXY_m[1]) -
                        y_endpoint1) / x_endpoint1;
      rtb_H_y_out[1] = ((offset_3 * OBXY_m[2] + end_heading * OBXY_m[3]) -
                        y_endpoint1) / x_endpoint1;
      rtb_H_y_out[2] = ((offset_3 * OBXY_m[4] + end_heading * OBXY_m[5]) -
                        y_endpoint1) / x_endpoint1;
      rtb_H_y_out[3] = ((offset_3 * OBXY_m[6] + end_heading * OBXY_m[7]) -
                        y_endpoint1) / x_endpoint1;
      rtb_Oi_near_l[0] = ((offset_4 * x_0 - offset_2 * Length_1 * target_k) -
                          offset_2 * c) / (offset_2 * offset_2 + offset_4);
      end_heading = ((-offset_2 * Length_1 * x_0 + offset_2 * offset_2 *
                      target_k) - Length_1 * c) / (offset_2 * offset_2 +
        offset_4);
      b_x_0 = rtIsNaN(rtb_num_lane_direction_f[0]);
      if (!b_x_0) {
        d_idx = 1;
      } else {
        d_idx = 0;
        b_idx = 2;
        exitg4 = false;
        while ((!exitg4) && (b_idx < 5)) {
          if (!rtIsNaN(rtb_num_lane_direction_f[b_idx - 1])) {
            d_idx = b_idx;
            exitg4 = true;
          } else {
            b_idx++;
          }
        }
      }

      if (d_idx == 0) {
        offset_2 = rtb_num_lane_direction_f[0];
      } else {
        offset_2 = rtb_num_lane_direction_f[d_idx - 1];
        while (d_idx + 1 < 5) {
          if (offset_2 > rtb_num_lane_direction_f[d_idx]) {
            offset_2 = rtb_num_lane_direction_f[d_idx];
          }

          d_idx++;
        }
      }

      if (rtb_Oi_near_l[0] < ajj) {
        Length_1 = ajj;
      } else if (rtIsNaN(rtb_Oi_near_l[0])) {
        if (!rtIsNaN(ajj)) {
          Length_1 = ajj;
        } else {
          Length_1 = rtb_Oi_near_l[0];
        }
      } else {
        Length_1 = rtb_Oi_near_l[0];
      }

      guard1 = false;
      if (offset_2 <= Length_1) {
        if (!b_x_0) {
          b_idx = 1;
        } else {
          b_idx = 0;
          i = 2;
          exitg4 = false;
          while ((!exitg4) && (i < 5)) {
            if (!rtIsNaN(rtb_num_lane_direction_f[i - 1])) {
              b_idx = i;
              exitg4 = true;
            } else {
              i++;
            }
          }
        }

        if (b_idx == 0) {
          offset_2 = rtb_num_lane_direction_f[0];
        } else {
          offset_2 = rtb_num_lane_direction_f[b_idx - 1];
          while (b_idx + 1 < 5) {
            if (offset_2 < rtb_num_lane_direction_f[b_idx]) {
              offset_2 = rtb_num_lane_direction_f[b_idx];
            }

            b_idx++;
          }
        }

        if (rtb_Oi_near_l[0] > ajj) {
          Length_1 = ajj;
        } else if (rtIsNaN(rtb_Oi_near_l[0])) {
          if (!rtIsNaN(ajj)) {
            Length_1 = ajj;
          } else {
            Length_1 = rtb_Oi_near_l[0];
          }
        } else {
          Length_1 = rtb_Oi_near_l[0];
        }

        if (offset_2 >= Length_1) {
          b_x_0 = rtIsNaN(rtb_H_y_out[0]);
          if (!b_x_0) {
            b_idx = 1;
          } else {
            b_idx = 0;
            iy = 2;
            exitg4 = false;
            while ((!exitg4) && (iy < 5)) {
              if (!rtIsNaN(rtb_H_y_out[iy - 1])) {
                b_idx = iy;
                exitg4 = true;
              } else {
                iy++;
              }
            }
          }

          if (b_idx == 0) {
            offset_2 = rtb_H_y_out[0];
          } else {
            offset_2 = rtb_H_y_out[b_idx - 1];
            while (b_idx + 1 < 5) {
              if (offset_2 > rtb_H_y_out[b_idx]) {
                offset_2 = rtb_H_y_out[b_idx];
              }

              b_idx++;
            }
          }

          if (end_heading < table_tmp) {
            Length_1 = table_tmp;
          } else if (rtIsNaN(end_heading)) {
            if (!rtIsNaN(table_tmp)) {
              Length_1 = table_tmp;
            } else {
              Length_1 = end_heading;
            }
          } else {
            Length_1 = end_heading;
          }

          if (offset_2 <= Length_1) {
            if (!b_x_0) {
              d_idx = 1;
            } else {
              d_idx = 0;
              i = 2;
              exitg4 = false;
              while ((!exitg4) && (i < 5)) {
                if (!rtIsNaN(rtb_H_y_out[i - 1])) {
                  d_idx = i;
                  exitg4 = true;
                } else {
                  i++;
                }
              }
            }

            if (d_idx == 0) {
              offset_2 = rtb_H_y_out[0];
            } else {
              offset_2 = rtb_H_y_out[d_idx - 1];
              while (d_idx + 1 < 5) {
                if (offset_2 < rtb_H_y_out[d_idx]) {
                  offset_2 = rtb_H_y_out[d_idx];
                }

                d_idx++;
              }
            }

            if (end_heading > table_tmp) {
              end_heading = table_tmp;
            } else {
              if (rtIsNaN(end_heading) && (!rtIsNaN(table_tmp))) {
                end_heading = table_tmp;
              }
            }

            if (offset_2 >= end_heading) {
              if (!rtIsNaN(K1[0])) {
                d_idx = 1;
              } else {
                d_idx = 0;
                i = 2;
                exitg4 = false;
                while ((!exitg4) && (i < 5)) {
                  if (!rtIsNaN(K1[i - 1])) {
                    d_idx = i;
                    exitg4 = true;
                  } else {
                    i++;
                  }
                }
              }

              if (d_idx == 0) {
                offset_2 = K1[0];
              } else {
                offset_2 = K1[d_idx - 1];
                while (d_idx + 1 < 5) {
                  if (offset_2 > K1[d_idx]) {
                    offset_2 = K1[d_idx];
                  }

                  d_idx++;
                }
              }

              if (offset_2 <=
                  rtDW.Forward_Static_Path_0_data[rtDW.SFunction_DIMS4_f * 10 +
                  b_colj] / 2.0) {
                Length_1 = 1.0;
                jj = 1;
                exitg3 = true;
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
        Length_1 = rtDW.UnitDelay8_DSTATE;
        rtb_num_lane_direction_f[0] = rtDW.UnitDelay9_DSTATE[0];
        rtb_H_y_out[0] = rtDW.UnitDelay10_DSTATE[0];
        rtb_num_lane_direction_f[1] = rtDW.UnitDelay9_DSTATE[1];
        rtb_H_y_out[1] = rtDW.UnitDelay10_DSTATE[1];
        rtb_num_lane_direction_f[2] = rtDW.UnitDelay9_DSTATE[2];
        rtb_H_y_out[2] = rtDW.UnitDelay10_DSTATE[2];
        rtb_num_lane_direction_f[3] = rtDW.UnitDelay9_DSTATE[3];
        rtb_H_y_out[3] = rtDW.UnitDelay10_DSTATE[3];
        b_colj++;
      }
    }

    if (jj == 1) {
      exitg1 = true;
    } else {
      n++;
    }
  }

  if (Length_1 == 1.0) {
    ex[0] = ajj - rtb_num_lane_direction_f[0];
    ex[1] = ajj - rtb_num_lane_direction_f[1];
    ex[2] = ajj - rtb_num_lane_direction_f[2];
    ex[3] = ajj - rtb_num_lane_direction_f[3];
    power_j(ex, rtb_num_lane_direction_b);
    ex[0] = table_tmp - rtb_H_y_out[0];
    ex[1] = table_tmp - rtb_H_y_out[1];
    ex[2] = table_tmp - rtb_H_y_out[2];
    ex[3] = table_tmp - rtb_H_y_out[3];
    power_j(ex, tmp);
    K1[0] = rtb_num_lane_direction_b[0] + tmp[0];
    K1[1] = rtb_num_lane_direction_b[1] + tmp[1];
    K1[2] = rtb_num_lane_direction_b[2] + tmp[2];
    K1[3] = rtb_num_lane_direction_b[3] + tmp[3];
    if (!rtIsNaN(K1[0])) {
      b_idx = 1;
    } else {
      b_idx = 0;
      d_idx = 2;
      exitg1 = false;
      while ((!exitg1) && (d_idx < 5)) {
        if (!rtIsNaN(K1[d_idx - 1])) {
          b_idx = d_idx;
          exitg1 = true;
        } else {
          d_idx++;
        }
      }
    }

    if (b_idx == 0) {
      end_heading = K1[0];
    } else {
      end_heading = K1[b_idx - 1];
      while (b_idx + 1 < 5) {
        if (end_heading > K1[b_idx]) {
          end_heading = K1[b_idx];
        }

        b_idx++;
      }
    }

    if (std::sqrt(end_heading) > rtb_Gain_p + 3.0) {
      Length_1 = 0.0;
    }
  }

  if (Length_1 == 1.0) {
    target_k = 100.0;

    // MATLAB Function: '<S2>/DynamicPathPlanning'
    c = 5.0;
  } else {
    target_k = rtDW.UnitDelay12_DSTATE - 1.0;
    if (rtDW.UnitDelay12_DSTATE - 1.0 < 0.0) {
      target_k = 0.0;
    }

    // MATLAB Function: '<S2>/DynamicPathPlanning' incorporates:
    //   UnitDelay: '<S2>/Unit Delay12'

    c = (100.0 - target_k) * 5.0 / 100.0 + 5.0;
  }

  // SignalConversion: '<S11>/TmpSignal ConversionAt SFunction Inport6' incorporates:
  //   Gain: '<S2>/Gain1'
  //   MATLAB Function: '<S2>/DynamicPathPlanning'

  rtb_TmpSignalConversionAtSFun_0 = 0.017453292519943295 * rtb_Gain1;

  // MATLAB Function: '<S2>/DynamicPathPlanning' incorporates:
  //   Constant: '<S2>/Constant16'
  //   MATLAB Function: '<S2>/EndPointDecision'
  //   MATLAB Function: '<S2>/MM'
  //   SignalConversion: '<S11>/TmpSignal ConversionAt SFunction Inport6'
  //   UnitDelay: '<S2>/Unit Delay5'

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    x_data[iy] = (shortest_distance[rtDW.SFunction_DIMS4_f - 1] ==
                  rtDW.Static_Path_0[iy]);
  }

  b_idx = rtDW.SFunction_DIMS4_h[0] - 1;
  d_idx = 0;
  for (b_colj = 0; b_colj <= b_idx; b_colj++) {
    if (x_data[b_colj]) {
      d_idx++;
    }
  }

  i = d_idx;
  d_idx = 0;
  for (b_colj = 0; b_colj <= b_idx; b_colj++) {
    if (x_data[b_colj]) {
      p_data[d_idx] = b_colj + 1;
      d_idx++;
    }
  }

  for (iy = 0; iy < i; iy++) {
    dist_op_data[iy] = rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 7 +
      p_data[iy]) - 1] * 3.1415926535897931 / 180.0;
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    x_data[iy] = (shortest_distance[rtDW.SFunction_DIMS4_f - 1] ==
                  rtDW.Static_Path_0[iy]);
  }

  d_idx = 0;
  for (b_colj = 0; b_colj < rtDW.SFunction_DIMS4_h[0]; b_colj++) {
    if (x_data[b_colj]) {
      q_data[d_idx] = b_colj + 1;
      d_idx++;
    }
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    x_data[iy] = (shortest_distance[rtDW.SFunction_DIMS4_f - 1] ==
                  rtDW.Static_Path_0[iy]);
  }

  d_idx = 0;
  for (b_colj = 0; b_colj < rtDW.SFunction_DIMS4_h[0]; b_colj++) {
    if (x_data[b_colj]) {
      s_data[d_idx] = b_colj + 1;
      d_idx++;
    }
  }

  x_0 = rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 10 + s_data[0]) - 1] /
    4.0;
  offset_2 = x_0 * 2.0;
  offset_3 = x_0 * 3.0;
  offset_4 = x_0 * 4.0;
  offset_5 = x_0 * 5.0;
  offset_6 = x_0 * 6.0;
  offset[0] = offset_6;
  offset[1] = offset_5;
  offset[2] = offset_4;
  offset[3] = offset_3;
  offset[4] = offset_2;
  offset[5] = x_0;
  offset[6] = 0.0;
  offset[7] = x_0;
  offset[8] = offset_2;
  offset[9] = offset_3;
  offset[10] = offset_4;
  offset[11] = offset_5;
  offset[12] = offset_6;
  x_endpoint6 = std::cos(dist_op_data[0] + 1.5707963267948966);
  x_endpoint1 = x_endpoint6 * offset_6 + ajj;
  y_endpoint6 = std::sin(dist_op_data[0] + 1.5707963267948966);
  y_endpoint1 = y_endpoint6 * offset_6 + table_tmp;
  x_endpoint2 = x_endpoint6 * offset_5 + ajj;
  y_endpoint2 = y_endpoint6 * offset_5 + table_tmp;
  x_endpoint3 = x_endpoint6 * offset_4 + ajj;
  y_endpoint3 = y_endpoint6 * offset_4 + table_tmp;
  x_endpoint4 = x_endpoint6 * offset_3 + ajj;
  y_endpoint4 = y_endpoint6 * offset_3 + table_tmp;
  x_endpoint5 = x_endpoint6 * offset_2 + ajj;
  y_endpoint5 = y_endpoint6 * offset_2 + table_tmp;
  x_endpoint6 = x_endpoint6 * x_0 + ajj;
  y_endpoint6 = y_endpoint6 * x_0 + table_tmp;
  x_endpoint13 = std::cos(dist_op_data[0] - 1.5707963267948966);
  x_endpoint8 = x_endpoint13 * x_0 + ajj;
  end_heading = std::sin(dist_op_data[0] - 1.5707963267948966);
  y_endpoint8 = end_heading * x_0 + table_tmp;
  x_endpoint9 = x_endpoint13 * offset_2 + ajj;
  y_endpoint9 = end_heading * offset_2 + table_tmp;
  x_endpoint10 = x_endpoint13 * offset_3 + ajj;
  y_endpoint10 = end_heading * offset_3 + table_tmp;
  x_endpoint11 = x_endpoint13 * offset_4 + ajj;
  y_endpoint11 = end_heading * offset_4 + table_tmp;
  x_endpoint12 = x_endpoint13 * offset_5 + ajj;
  y_endpoint12 = end_heading * offset_5 + table_tmp;
  x_endpoint13 = x_endpoint13 * offset_6 + ajj;
  offset_6 = end_heading * offset_6 + table_tmp;
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_0,
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint1, y_endpoint1, dist_op_data[0], rtDW.Static_Path_0
            [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], head_err, x,
            b_Path_dis_data, XP1, YP1, K1_0, K_11, &rtb_J_out_d[0]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_0,
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint2, y_endpoint2, dist_op_data[0], rtDW.Static_Path_0
            [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], head_err, X2, Y2,
            XP2, YP2, K2, K_12, &rtb_J_out_d[1]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_0,
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint3, y_endpoint3, dist_op_data[0], rtDW.Static_Path_0
            [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], head_err, X3, Y3,
            XP3, YP3, K3, K_13, &rtb_J_out_d[2]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_0,
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint4, y_endpoint4, dist_op_data[0], rtDW.Static_Path_0
            [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], head_err, X4, Y4,
            XP4, YP4, K4, K_14, &rtb_J_out_d[3]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_0,
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint5, y_endpoint5, dist_op_data[0], rtDW.Static_Path_0
            [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], head_err, X5, Y5,
            XP5, YP5, K5, K_15, &rtb_J_out_d[4]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_0,
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint6, y_endpoint6, dist_op_data[0], rtDW.Static_Path_0
            [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], head_err, X6, Y6,
            XP6, YP6, K6, K_16, &rtb_J_out_d[5]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_0,
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13], ajj,
            table_tmp, dist_op_data[0], rtDW.Static_Path_0[(q_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], head_err, X7, Y7, XP7, YP7,
            K7, K_17, &rtb_J_out_d[6]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_0,
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint8, y_endpoint8, dist_op_data[0], rtDW.Static_Path_0
            [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], head_err, X8, Y8,
            XP8, YP8, K8, K_18, &rtb_J_out_d[7]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_0,
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint9, y_endpoint9, dist_op_data[0], rtDW.Static_Path_0
            [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], head_err, X9, Y9,
            XP9, YP9, K9, K_19, &rtb_J_out_d[8]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_0,
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint10, y_endpoint10, dist_op_data[0], rtDW.Static_Path_0
            [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], head_err, X10,
            Y10, XP10, YP10, K10, K_110, &rtb_J_out_d[9]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_0,
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint11, y_endpoint11, dist_op_data[0], rtDW.Static_Path_0
            [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], head_err, X11,
            Y11, XP11, YP11, K11, K_111, &rtb_J_out_d[10]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_0,
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint12, y_endpoint12, dist_op_data[0], rtDW.Static_Path_0
            [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], head_err, X12,
            Y12, XP12, YP12, K12, K_112, &rtb_J_out_d[11]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_0,
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint13, offset_6, dist_op_data[0], rtDW.Static_Path_0[(q_data
             [0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], head_err, X13, Y13,
            XP13, YP13, K13, K_113, &rtb_J_out_d[12]);
  for (iy = 0; iy < 11; iy++) {
    X_2[iy] = x[iy];
    X_2[iy + 11] = X2[iy];
    X_2[iy + 22] = X3[iy];
    X_2[iy + 33] = X4[iy];
    X_2[iy + 44] = X5[iy];
    X_2[iy + 55] = X6[iy];
    X_2[iy + 66] = X7[iy];
    X_2[iy + 77] = X8[iy];
    X_2[iy + 88] = X9[iy];
    X_2[iy + 99] = X10[iy];
    X_2[iy + 110] = X11[iy];
    X_2[iy + 121] = X12[iy];
    X_2[iy + 132] = X13[iy];
    Y[iy] = b_Path_dis_data[iy];
    Y[iy + 11] = Y2[iy];
    Y[iy + 22] = Y3[iy];
    Y[iy + 33] = Y4[iy];
    Y[iy + 44] = Y5[iy];
    Y[iy + 55] = Y6[iy];
    Y[iy + 66] = Y7[iy];
    Y[iy + 77] = Y8[iy];
    Y[iy + 88] = Y9[iy];
    Y[iy + 99] = Y10[iy];
    Y[iy + 110] = Y11[iy];
    Y[iy + 121] = Y12[iy];
    Y[iy + 132] = Y13[iy];
    K[iy] = K1_0[iy];
    K[iy + 11] = K2[iy];
    K[iy + 22] = K3[iy];
    K[iy + 33] = K4[iy];
    K[iy + 44] = K5[iy];
    K[iy + 55] = K6[iy];
    K[iy + 66] = K7[iy];
    K[iy + 77] = K8[iy];
    K[iy + 88] = K9[iy];
    K[iy + 99] = K10[iy];
    K[iy + 110] = K11[iy];
    K[iy + 121] = K12[iy];
    K[iy + 132] = K13[iy];
    K_1[iy] = K_11[iy];
    K_1[iy + 11] = K_12[iy];
    K_1[iy + 22] = K_13[iy];
    K_1[iy + 33] = K_14[iy];
    K_1[iy + 44] = K_15[iy];
    K_1[iy + 55] = K_16[iy];
    K_1[iy + 66] = K_17[iy];
    K_1[iy + 77] = K_18[iy];
    K_1[iy + 88] = K_19[iy];
    K_1[iy + 99] = K_110[iy];
    K_1[iy + 110] = K_111[iy];
    K_1[iy + 121] = K_112[iy];
    K_1[iy + 132] = K_113[iy];
  }

  rtb_TmpSignalConversionAtSFun_0 = offset_6;
  memset(&Path_col[0], 0, 52U * sizeof(real_T));
  for (iy = 0; iy < 5; iy++) {
    Path_col[3 + ((8 + iy) << 2)] = 1.0;
  }

  memcpy(&OBXY_EL[0], &rtb_V_boundingbox[0], 400U * sizeof(real_T));
  for (n = 0; n <= shortest_distance_tmp; n++) {
    end_heading = (1.0 + (real_T)n) * 2.0;
    b_idx = (int32_T)(end_heading + -1.0);
    idx = b_idx - 1;
    OBXY_EL[idx] = ((rtb_V_boundingbox[idx] - rtb_V_boundingbox[b_idx + 99]) *
                    0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 +
      -1.0) - 1]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0)
                     - 1] - rtb_V_boundingbox[b_idx + 299]) * 0.3;
    idx = (int32_T)end_heading;
    d_idx = idx - 1;
    OBXY_EL[d_idx] = ((rtb_V_boundingbox[d_idx] - rtb_V_boundingbox[idx + 99]) *
                      0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0)
                      - 1]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
      2.0) - 1] - rtb_V_boundingbox[idx + 299]) * 0.3;
    OBXY_EL[(int32_T)(end_heading + -1.0) + 99] = ((rtb_V_boundingbox[(int32_T)
      ((1.0 + (real_T)n) * 2.0 + -1.0) + 99] - rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)n) * 2.0 + -1.0) - 1]) * 0.3 + rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)n) * 2.0 + -1.0) + 99]) + (rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)n) * 2.0 + -1.0) + 99] - rtb_V_boundingbox[b_idx + 199]) * 0.3;
    OBXY_EL[(int32_T)end_heading + 99] = ((rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)n) * 2.0) + 99] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
      2.0) - 1]) * 0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) +
      99]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) + 99] -
              rtb_V_boundingbox[idx + 199]) * 0.3;
    OBXY_EL[(int32_T)(end_heading + -1.0) + 199] = ((rtb_V_boundingbox[(int32_T)
      ((1.0 + (real_T)n) * 2.0 + -1.0) + 199] - rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)n) * 2.0 + -1.0) + 299]) * 0.3 + rtb_V_boundingbox[(int32_T)
      ((1.0 + (real_T)n) * 2.0 + -1.0) + 199]) + (rtb_V_boundingbox[(int32_T)
      ((1.0 + (real_T)n) * 2.0 + -1.0) + 199] - rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)n) * 2.0 + -1.0) + 99]) * 0.3;
    OBXY_EL[(int32_T)end_heading + 199] = ((rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)n) * 2.0) + 199] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
      2.0) + 299]) * 0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0)
      + 199]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) + 199] -
                 rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) + 99]) *
      0.3;
    OBXY_EL[(int32_T)(end_heading + -1.0) + 299] = ((rtb_V_boundingbox[(int32_T)
      ((1.0 + (real_T)n) * 2.0 + -1.0) + 299] - rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)n) * 2.0 + -1.0) + 199]) * 0.3 + rtb_V_boundingbox[(int32_T)
      ((1.0 + (real_T)n) * 2.0 + -1.0) + 299]) + (rtb_V_boundingbox[(int32_T)
      ((1.0 + (real_T)n) * 2.0 + -1.0) + 299] - rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)n) * 2.0 + -1.0) - 1]) * 0.3;
    OBXY_EL[(int32_T)end_heading + 299] = ((rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)n) * 2.0) + 299] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
      2.0) + 199]) * 0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0)
      + 299]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) + 299] -
                 rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) - 1]) *
      0.3;
  }

  for (iy = 0; iy < 13; iy++) {
    for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 10;
         Path_RES_0_size_idx_1++) {
      b_idx = 11 * iy + Path_RES_0_size_idx_1;
      end_heading = X_2[b_idx + 1] - X_2[b_idx];
      X_diff[Path_RES_0_size_idx_1 + 11 * iy] = end_heading;
      X_diff_0[Path_RES_0_size_idx_1 + 10 * iy] = end_heading;
    }

    idx = 10 + 11 * iy;
    X_diff[idx] = X_diff_0[10 * iy + 9];
    for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 10;
         Path_RES_0_size_idx_1++) {
      b_idx = 11 * iy + Path_RES_0_size_idx_1;
      end_heading = Y[b_idx + 1] - Y[b_idx];
      Y_diff[Path_RES_0_size_idx_1 + 11 * iy] = end_heading;
      X_diff_0[Path_RES_0_size_idx_1 + 10 * iy] = end_heading;
    }

    Y_diff[idx] = X_diff_0[10 * iy + 9];
  }

  power_dw(X_diff, XY_difflen);
  power_dw(Y_diff, Path_vehFLY);
  for (iy = 0; iy < 143; iy++) {
    Path_vehFLX[iy] = XY_difflen[iy] + Path_vehFLY[iy];
  }

  power_dw3(Path_vehFLX, XY_difflen);
  for (iy = 0; iy < 143; iy++) {
    end_heading = X_diff[iy] / XY_difflen[iy];
    x_0 = Y_diff[iy] / XY_difflen[iy];
    offset_2 = 0.99993 * -x_0 + X_2[iy];
    Path_vehFLX[iy] = offset_2 + 1.219116 * end_heading;
    offset_3 = 0.99993 * end_heading + Y[iy];
    Path_vehFLY[iy] = offset_3 + 1.219116 * x_0;
    offset_4 = X_2[iy] - 0.99993 * -x_0;
    Path_vehFRX[iy] = offset_4 + 1.219116 * end_heading;
    offset_5 = Y[iy] - 0.99993 * end_heading;
    Path_vehFRY[iy] = offset_5 + 1.219116 * x_0;
    Path_vehRLX[iy] = offset_2 - 4.876464 * end_heading;
    Path_vehRLY[iy] = offset_3 - 4.876464 * x_0;
    Path_vehRRX[iy] = offset_4 - 4.876464 * end_heading;
    Path_vehRRY[iy] = offset_5 - 4.876464 * x_0;
    X_diff[iy] = end_heading;
    XY_difflen[iy] = -x_0;
    Y_diff[iy] = x_0;
  }

  for (i = 0; i < 13; i++) {
    Path_col[i << 2] = 0.0;
    if (!(Path_col[(i << 2) + 3] == 1.0)) {
      b_colj = 0;
      exitg1 = false;
      while ((!exitg1) && (b_colj < 11)) {
        jj = 11 * i + b_colj;
        OBXY_m[0] = Path_vehFLX[jj];
        OBXY_m[2] = Path_vehFRX[jj];
        OBXY_m[4] = Path_vehRLX[jj];
        OBXY_m[6] = Path_vehRRX[jj];
        OBXY_m[1] = Path_vehFLY[jj];
        OBXY_m[3] = Path_vehFRY[jj];
        OBXY_m[5] = Path_vehRLY[jj];
        OBXY_m[7] = Path_vehRRY[jj];
        b_idx = 0;
        exitg3 = false;
        while ((!exitg3) && (b_idx <= shortest_distance_tmp)) {
          offset_2 = (1.0 + (real_T)b_idx) * 2.0;
          idx = (int32_T)(offset_2 + -1.0);
          end_heading = OBXY_EL[idx + 99] - OBXY_EL[idx - 1];
          x_0 = std::sqrt(end_heading * end_heading + end_heading * end_heading);
          d_idx = (int32_T)offset_2;
          rtb_Oi_near_l[0] = -(OBXY_EL[d_idx + 99] - OBXY_EL[d_idx - 1]) / x_0;
          rtb_Oi_near_l[1] = end_heading / x_0;
          offset_4 = OBXY_EL[d_idx + 199] - OBXY_EL[(int32_T)((1.0 + (real_T)
            b_idx) * 2.0) + 99];
          x_0 = OBXY_EL[idx + 199] - OBXY_EL[(int32_T)((1.0 + (real_T)b_idx) *
            2.0 + -1.0) + 99];
          offset_3 = std::sqrt(offset_4 * offset_4 + x_0 * x_0);
          end_heading = -offset_4 / offset_3;
          x_0 /= offset_3;
          rtb_Oi_near_o[0] = rtb_Oi_near_l[0];
          rtb_Oi_near_o[1] = end_heading;
          rtb_Oi_near_o[4] = rtb_Oi_near_l[1];
          rtb_Oi_near_o[5] = x_0;
          rtb_Oi_near_o[2] = X_diff[jj];
          rtb_Oi_near_o[6] = Y_diff[jj];
          rtb_Oi_near_o[3] = XY_difflen[jj];
          rtb_Oi_near_o[7] = X_diff[11 * i + b_colj];
          rtb_Oi_near_o_0[0] = rtb_Oi_near_l[0];
          rtb_Oi_near_o_0[1] = end_heading;
          rtb_Oi_near_o_0[4] = rtb_Oi_near_l[1];
          rtb_Oi_near_o_0[5] = x_0;
          rtb_Oi_near_o_0[2] = X_diff[11 * i + b_colj];
          rtb_Oi_near_o_0[6] = Y_diff[11 * i + b_colj];
          rtb_Oi_near_o_0[3] = XY_difflen[11 * i + b_colj];
          rtb_Oi_near_o_0[7] = X_diff[11 * i + b_colj];
          for (iy = 0; iy < 4; iy++) {
            for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 4;
                 Path_RES_0_size_idx_1++) {
              proj_veh[iy + (Path_RES_0_size_idx_1 << 2)] = 0.0;
              proj_veh[iy + (Path_RES_0_size_idx_1 << 2)] +=
                OBXY_m[Path_RES_0_size_idx_1 << 1] * rtb_Oi_near_o[iy];
              proj_veh[iy + (Path_RES_0_size_idx_1 << 2)] += OBXY_m
                [(Path_RES_0_size_idx_1 << 1) + 1] * rtb_Oi_near_o[iy + 4];
            }

            OBXY_EL_0[iy << 1] = OBXY_EL[((int32_T)(offset_2 + -1.0) + 100 * iy)
              - 1];
            OBXY_EL_0[1 + (iy << 1)] = OBXY_EL[(100 * iy + (int32_T)offset_2) -
              1];
          }

          for (idx = 0; idx < 4; idx++) {
            for (iy = 0; iy < 4; iy++) {
              proj_ob[idx + (iy << 2)] = 0.0;
              proj_ob[idx + (iy << 2)] += OBXY_EL_0[iy << 1] *
                rtb_Oi_near_o_0[idx];
              proj_ob[idx + (iy << 2)] += OBXY_EL_0[(iy << 1) + 1] *
                rtb_Oi_near_o_0[idx + 4];
            }

            K1[idx] = proj_veh[idx];
          }

          end_heading = proj_veh[0];
          x_0 = proj_veh[1];
          offset_2 = proj_veh[2];
          offset_3 = proj_veh[3];
          for (idx = 0; idx < 3; idx++) {
            if ((!rtIsNaN(proj_veh[(idx + 1) << 2])) && (rtIsNaN(K1[0]) || (K1[0]
                  > proj_veh[(idx + 1) << 2]))) {
              K1[0] = proj_veh[(idx + 1) << 2];
            }

            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 1])) && (rtIsNaN(K1[1]) ||
                 (K1[1] > proj_veh[((idx + 1) << 2) + 1]))) {
              K1[1] = proj_veh[((idx + 1) << 2) + 1];
            }

            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 2])) && (rtIsNaN(K1[2]) ||
                 (K1[2] > proj_veh[((idx + 1) << 2) + 2]))) {
              K1[2] = proj_veh[((idx + 1) << 2) + 2];
            }

            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 3])) && (rtIsNaN(K1[3]) ||
                 (K1[3] > proj_veh[((idx + 1) << 2) + 3]))) {
              K1[3] = proj_veh[((idx + 1) << 2) + 3];
            }

            offset_4 = end_heading;
            if ((!rtIsNaN(proj_veh[(idx + 1) << 2])) && (rtIsNaN(end_heading) ||
                 (end_heading < proj_veh[(idx + 1) << 2]))) {
              offset_4 = proj_veh[(idx + 1) << 2];
            }

            end_heading = offset_4;
            offset_4 = x_0;
            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 1])) && (rtIsNaN(x_0) ||
                 (x_0 < proj_veh[((idx + 1) << 2) + 1]))) {
              offset_4 = proj_veh[((idx + 1) << 2) + 1];
            }

            x_0 = offset_4;
            offset_4 = offset_2;
            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 2])) && (rtIsNaN(offset_2)
                 || (offset_2 < proj_veh[((idx + 1) << 2) + 2]))) {
              offset_4 = proj_veh[((idx + 1) << 2) + 2];
            }

            offset_2 = offset_4;
            offset_4 = offset_3;
            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 3])) && (rtIsNaN(offset_3)
                 || (offset_3 < proj_veh[((idx + 1) << 2) + 3]))) {
              offset_4 = proj_veh[((idx + 1) << 2) + 3];
            }

            offset_3 = offset_4;
          }

          minmax_veh[0] = K1[0];
          minmax_veh[4] = end_heading;
          minmax_veh[1] = K1[1];
          minmax_veh[5] = x_0;
          minmax_veh[2] = K1[2];
          minmax_veh[6] = offset_2;
          minmax_veh[3] = K1[3];
          minmax_veh[7] = offset_3;
          K1[0] = proj_ob[0];
          K1[1] = proj_ob[1];
          K1[2] = proj_ob[2];
          K1[3] = proj_ob[3];
          end_heading = proj_ob[0];
          x_0 = proj_ob[1];
          offset_2 = proj_ob[2];
          offset_3 = proj_ob[3];
          for (idx = 0; idx < 3; idx++) {
            if ((!rtIsNaN(proj_ob[(idx + 1) << 2])) && (rtIsNaN(K1[0]) || (K1[0]
                  > proj_ob[(idx + 1) << 2]))) {
              K1[0] = proj_ob[(idx + 1) << 2];
            }

            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 1])) && (rtIsNaN(K1[1]) ||
                 (K1[1] > proj_ob[((idx + 1) << 2) + 1]))) {
              K1[1] = proj_ob[((idx + 1) << 2) + 1];
            }

            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 2])) && (rtIsNaN(K1[2]) ||
                 (K1[2] > proj_ob[((idx + 1) << 2) + 2]))) {
              K1[2] = proj_ob[((idx + 1) << 2) + 2];
            }

            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 3])) && (rtIsNaN(K1[3]) ||
                 (K1[3] > proj_ob[((idx + 1) << 2) + 3]))) {
              K1[3] = proj_ob[((idx + 1) << 2) + 3];
            }

            offset_4 = end_heading;
            if ((!rtIsNaN(proj_ob[(idx + 1) << 2])) && (rtIsNaN(end_heading) ||
                 (end_heading < proj_ob[(idx + 1) << 2]))) {
              offset_4 = proj_ob[(idx + 1) << 2];
            }

            end_heading = offset_4;
            offset_4 = x_0;
            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 1])) && (rtIsNaN(x_0) ||
                 (x_0 < proj_ob[((idx + 1) << 2) + 1]))) {
              offset_4 = proj_ob[((idx + 1) << 2) + 1];
            }

            x_0 = offset_4;
            offset_4 = offset_2;
            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 2])) && (rtIsNaN(offset_2) ||
                 (offset_2 < proj_ob[((idx + 1) << 2) + 2]))) {
              offset_4 = proj_ob[((idx + 1) << 2) + 2];
            }

            offset_2 = offset_4;
            offset_4 = offset_3;
            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 3])) && (rtIsNaN(offset_3) ||
                 (offset_3 < proj_ob[((idx + 1) << 2) + 3]))) {
              offset_4 = proj_ob[((idx + 1) << 2) + 3];
            }

            offset_3 = offset_4;
          }

          minmax_obj[0] = K1[0];
          minmax_obj[4] = end_heading;
          minmax_obj[1] = K1[1];
          minmax_obj[5] = x_0;
          minmax_obj[2] = K1[2];
          minmax_obj[6] = offset_2;
          minmax_obj[3] = K1[3];
          minmax_obj[7] = offset_3;
          n = 0;
          exitg4 = false;
          while ((!exitg4) && (n < 4)) {
            if (minmax_veh[n] > minmax_obj[4 + n]) {
              Path_col[i << 2] = 0.0;
              exitg4 = true;
            } else if (minmax_veh[4 + n] < minmax_obj[n]) {
              Path_col[i << 2] = 0.0;
              exitg4 = true;
            } else {
              Path_col[i << 2] = 1.0;
              n++;
            }
          }

          if (Path_col[i << 2] == 1.0) {
            Path_col[2 + (i << 2)] = 1.0 + (real_T)b_idx;
            exitg3 = true;
          } else {
            b_idx++;
          }
        }

        if (Path_col[i << 2] == 1.0) {
          Path_col[1 + (i << 2)] = 1.0 + (real_T)b_colj;
          exitg1 = true;
        } else {
          b_colj++;
        }
      }
    }

    Cobs[i] = Path_col[i << 2];
    Path_col_0[i] = Path_col[i << 2];
  }

  x_0 = std(Path_col_0);
  if (x_0 != 0.0) {
    offset_2 = x_0 * x_0 * 2.0;
    offset_3 = 2.5066282746310002 * x_0;
    for (idx = 0; idx < 13; idx++) {
      iy = 1 + idx;
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 13;
           Path_RES_0_size_idx_1++) {
        Cc_0[Path_RES_0_size_idx_1] = (iy - Path_RES_0_size_idx_1) - 1;
      }

      power_dw3x(Cc_0, Path_col_0);
      for (iy = 0; iy < 13; iy++) {
        Cc_0[iy] = -Path_col_0[iy] / offset_2;
      }

      exp_n(Cc_0);
      for (iy = 0; iy < 13; iy++) {
        Path_col_0[iy] = Path_col[iy << 2] * (Cc_0[iy] / offset_3);
      }

      Cobs[idx] = sum_a(Path_col_0);
      if ((1 + idx == 1) && (Path_col[0] == 1.0)) {
        Cobs[0] += std::exp(-1.0 / (x_0 * x_0 * 2.0)) / (2.5066282746310002 *
          x_0);
      } else {
        if ((1 + idx == 13) && (Path_col[48] == 1.0)) {
          Cobs[12] += std::exp(-1.0 / (x_0 * x_0 * 2.0)) / (2.5066282746310002 *
            x_0);
        }
      }
    }

    b_x_0 = rtIsNaN(Cobs[0]);
    if (!b_x_0) {
      idx = 1;
    } else {
      idx = 0;
      b_colj = 2;
      exitg1 = false;
      while ((!exitg1) && (b_colj < 14)) {
        if (!rtIsNaN(Cobs[b_colj - 1])) {
          idx = b_colj;
          exitg1 = true;
        } else {
          b_colj++;
        }
      }
    }

    if (idx == 0) {
      offset_2 = Cobs[0];
    } else {
      offset_2 = Cobs[idx - 1];
      while (idx + 1 < 14) {
        if (offset_2 < Cobs[idx]) {
          offset_2 = Cobs[idx];
        }

        idx++;
      }
    }

    if (offset_2 != 1.0) {
      if (!b_x_0) {
        b_idx = 1;
      } else {
        b_idx = 0;
        d_idx = 2;
        exitg1 = false;
        while ((!exitg1) && (d_idx < 14)) {
          if (!rtIsNaN(Cobs[d_idx - 1])) {
            b_idx = d_idx;
            exitg1 = true;
          } else {
            d_idx++;
          }
        }
      }

      if (b_idx == 0) {
        offset_2 = Cobs[0];
      } else {
        offset_2 = Cobs[b_idx - 1];
        while (b_idx + 1 < 14) {
          if (offset_2 < Cobs[b_idx]) {
            offset_2 = Cobs[b_idx];
          }

          b_idx++;
        }
      }

      for (iy = 0; iy < 13; iy++) {
        Cobs[iy] /= offset_2;
      }
    }
  }

  for (iy = 0; iy < 13; iy++) {
    Clane[iy] = Path_col[(iy << 2) + 3];
    Path_col_0[iy] = Path_col[(iy << 2) + 3];
  }

  x_0 = std(Path_col_0);
  if (x_0 != 0.0) {
    offset_2 = x_0 * x_0 * 2.0;
    offset_3 = 2.5066282746310002 * x_0;
    for (i = 0; i < 13; i++) {
      iy = 1 + i;
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 13;
           Path_RES_0_size_idx_1++) {
        Cc_0[Path_RES_0_size_idx_1] = (iy - Path_RES_0_size_idx_1) - 1;
      }

      power_dw3x(Cc_0, Path_col_0);
      for (iy = 0; iy < 13; iy++) {
        Cc_0[iy] = -Path_col_0[iy] / offset_2;
      }

      exp_n(Cc_0);
      for (iy = 0; iy < 13; iy++) {
        Path_col_0[iy] = Path_col[(iy << 2) + 3] * (Cc_0[iy] / offset_3);
      }

      Clane[i] = sum_a(Path_col_0);
      if ((1 + i == 1) && (Path_col[3] == 1.0)) {
        Clane[0] += std::exp(-1.0 / (x_0 * x_0 * 2.0)) / (2.5066282746310002 *
          x_0);
      } else {
        if ((1 + i == 13) && (Path_col[51] == 1.0)) {
          Clane[12] += std::exp(-1.0 / (x_0 * x_0 * 2.0)) / (2.5066282746310002 *
            x_0);
        }
      }
    }

    b_x_0 = rtIsNaN(Clane[0]);
    if (!b_x_0) {
      b_idx = 1;
    } else {
      b_idx = 0;
      d_idx = 2;
      exitg1 = false;
      while ((!exitg1) && (d_idx < 14)) {
        if (!rtIsNaN(Clane[d_idx - 1])) {
          b_idx = d_idx;
          exitg1 = true;
        } else {
          d_idx++;
        }
      }
    }

    if (b_idx == 0) {
      offset_2 = Clane[0];
    } else {
      offset_2 = Clane[b_idx - 1];
      while (b_idx + 1 < 14) {
        if (offset_2 < Clane[b_idx]) {
          offset_2 = Clane[b_idx];
        }

        b_idx++;
      }
    }

    if (offset_2 != 1.0) {
      if (!b_x_0) {
        d_idx = 1;
      } else {
        d_idx = 0;
        b_colj = 2;
        exitg1 = false;
        while ((!exitg1) && (b_colj < 14)) {
          if (!rtIsNaN(Clane[b_colj - 1])) {
            d_idx = b_colj;
            exitg1 = true;
          } else {
            b_colj++;
          }
        }
      }

      if (d_idx == 0) {
        offset_2 = Clane[0];
      } else {
        offset_2 = Clane[d_idx - 1];
        while (d_idx + 1 < 14) {
          if (offset_2 < Clane[d_idx]) {
            offset_2 = Clane[d_idx];
          }

          d_idx++;
        }
      }

      for (iy = 0; iy < 13; iy++) {
        Clane[iy] /= offset_2;
      }
    }
  }

  for (iy = 0; iy < 11; iy++) {
    x[iy] = rtDW.UnitDelay5_DSTATE[iy] - rtb_X[0];
  }

  power_d(x, K1_0);
  for (iy = 0; iy < 11; iy++) {
    x[iy] = rtDW.UnitDelay5_DSTATE[11 + iy] - rtb_X[1];
  }

  power_d(x, X2);
  for (iy = 0; iy < 11; iy++) {
    b_Path_dis_data[iy] = K1_0[iy] + X2[iy];
  }

  sqrt_l(b_Path_dis_data);
  if (!rtIsNaN(b_Path_dis_data[0])) {
    d_idx = 1;
  } else {
    d_idx = 0;
    b_colj = 2;
    exitg1 = false;
    while ((!exitg1) && (b_colj < 12)) {
      if (!rtIsNaN(b_Path_dis_data[b_colj - 1])) {
        d_idx = b_colj;
        exitg1 = true;
      } else {
        b_colj++;
      }
    }
  }

  if (d_idx == 0) {
    d_idx = 1;
  } else {
    offset_2 = b_Path_dis_data[d_idx - 1];
    for (b_colj = d_idx; b_colj + 1 < 12; b_colj++) {
      if (offset_2 > b_Path_dis_data[b_colj]) {
        offset_2 = b_Path_dis_data[b_colj];
        d_idx = b_colj + 1;
      }
    }
  }

  jj = 12 - d_idx;
  d_ix = -d_idx;
  for (iy = 0; iy <= d_ix + 11; iy++) {
    LastPath_overlap_data[iy] = rtDW.UnitDelay5_DSTATE[(d_idx + iy) - 1];
  }

  d_ix = -d_idx;
  for (iy = 0; iy <= d_ix + 11; iy++) {
    LastPath_overlap_data[iy + jj] = rtDW.UnitDelay5_DSTATE[(d_idx + iy) + 10];
  }

  for (b_idx = 0; b_idx < 13; b_idx++) {
    for (iy = 0; iy < 11; iy++) {
      b_Path_dis_data[iy] = X_2[11 * b_idx + iy] - rtDW.UnitDelay5_DSTATE[10];
    }

    power_d(b_Path_dis_data, x);
    for (iy = 0; iy < 11; iy++) {
      b_Path_dis_data[iy] = Y[11 * b_idx + iy] - rtDW.UnitDelay5_DSTATE[21];
    }

    power_d(b_Path_dis_data, K1_0);
    for (iy = 0; iy < 11; iy++) {
      b_Path_dis_data[iy] = x[iy] + K1_0[iy];
    }

    sqrt_l(b_Path_dis_data);
    if (!rtIsNaN(b_Path_dis_data[0])) {
      idx = 0;
    } else {
      idx = -1;
      b_colj = 2;
      exitg1 = false;
      while ((!exitg1) && (b_colj < 12)) {
        if (!rtIsNaN(b_Path_dis_data[b_colj - 1])) {
          idx = b_colj - 1;
          exitg1 = true;
        } else {
          b_colj++;
        }
      }
    }

    if (idx + 1 == 0) {
      idx = 0;
    } else {
      x_0 = b_Path_dis_data[idx];
      for (b_colj = idx + 1; b_colj + 1 < 12; b_colj++) {
        if (x_0 > b_Path_dis_data[b_colj]) {
          x_0 = b_Path_dis_data[b_colj];
          idx = b_colj;
        }
      }
    }

    Path_overlap_size[0] = idx + 1;
    if (0 <= idx) {
      memcpy(&Path_overlap_data[0], &X_2[b_idx * 11], (idx + 1) * sizeof(real_T));
    }

    for (iy = 0; iy <= idx; iy++) {
      Path_overlap_data[iy + Path_overlap_size[0]] = Y[11 * b_idx + iy];
    }

    if (12 - d_idx >= Path_overlap_size[0]) {
      idx = 13 - (d_idx + Path_overlap_size[0]);
      if (idx > 12 - d_idx) {
        idx = 1;
        b_colj = 0;
      } else {
        b_colj = 12 - d_idx;
      }

      iy = idx - 1;
      idx = b_colj - iy;
      LastPath_overlap_size_0[0] = idx;
      LastPath_overlap_size_0[1] = 2;
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < idx;
           Path_RES_0_size_idx_1++) {
        LastPath_overlap_data_0[Path_RES_0_size_idx_1] =
          LastPath_overlap_data[iy + Path_RES_0_size_idx_1] -
          Path_overlap_data[Path_RES_0_size_idx_1];
      }

      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < idx;
           Path_RES_0_size_idx_1++) {
        LastPath_overlap_data_0[Path_RES_0_size_idx_1 + idx] =
          LastPath_overlap_data[(iy + Path_RES_0_size_idx_1) + jj] -
          Path_overlap_data[Path_RES_0_size_idx_1 + Path_overlap_size[0]];
      }

      power_dw3xd(LastPath_overlap_data_0, LastPath_overlap_size_0,
                  Path_overlap_data, Path_overlap_size);
      Path_overlap_size_1[0] = 2;
      Path_overlap_size_1[1] = Path_overlap_size[0];
      d_ix = Path_overlap_size[0];
      for (iy = 0; iy < d_ix; iy++) {
        LastPath_overlap_data_0[iy << 1] = Path_overlap_data[iy];
        LastPath_overlap_data_0[1 + (iy << 1)] = Path_overlap_data[iy +
          Path_overlap_size[0]];
      }

      sum_ae(LastPath_overlap_data_0, Path_overlap_size_1, b_Path_dis_data,
             oi_xy_size);
      sqrt_l5(b_Path_dis_data, oi_xy_size);
      d_ix = oi_xy_size[1];
      for (iy = 0; iy < d_ix; iy++) {
        K_11[iy] = b_Path_dis_data[oi_xy_size[0] * iy];
      }

      idx = oi_xy_size[1];
      b_colj = oi_xy_size[1];
      if (0 <= idx - 1) {
        memcpy(&Path_dis_data[0], &K_11[0], idx * sizeof(real_T));
      }
    } else {
      i = 12 - d_idx;
      LastPath_overlap_size[0] = i;
      LastPath_overlap_size[1] = 2;
      for (iy = 0; iy < i; iy++) {
        LastPath_overlap_data_0[iy] = LastPath_overlap_data[iy] -
          Path_overlap_data[iy];
      }

      for (iy = 0; iy < i; iy++) {
        LastPath_overlap_data_0[iy + i] = LastPath_overlap_data[iy + jj] -
          Path_overlap_data[iy + Path_overlap_size[0]];
      }

      power_dw3xd(LastPath_overlap_data_0, LastPath_overlap_size,
                  Path_overlap_data, Path_overlap_size);
      Path_overlap_size_0[0] = 2;
      Path_overlap_size_0[1] = Path_overlap_size[0];
      d_ix = Path_overlap_size[0];
      for (iy = 0; iy < d_ix; iy++) {
        LastPath_overlap_data_0[iy << 1] = Path_overlap_data[iy];
        LastPath_overlap_data_0[1 + (iy << 1)] = Path_overlap_data[iy +
          Path_overlap_size[0]];
      }

      sum_ae(LastPath_overlap_data_0, Path_overlap_size_0, b_Path_dis_data,
             oi_xy_size);
      sqrt_l5(b_Path_dis_data, oi_xy_size);
      d_ix = oi_xy_size[1];
      for (iy = 0; iy < d_ix; iy++) {
        b_Path_dis_data_0[iy] = b_Path_dis_data[oi_xy_size[0] * iy];
      }

      idx = oi_xy_size[1];
      b_colj = oi_xy_size[1];
      if (0 <= idx - 1) {
        memcpy(&Path_dis_data[0], &b_Path_dis_data_0[0], idx * sizeof(real_T));
      }
    }

    if (b_colj > 1) {
      idx = b_colj;
    } else {
      idx = 1;
    }

    if (mod((real_T)idx) == 0.0) {
      if (b_colj > 1) {
        idx = b_colj - 1;
      } else {
        idx = 0;
      }

      oi_xy_size[1] = idx;
      d_ix = idx - 1;
      for (iy = 0; iy <= d_ix; iy++) {
        b_Path_dis_data[iy] = 4.0;
      }
    } else {
      if (b_colj > 1) {
        idx = b_colj;
      } else {
        idx = 1;
      }

      oi_xy_size[1] = idx;
      d_ix = idx - 1;
      for (iy = 0; iy <= d_ix; iy++) {
        b_Path_dis_data[iy] = 4.0;
      }
    }

    b_Path_dis_data[0] = 1.0;
    b_Path_dis_data[oi_xy_size[1] - 1] = 1.0;
    if (3 > oi_xy_size[1] - 2) {
      i = 1;
      n = 1;
      iy = 0;
    } else {
      i = 3;
      n = 2;
      iy = oi_xy_size[1] - 2;
    }

    idx = div_nde_s32_floor((int8_T)iy - i, n);
    for (iy = 0; iy <= idx; iy++) {
      l_data[iy] = (int8_T)((int8_T)((int8_T)(n * (int8_T)iy) + i) - 1);
    }

    for (iy = 0; iy <= idx; iy++) {
      b_Path_dis_data[l_data[iy]] = 2.0;
    }

    end_heading = 0.0;
    for (iy = 0; iy < oi_xy_size[1]; iy++) {
      end_heading += b_Path_dis_data[iy] * Path_dis_data[iy];
    }

    if (!(b_colj > 1)) {
      b_colj = 1;
    }

    Cc_0[b_idx] = rtb_J_out_d[b_idx] / 11.0 * end_heading / 3.0 /
      (rtb_J_out_d[b_idx] * (real_T)b_colj / 11.0);
  }

  abs_a(K, XY_difflen);
  for (idx = 0; idx < 13; idx++) {
    Path_col_0[idx] = XY_difflen[11 * idx];
    for (d_idx = 0; d_idx < 10; d_idx++) {
      end_heading = Path_col_0[idx];
      iy = (11 * idx + d_idx) + 1;
      if ((!rtIsNaN(XY_difflen[iy])) && (rtIsNaN(Path_col_0[idx]) ||
           (Path_col_0[idx] < XY_difflen[iy]))) {
        end_heading = XY_difflen[iy];
      }

      Path_col_0[idx] = end_heading;
    }
  }

  abs_a(K, XY_difflen);
  for (idx = 0; idx < 13; idx++) {
    f_maxval[idx] = XY_difflen[11 * idx];
    for (d_idx = 0; d_idx < 10; d_idx++) {
      x_0 = f_maxval[idx];
      if ((!rtIsNaN(XY_difflen[(11 * idx + d_idx) + 1])) && (rtIsNaN
           (f_maxval[idx]) || (f_maxval[idx] < XY_difflen[(11 * idx + d_idx) + 1])))
      {
        x_0 = XY_difflen[(11 * idx + d_idx) + 1];
      }

      f_maxval[idx] = x_0;
    }

    f_maxval[idx] *= 10.0;
  }

  if (!rtIsNaN(f_maxval[0])) {
    d_idx = 1;
  } else {
    d_idx = 0;
    b_colj = 2;
    exitg1 = false;
    while ((!exitg1) && (b_colj < 14)) {
      if (!rtIsNaN(f_maxval[b_colj - 1])) {
        d_idx = b_colj;
        exitg1 = true;
      } else {
        b_colj++;
      }
    }
  }

  if (d_idx == 0) {
    end_heading = f_maxval[0];
  } else {
    end_heading = f_maxval[d_idx - 1];
    while (d_idx + 1 < 14) {
      if (end_heading < f_maxval[d_idx]) {
        end_heading = f_maxval[d_idx];
      }

      d_idx++;
    }
  }

  abs_a(K_1, XY_difflen);
  for (idx = 0; idx < 13; idx++) {
    f_maxval[idx] = XY_difflen[11 * idx];
    for (d_idx = 0; d_idx < 10; d_idx++) {
      x_0 = f_maxval[idx];
      iy = (11 * idx + d_idx) + 1;
      if ((!rtIsNaN(XY_difflen[iy])) && (rtIsNaN(f_maxval[idx]) || (f_maxval[idx]
            < XY_difflen[iy]))) {
        x_0 = XY_difflen[iy];
      }

      f_maxval[idx] = x_0;
    }
  }

  abs_a(K_1, XY_difflen);
  for (d_idx = 0; d_idx < 13; d_idx++) {
    Cobs_0[d_idx] = XY_difflen[11 * d_idx];
    for (n = 0; n < 10; n++) {
      x_0 = Cobs_0[d_idx];
      iy = (11 * d_idx + n) + 1;
      if ((!rtIsNaN(XY_difflen[iy])) && (rtIsNaN(Cobs_0[d_idx]) || (Cobs_0[d_idx]
            < XY_difflen[iy]))) {
        x_0 = XY_difflen[iy];
      }

      Cobs_0[d_idx] = x_0;
    }

    Cobs_0[d_idx] *= 10.0;
  }

  if (!rtIsNaN(Cobs_0[0])) {
    idx = 1;
  } else {
    idx = 0;
    d_idx = 2;
    exitg1 = false;
    while ((!exitg1) && (d_idx < 14)) {
      if (!rtIsNaN(Cobs_0[d_idx - 1])) {
        idx = d_idx;
        exitg1 = true;
      } else {
        d_idx++;
      }
    }
  }

  if (idx == 0) {
    x_0 = Cobs_0[0];
  } else {
    x_0 = Cobs_0[idx - 1];
    while (idx + 1 < 14) {
      if (x_0 < Cobs_0[idx]) {
        x_0 = Cobs_0[idx];
      }

      idx++;
    }
  }

  if (!rtIsNaN(offset[0])) {
    idx = 1;
  } else {
    idx = 0;
    b_colj = 2;
    exitg1 = false;
    while ((!exitg1) && (b_colj < 14)) {
      if (!rtIsNaN(offset[b_colj - 1])) {
        idx = b_colj;
        exitg1 = true;
      } else {
        b_colj++;
      }
    }
  }

  if (idx == 0) {
    offset_2 = offset[0];
  } else {
    offset_2 = offset[idx - 1];
    while (idx + 1 < 14) {
      if (offset_2 < offset[idx]) {
        offset_2 = offset[idx];
      }

      idx++;
    }
  }

  b_x_0 = rtIsNaN(Cc_0[0]);
  if (!b_x_0) {
    idx = 1;
  } else {
    idx = 0;
    b_idx = 2;
    exitg1 = false;
    while ((!exitg1) && (b_idx < 14)) {
      if (!rtIsNaN(Cc_0[b_idx - 1])) {
        idx = b_idx;
        exitg1 = true;
      } else {
        b_idx++;
      }
    }
  }

  if (idx == 0) {
    offset_3 = Cc_0[0];
  } else {
    offset_3 = Cc_0[idx - 1];
    while (idx + 1 < 14) {
      if (offset_3 < Cc_0[idx]) {
        offset_3 = Cc_0[idx];
      }

      idx++;
    }
  }

  if (!(offset_3 == 0.0)) {
    if (!b_x_0) {
      idx = 1;
    } else {
      idx = 0;
      b_idx = 2;
      exitg1 = false;
      while ((!exitg1) && (b_idx < 14)) {
        if (!rtIsNaN(Cc_0[b_idx - 1])) {
          idx = b_idx;
          exitg1 = true;
        } else {
          b_idx++;
        }
      }
    }

    if (idx == 0) {
      offset_3 = Cc_0[0];
    } else {
      offset_3 = Cc_0[idx - 1];
      while (idx + 1 < 14) {
        if (offset_3 < Cc_0[idx]) {
          offset_3 = Cc_0[idx];
        }

        idx++;
      }
    }

    for (iy = 0; iy < 13; iy++) {
      Cc_0[iy] /= offset_3;
    }
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
    rtb_YP_g[i] = YP1[i];
    rtb_YP_g[i + 6] = YP2[i];
    rtb_YP_g[i + 12] = YP3[i];
    rtb_YP_g[i + 18] = YP4[i];
    rtb_YP_g[i + 24] = YP5[i];
    rtb_YP_g[i + 30] = YP6[i];
    rtb_YP_g[i + 36] = YP7[i];
    rtb_YP_g[i + 42] = YP8[i];
    rtb_YP_g[i + 48] = YP9[i];
    rtb_YP_g[i + 54] = YP10[i];
    rtb_YP_g[i + 60] = YP11[i];
    rtb_YP_g[i + 66] = YP12[i];
    rtb_YP_g[i + 72] = YP13[i];
  }

  for (iy = 0; iy < 13; iy++) {
    rtb_J_out_d[iy] = (((((rtb_J_out_d[iy] / head_err * 0.5 + Path_col_0[iy] *
      10.0 / end_heading) + f_maxval[iy] * 10.0 / x_0) + offset[iy] / offset_2 *
                         c) + 30.0 * Cobs[iy]) + 5.0 * Cc_0[iy]) + 30.0 *
      Clane[iy];
  }

  // MATLAB Function: '<S2>/DynamicPathPlanning1' incorporates:
  //   MATLAB Function: '<S2>/DynamicPathPlanning'
  //   MATLAB Function: '<S2>/EndPointDecision'
  //   MATLAB Function: '<S2>/EndPointDecision1'
  //   MATLAB Function: '<S2>/MATLAB Function1'
  //   UnitDelay: '<S2>/Unit Delay6'

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    x_data[iy] = (rtb_Forward_Static_Path_id[rtDW.SFunction_DIMS4 - 1] ==
                  rtDW.Static_Path_0[iy]);
  }

  b_idx = rtDW.SFunction_DIMS4_h[0] - 1;
  d_idx = 0;
  for (b_colj = 0; b_colj <= b_idx; b_colj++) {
    if (x_data[b_colj]) {
      d_idx++;
    }
  }

  i = d_idx;
  d_idx = 0;
  for (b_colj = 0; b_colj <= b_idx; b_colj++) {
    if (x_data[b_colj]) {
      p_data_0[d_idx] = b_colj + 1;
      d_idx++;
    }
  }

  for (iy = 0; iy < i; iy++) {
    Forward_Static_Path_id_0_data[iy] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS4_h[0] * 7 + p_data_0[iy]) - 1] * 3.1415926535897931;
  }

  for (iy = 0; iy < i; iy++) {
    end_heading_0_data[iy] = Forward_Static_Path_id_0_data[iy] / 180.0;
  }

  end_heading = Forward_Static_Path_id_0_data[0] / 180.0;
  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    x_data[iy] = (rtb_Forward_Static_Path_id[rtDW.SFunction_DIMS4 - 1] ==
                  rtDW.Static_Path_0[iy]);
  }

  d_idx = 0;
  for (b_colj = 0; b_colj < rtDW.SFunction_DIMS4_h[0]; b_colj++) {
    if (x_data[b_colj]) {
      q_data_0[d_idx] = b_colj + 1;
      d_idx++;
    }
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (iy = 0; iy < d_ix; iy++) {
    x_data[iy] = (rtb_Forward_Static_Path_id[rtDW.SFunction_DIMS4 - 1] ==
                  rtDW.Static_Path_0[iy]);
  }

  d_idx = 0;
  for (b_colj = 0; b_colj < rtDW.SFunction_DIMS4_h[0]; b_colj++) {
    if (x_data[b_colj]) {
      s_data_0[d_idx] = b_colj + 1;
      d_idx++;
    }
  }

  x_0 = rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 10 + s_data_0[0]) - 1] /
    4.0;
  offset_2 = x_0 * 2.0;
  offset_3 = x_0 * 3.0;
  offset_4 = x_0 * 4.0;
  offset_5 = x_0 * 5.0;
  offset_6 = x_0 * 6.0;
  G2splines_e(x_endpoint1, y_endpoint1, dist_op_data[0], rtDW.Static_Path_0
              [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length +
              offset_6 * std::cos(end_heading + 1.5707963267948966), ang_1 +
              offset_6 * std::sin(end_heading + 1.5707963267948966),
              end_heading_0_data[0], rtDW.Static_Path_0[(q_data_0[0] +
    rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Look_ahead_S0, x, b_Path_dis_data,
              XP1, YP1, K1_0, K_11, &offset[0]);
  G2splines_e(x_endpoint2, y_endpoint2, dist_op_data[0], rtDW.Static_Path_0
              [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length +
              offset_5 * std::cos(end_heading + 1.5707963267948966), ang_1 +
              offset_5 * std::sin(end_heading + 1.5707963267948966),
              end_heading_0_data[0], rtDW.Static_Path_0[(q_data_0[0] +
    rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Look_ahead_S0, X2, Y2, XP2, YP2,
              K1_0, K_11, &offset[1]);
  G2splines_e(x_endpoint3, y_endpoint3, dist_op_data[0], rtDW.Static_Path_0
              [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length +
              offset_4 * std::cos(end_heading + 1.5707963267948966), ang_1 +
              offset_4 * std::sin(end_heading + 1.5707963267948966),
              end_heading_0_data[0], rtDW.Static_Path_0[(q_data_0[0] +
    rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Look_ahead_S0, X3, Y3, XP3, YP3,
              K1_0, K_11, &offset[2]);
  G2splines_e(x_endpoint4, y_endpoint4, dist_op_data[0], rtDW.Static_Path_0
              [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length +
              offset_3 * std::cos(end_heading + 1.5707963267948966), ang_1 +
              offset_3 * std::sin(end_heading + 1.5707963267948966),
              end_heading_0_data[0], rtDW.Static_Path_0[(q_data_0[0] +
    rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Look_ahead_S0, X4, Y4, XP4, YP4,
              K1_0, K_11, &offset[3]);
  G2splines_e(x_endpoint5, y_endpoint5, dist_op_data[0], rtDW.Static_Path_0
              [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length +
              offset_2 * std::cos(end_heading + 1.5707963267948966), ang_1 +
              offset_2 * std::sin(end_heading + 1.5707963267948966),
              end_heading_0_data[0], rtDW.Static_Path_0[(q_data_0[0] +
    rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Look_ahead_S0, X5, Y5, XP5, YP5,
              K1_0, K_11, &offset[4]);
  G2splines_e(x_endpoint6, y_endpoint6, dist_op_data[0], rtDW.Static_Path_0
              [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length +
              x_0 * std::cos(end_heading + 1.5707963267948966), ang_1 + x_0 *
              std::sin(end_heading + 1.5707963267948966), end_heading_0_data[0],
              rtDW.Static_Path_0[(q_data_0[0] + rtDW.SFunction_DIMS4_h[0] * 13)
              - 1], rtb_Look_ahead_S0, X6, Y6, XP6, YP6, K1_0, K_11, &offset[5]);
  G2splines_e(ajj, table_tmp, dist_op_data[0], rtDW.Static_Path_0[(q_data[0] +
    rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length, ang_1,
              end_heading_0_data[0], rtDW.Static_Path_0[(q_data_0[0] +
    rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Look_ahead_S0, X7, Y7, XP7, YP7,
              K1_0, K_11, &offset[6]);
  G2splines_e(x_endpoint8, y_endpoint8, dist_op_data[0], rtDW.Static_Path_0
              [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length +
              x_0 * std::cos(end_heading - 1.5707963267948966), ang_1 + x_0 *
              std::sin(end_heading - 1.5707963267948966), end_heading_0_data[0],
              rtDW.Static_Path_0[(q_data_0[0] + rtDW.SFunction_DIMS4_h[0] * 13)
              - 1], rtb_Look_ahead_S0, X8, Y8, XP8, YP8, K1_0, K_11, &offset[7]);
  G2splines_e(x_endpoint9, y_endpoint9, dist_op_data[0], rtDW.Static_Path_0
              [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length +
              offset_2 * std::cos(end_heading - 1.5707963267948966), ang_1 +
              offset_2 * std::sin(end_heading - 1.5707963267948966),
              end_heading_0_data[0], rtDW.Static_Path_0[(q_data_0[0] +
    rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Look_ahead_S0, X9, Y9, XP9, YP9,
              K1_0, K_11, &offset[8]);
  G2splines_e(x_endpoint10, y_endpoint10, dist_op_data[0], rtDW.Static_Path_0
              [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length +
              offset_3 * std::cos(end_heading - 1.5707963267948966), ang_1 +
              offset_3 * std::sin(end_heading - 1.5707963267948966),
              end_heading_0_data[0], rtDW.Static_Path_0[(q_data_0[0] +
    rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Look_ahead_S0, X10, Y10, XP10,
              YP10, K1_0, K_11, &offset[9]);
  G2splines_e(x_endpoint11, y_endpoint11, dist_op_data[0], rtDW.Static_Path_0
              [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length +
              offset_4 * std::cos(end_heading - 1.5707963267948966), ang_1 +
              offset_4 * std::sin(end_heading - 1.5707963267948966),
              end_heading_0_data[0], rtDW.Static_Path_0[(q_data_0[0] +
    rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Look_ahead_S0, X11, Y11, XP11,
              YP11, K1_0, K_11, &offset[10]);
  G2splines_e(x_endpoint12, y_endpoint12, dist_op_data[0], rtDW.Static_Path_0
              [(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length +
              offset_5 * std::cos(end_heading - 1.5707963267948966), ang_1 +
              offset_5 * std::sin(end_heading - 1.5707963267948966),
              end_heading_0_data[0], rtDW.Static_Path_0[(q_data_0[0] +
    rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Look_ahead_S0, X12, Y12, XP12,
              YP12, K1_0, K_11, &offset[11]);
  G2splines_e(x_endpoint13, rtb_TmpSignalConversionAtSFun_0, dist_op_data[0],
              rtDW.Static_Path_0[(q_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) -
              1], total_length + offset_6 * std::cos(end_heading -
    1.5707963267948966), ang_1 + offset_6 * std::sin(end_heading -
    1.5707963267948966), end_heading_0_data[0], rtDW.Static_Path_0[(q_data_0[0]
    + rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Look_ahead_S0, K1_0, K_11, XP13,
              YP13, K13, K_113, &offset[12]);
  for (iy = 0; iy < 11; iy++) {
    X_2[iy] = x[iy];
    X_2[iy + 11] = X2[iy];
    X_2[iy + 22] = X3[iy];
    X_2[iy + 33] = X4[iy];
    X_2[iy + 44] = X5[iy];
    X_2[iy + 55] = X6[iy];
    X_2[iy + 66] = X7[iy];
    X_2[iy + 77] = X8[iy];
    X_2[iy + 88] = X9[iy];
    X_2[iy + 99] = X10[iy];
    X_2[iy + 110] = X11[iy];
    X_2[iy + 121] = X12[iy];
    X_2[iy + 132] = K1_0[iy];
    Y[iy] = b_Path_dis_data[iy];
    Y[iy + 11] = Y2[iy];
    Y[iy + 22] = Y3[iy];
    Y[iy + 33] = Y4[iy];
    Y[iy + 44] = Y5[iy];
    Y[iy + 55] = Y6[iy];
    Y[iy + 66] = Y7[iy];
    Y[iy + 77] = Y8[iy];
    Y[iy + 88] = Y9[iy];
    Y[iy + 99] = Y10[iy];
    Y[iy + 110] = Y11[iy];
    Y[iy + 121] = Y12[iy];
    Y[iy + 132] = K_11[iy];
  }

  memset(&Path_col[0], 0, 52U * sizeof(real_T));
  for (iy = 0; iy < 5; iy++) {
    Path_col[3 + ((8 + iy) << 2)] = 1.0;
  }

  memcpy(&OBXY_EL[0], &rtb_V_boundingbox[0], 400U * sizeof(real_T));
  for (n = 0; n <= shortest_distance_tmp; n++) {
    end_heading = (1.0 + (real_T)n) * 2.0;
    b_idx = (int32_T)(end_heading + -1.0);
    idx = b_idx - 1;
    OBXY_EL[idx] = ((rtb_V_boundingbox[idx] - rtb_V_boundingbox[b_idx + 99]) *
                    0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 +
      -1.0) - 1]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0)
                     - 1] - rtb_V_boundingbox[b_idx + 299]) * 0.3;
    idx = (int32_T)end_heading;
    d_idx = idx - 1;
    OBXY_EL[d_idx] = ((rtb_V_boundingbox[d_idx] - rtb_V_boundingbox[idx + 99]) *
                      0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0)
                      - 1]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
      2.0) - 1] - rtb_V_boundingbox[idx + 299]) * 0.3;
    OBXY_EL[(int32_T)(end_heading + -1.0) + 99] = ((rtb_V_boundingbox[(int32_T)
      ((1.0 + (real_T)n) * 2.0 + -1.0) + 99] - rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)n) * 2.0 + -1.0) - 1]) * 0.3 + rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)n) * 2.0 + -1.0) + 99]) + (rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)n) * 2.0 + -1.0) + 99] - rtb_V_boundingbox[b_idx + 199]) * 0.3;
    OBXY_EL[(int32_T)end_heading + 99] = ((rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)n) * 2.0) + 99] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
      2.0) - 1]) * 0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) +
      99]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) + 99] -
              rtb_V_boundingbox[idx + 199]) * 0.3;
    OBXY_EL[(int32_T)(end_heading + -1.0) + 199] = ((rtb_V_boundingbox[(int32_T)
      ((1.0 + (real_T)n) * 2.0 + -1.0) + 199] - rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)n) * 2.0 + -1.0) + 299]) * 0.3 + rtb_V_boundingbox[(int32_T)
      ((1.0 + (real_T)n) * 2.0 + -1.0) + 199]) + (rtb_V_boundingbox[(int32_T)
      ((1.0 + (real_T)n) * 2.0 + -1.0) + 199] - rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)n) * 2.0 + -1.0) + 99]) * 0.3;
    OBXY_EL[(int32_T)end_heading + 199] = ((rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)n) * 2.0) + 199] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
      2.0) + 299]) * 0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0)
      + 199]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) + 199] -
                 rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) + 99]) *
      0.3;
    OBXY_EL[(int32_T)(end_heading + -1.0) + 299] = ((rtb_V_boundingbox[(int32_T)
      ((1.0 + (real_T)n) * 2.0 + -1.0) + 299] - rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)n) * 2.0 + -1.0) + 199]) * 0.3 + rtb_V_boundingbox[(int32_T)
      ((1.0 + (real_T)n) * 2.0 + -1.0) + 299]) + (rtb_V_boundingbox[(int32_T)
      ((1.0 + (real_T)n) * 2.0 + -1.0) + 299] - rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)n) * 2.0 + -1.0) - 1]) * 0.3;
    OBXY_EL[(int32_T)end_heading + 299] = ((rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)n) * 2.0) + 299] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
      2.0) + 199]) * 0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0)
      + 299]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) + 299] -
                 rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) - 1]) *
      0.3;
  }

  for (iy = 0; iy < 13; iy++) {
    for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 10;
         Path_RES_0_size_idx_1++) {
      b_idx = 11 * iy + Path_RES_0_size_idx_1;
      end_heading = X_2[b_idx + 1] - X_2[b_idx];
      X_diff[Path_RES_0_size_idx_1 + 11 * iy] = end_heading;
      X_diff_0[Path_RES_0_size_idx_1 + 10 * iy] = end_heading;
    }

    idx = 10 + 11 * iy;
    X_diff[idx] = X_diff_0[10 * iy + 9];
    for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 10;
         Path_RES_0_size_idx_1++) {
      b_idx = 11 * iy + Path_RES_0_size_idx_1;
      end_heading = Y[b_idx + 1] - Y[b_idx];
      Y_diff[Path_RES_0_size_idx_1 + 11 * iy] = end_heading;
      X_diff_0[Path_RES_0_size_idx_1 + 10 * iy] = end_heading;
    }

    Y_diff[idx] = X_diff_0[10 * iy + 9];
  }

  power_dw(X_diff, XY_difflen);
  power_dw(Y_diff, Path_vehFLY);
  for (iy = 0; iy < 143; iy++) {
    Path_vehFLX[iy] = XY_difflen[iy] + Path_vehFLY[iy];
  }

  power_dw3(Path_vehFLX, XY_difflen);
  for (iy = 0; iy < 143; iy++) {
    end_heading = X_diff[iy] / XY_difflen[iy];
    x_0 = Y_diff[iy] / XY_difflen[iy];
    offset_2 = 0.99993 * -x_0 + X_2[iy];
    Path_vehFLX[iy] = offset_2 + 1.219116 * end_heading;
    offset_3 = 0.99993 * end_heading + Y[iy];
    Path_vehFLY[iy] = offset_3 + 1.219116 * x_0;
    offset_4 = X_2[iy] - 0.99993 * -x_0;
    Path_vehFRX[iy] = offset_4 + 1.219116 * end_heading;
    offset_5 = Y[iy] - 0.99993 * end_heading;
    Path_vehFRY[iy] = offset_5 + 1.219116 * x_0;
    Path_vehRLX[iy] = offset_2 - 4.876464 * end_heading;
    Path_vehRLY[iy] = offset_3 - 4.876464 * x_0;
    Path_vehRRX[iy] = offset_4 - 4.876464 * end_heading;
    Path_vehRRY[iy] = offset_5 - 4.876464 * x_0;
    X_diff[iy] = end_heading;
    XY_difflen[iy] = -x_0;
    Y_diff[iy] = x_0;
  }

  for (i = 0; i < 13; i++) {
    Path_col[i << 2] = 0.0;
    if (!(Path_col[(i << 2) + 3] == 1.0)) {
      b_colj = 0;
      exitg1 = false;
      while ((!exitg1) && (b_colj < 11)) {
        jj = 11 * i + b_colj;
        OBXY_m[0] = Path_vehFLX[jj];
        OBXY_m[2] = Path_vehFRX[jj];
        OBXY_m[4] = Path_vehRLX[jj];
        OBXY_m[6] = Path_vehRRX[jj];
        OBXY_m[1] = Path_vehFLY[jj];
        OBXY_m[3] = Path_vehFRY[jj];
        OBXY_m[5] = Path_vehRLY[jj];
        OBXY_m[7] = Path_vehRRY[jj];
        b_idx = 0;
        exitg3 = false;
        while ((!exitg3) && (b_idx <= shortest_distance_tmp)) {
          offset_2 = (1.0 + (real_T)b_idx) * 2.0;
          idx = (int32_T)(offset_2 + -1.0);
          end_heading = OBXY_EL[idx + 99] - OBXY_EL[idx - 1];
          x_0 = std::sqrt(end_heading * end_heading + end_heading * end_heading);
          d_idx = (int32_T)offset_2;
          rtb_Oi_near_l[0] = -(OBXY_EL[d_idx + 99] - OBXY_EL[d_idx - 1]) / x_0;
          rtb_Oi_near_l[1] = end_heading / x_0;
          offset_4 = OBXY_EL[d_idx + 199] - OBXY_EL[(int32_T)((1.0 + (real_T)
            b_idx) * 2.0) + 99];
          x_0 = OBXY_EL[idx + 199] - OBXY_EL[(int32_T)((1.0 + (real_T)b_idx) *
            2.0 + -1.0) + 99];
          offset_3 = std::sqrt(offset_4 * offset_4 + x_0 * x_0);
          end_heading = -offset_4 / offset_3;
          x_0 /= offset_3;
          rtb_Oi_near_o[0] = rtb_Oi_near_l[0];
          rtb_Oi_near_o[1] = end_heading;
          rtb_Oi_near_o[4] = rtb_Oi_near_l[1];
          rtb_Oi_near_o[5] = x_0;
          rtb_Oi_near_o[2] = X_diff[jj];
          rtb_Oi_near_o[6] = Y_diff[jj];
          rtb_Oi_near_o[3] = XY_difflen[jj];
          rtb_Oi_near_o[7] = X_diff[11 * i + b_colj];
          rtb_Oi_near_o_0[0] = rtb_Oi_near_l[0];
          rtb_Oi_near_o_0[1] = end_heading;
          rtb_Oi_near_o_0[4] = rtb_Oi_near_l[1];
          rtb_Oi_near_o_0[5] = x_0;
          rtb_Oi_near_o_0[2] = X_diff[11 * i + b_colj];
          rtb_Oi_near_o_0[6] = Y_diff[11 * i + b_colj];
          rtb_Oi_near_o_0[3] = XY_difflen[11 * i + b_colj];
          rtb_Oi_near_o_0[7] = X_diff[11 * i + b_colj];
          for (iy = 0; iy < 4; iy++) {
            for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 4;
                 Path_RES_0_size_idx_1++) {
              proj_veh[iy + (Path_RES_0_size_idx_1 << 2)] = 0.0;
              proj_veh[iy + (Path_RES_0_size_idx_1 << 2)] +=
                OBXY_m[Path_RES_0_size_idx_1 << 1] * rtb_Oi_near_o[iy];
              proj_veh[iy + (Path_RES_0_size_idx_1 << 2)] += OBXY_m
                [(Path_RES_0_size_idx_1 << 1) + 1] * rtb_Oi_near_o[iy + 4];
            }

            OBXY_EL_0[iy << 1] = OBXY_EL[((int32_T)(offset_2 + -1.0) + 100 * iy)
              - 1];
            OBXY_EL_0[1 + (iy << 1)] = OBXY_EL[(100 * iy + (int32_T)offset_2) -
              1];
          }

          for (d_idx = 0; d_idx < 4; d_idx++) {
            for (iy = 0; iy < 4; iy++) {
              proj_ob[d_idx + (iy << 2)] = 0.0;
              proj_ob[d_idx + (iy << 2)] += OBXY_EL_0[iy << 1] *
                rtb_Oi_near_o_0[d_idx];
              proj_ob[d_idx + (iy << 2)] += OBXY_EL_0[(iy << 1) + 1] *
                rtb_Oi_near_o_0[d_idx + 4];
            }

            K1[d_idx] = proj_veh[d_idx];
          }

          end_heading = proj_veh[0];
          x_0 = proj_veh[1];
          offset_2 = proj_veh[2];
          offset_3 = proj_veh[3];
          for (idx = 0; idx < 3; idx++) {
            if ((!rtIsNaN(proj_veh[(idx + 1) << 2])) && (rtIsNaN(K1[0]) || (K1[0]
                  > proj_veh[(idx + 1) << 2]))) {
              K1[0] = proj_veh[(idx + 1) << 2];
            }

            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 1])) && (rtIsNaN(K1[1]) ||
                 (K1[1] > proj_veh[((idx + 1) << 2) + 1]))) {
              K1[1] = proj_veh[((idx + 1) << 2) + 1];
            }

            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 2])) && (rtIsNaN(K1[2]) ||
                 (K1[2] > proj_veh[((idx + 1) << 2) + 2]))) {
              K1[2] = proj_veh[((idx + 1) << 2) + 2];
            }

            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 3])) && (rtIsNaN(K1[3]) ||
                 (K1[3] > proj_veh[((idx + 1) << 2) + 3]))) {
              K1[3] = proj_veh[((idx + 1) << 2) + 3];
            }

            offset_4 = end_heading;
            if ((!rtIsNaN(proj_veh[(idx + 1) << 2])) && (rtIsNaN(end_heading) ||
                 (end_heading < proj_veh[(idx + 1) << 2]))) {
              offset_4 = proj_veh[(idx + 1) << 2];
            }

            end_heading = offset_4;
            offset_4 = x_0;
            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 1])) && (rtIsNaN(x_0) ||
                 (x_0 < proj_veh[((idx + 1) << 2) + 1]))) {
              offset_4 = proj_veh[((idx + 1) << 2) + 1];
            }

            x_0 = offset_4;
            offset_4 = offset_2;
            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 2])) && (rtIsNaN(offset_2)
                 || (offset_2 < proj_veh[((idx + 1) << 2) + 2]))) {
              offset_4 = proj_veh[((idx + 1) << 2) + 2];
            }

            offset_2 = offset_4;
            offset_4 = offset_3;
            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 3])) && (rtIsNaN(offset_3)
                 || (offset_3 < proj_veh[((idx + 1) << 2) + 3]))) {
              offset_4 = proj_veh[((idx + 1) << 2) + 3];
            }

            offset_3 = offset_4;
          }

          minmax_veh[0] = K1[0];
          minmax_veh[4] = end_heading;
          minmax_veh[1] = K1[1];
          minmax_veh[5] = x_0;
          minmax_veh[2] = K1[2];
          minmax_veh[6] = offset_2;
          minmax_veh[3] = K1[3];
          minmax_veh[7] = offset_3;
          K1[0] = proj_ob[0];
          K1[1] = proj_ob[1];
          K1[2] = proj_ob[2];
          K1[3] = proj_ob[3];
          end_heading = proj_ob[0];
          x_0 = proj_ob[1];
          offset_2 = proj_ob[2];
          offset_3 = proj_ob[3];
          for (idx = 0; idx < 3; idx++) {
            if ((!rtIsNaN(proj_ob[(idx + 1) << 2])) && (rtIsNaN(K1[0]) || (K1[0]
                  > proj_ob[(idx + 1) << 2]))) {
              K1[0] = proj_ob[(idx + 1) << 2];
            }

            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 1])) && (rtIsNaN(K1[1]) ||
                 (K1[1] > proj_ob[((idx + 1) << 2) + 1]))) {
              K1[1] = proj_ob[((idx + 1) << 2) + 1];
            }

            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 2])) && (rtIsNaN(K1[2]) ||
                 (K1[2] > proj_ob[((idx + 1) << 2) + 2]))) {
              K1[2] = proj_ob[((idx + 1) << 2) + 2];
            }

            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 3])) && (rtIsNaN(K1[3]) ||
                 (K1[3] > proj_ob[((idx + 1) << 2) + 3]))) {
              K1[3] = proj_ob[((idx + 1) << 2) + 3];
            }

            offset_4 = end_heading;
            if ((!rtIsNaN(proj_ob[(idx + 1) << 2])) && (rtIsNaN(end_heading) ||
                 (end_heading < proj_ob[(idx + 1) << 2]))) {
              offset_4 = proj_ob[(idx + 1) << 2];
            }

            end_heading = offset_4;
            offset_4 = x_0;
            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 1])) && (rtIsNaN(x_0) ||
                 (x_0 < proj_ob[((idx + 1) << 2) + 1]))) {
              offset_4 = proj_ob[((idx + 1) << 2) + 1];
            }

            x_0 = offset_4;
            offset_4 = offset_2;
            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 2])) && (rtIsNaN(offset_2) ||
                 (offset_2 < proj_ob[((idx + 1) << 2) + 2]))) {
              offset_4 = proj_ob[((idx + 1) << 2) + 2];
            }

            offset_2 = offset_4;
            offset_4 = offset_3;
            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 3])) && (rtIsNaN(offset_3) ||
                 (offset_3 < proj_ob[((idx + 1) << 2) + 3]))) {
              offset_4 = proj_ob[((idx + 1) << 2) + 3];
            }

            offset_3 = offset_4;
          }

          minmax_obj[0] = K1[0];
          minmax_obj[4] = end_heading;
          minmax_obj[1] = K1[1];
          minmax_obj[5] = x_0;
          minmax_obj[2] = K1[2];
          minmax_obj[6] = offset_2;
          minmax_obj[3] = K1[3];
          minmax_obj[7] = offset_3;
          n = 0;
          exitg4 = false;
          while ((!exitg4) && (n < 4)) {
            if (minmax_veh[n] > minmax_obj[4 + n]) {
              Path_col[i << 2] = 0.0;
              exitg4 = true;
            } else if (minmax_veh[4 + n] < minmax_obj[n]) {
              Path_col[i << 2] = 0.0;
              exitg4 = true;
            } else {
              Path_col[i << 2] = 1.0;
              n++;
            }
          }

          if (Path_col[i << 2] == 1.0) {
            Path_col[2 + (i << 2)] = 1.0 + (real_T)b_idx;
            exitg3 = true;
          } else {
            b_idx++;
          }
        }

        if (Path_col[i << 2] == 1.0) {
          Path_col[1 + (i << 2)] = 1.0 + (real_T)b_colj;
          exitg1 = true;
        } else {
          b_colj++;
        }
      }
    }

    Cobs[i] = Path_col[i << 2];
    Path_col_0[i] = Path_col[i << 2];
  }

  x_0 = std(Path_col_0);
  if (x_0 != 0.0) {
    offset_3 = x_0 * x_0 * 2.0;
    total_length = 2.5066282746310002 * x_0;
    for (idx = 0; idx < 13; idx++) {
      shortest_distance_tmp = 1 + idx;
      for (iy = 0; iy < 13; iy++) {
        Cc_0[iy] = (shortest_distance_tmp - iy) - 1;
      }

      power_dw3x(Cc_0, Path_col_0);
      for (shortest_distance_tmp = 0; shortest_distance_tmp < 13;
           shortest_distance_tmp++) {
        Cobs_0[shortest_distance_tmp] = -Path_col_0[shortest_distance_tmp] /
          offset_3;
      }

      exp_n(Cobs_0);
      for (shortest_distance_tmp = 0; shortest_distance_tmp < 13;
           shortest_distance_tmp++) {
        Path_col_0[shortest_distance_tmp] = Path_col[shortest_distance_tmp << 2]
          * (Cobs_0[shortest_distance_tmp] / total_length);
      }

      Cobs[idx] = sum_a(Path_col_0);
      if ((1 + idx == 1) && (Path_col[0] == 1.0)) {
        Cobs[0] += std::exp(-1.0 / (x_0 * x_0 * 2.0)) / (2.5066282746310002 *
          x_0);
      } else {
        if ((1 + idx == 13) && (Path_col[48] == 1.0)) {
          Cobs[12] += std::exp(-1.0 / (x_0 * x_0 * 2.0)) / (2.5066282746310002 *
            x_0);
        }
      }
    }

    b_x_0 = rtIsNaN(Cobs[0]);
    if (!b_x_0) {
      idx = 1;
    } else {
      idx = 0;
      b_colj = 2;
      exitg1 = false;
      while ((!exitg1) && (b_colj < 14)) {
        if (!rtIsNaN(Cobs[b_colj - 1])) {
          idx = b_colj;
          exitg1 = true;
        } else {
          b_colj++;
        }
      }
    }

    if (idx == 0) {
      offset_2 = Cobs[0];
    } else {
      offset_2 = Cobs[idx - 1];
      while (idx + 1 < 14) {
        if (offset_2 < Cobs[idx]) {
          offset_2 = Cobs[idx];
        }

        idx++;
      }
    }

    if (offset_2 != 1.0) {
      if (!b_x_0) {
        b_idx = 1;
      } else {
        b_idx = 0;
        d_idx = 2;
        exitg1 = false;
        while ((!exitg1) && (d_idx < 14)) {
          if (!rtIsNaN(Cobs[d_idx - 1])) {
            b_idx = d_idx;
            exitg1 = true;
          } else {
            d_idx++;
          }
        }
      }

      if (b_idx == 0) {
        offset_2 = Cobs[0];
      } else {
        offset_2 = Cobs[b_idx - 1];
        while (b_idx + 1 < 14) {
          if (offset_2 < Cobs[b_idx]) {
            offset_2 = Cobs[b_idx];
          }

          b_idx++;
        }
      }

      for (shortest_distance_tmp = 0; shortest_distance_tmp < 13;
           shortest_distance_tmp++) {
        Cobs[shortest_distance_tmp] /= offset_2;
      }
    }
  }

  for (shortest_distance_tmp = 0; shortest_distance_tmp < 13;
       shortest_distance_tmp++) {
    Clane[shortest_distance_tmp] = Path_col[(shortest_distance_tmp << 2) + 3];
    Path_col_0[shortest_distance_tmp] = Path_col[(shortest_distance_tmp << 2) +
      3];
  }

  x_0 = std(Path_col_0);
  if (x_0 != 0.0) {
    offset_3 = x_0 * x_0 * 2.0;
    offset_2 = 2.5066282746310002 * x_0;
    for (i = 0; i < 13; i++) {
      shortest_distance_tmp = 1 + i;
      for (iy = 0; iy < 13; iy++) {
        Cc_0[iy] = (shortest_distance_tmp - iy) - 1;
      }

      power_dw3x(Cc_0, Path_col_0);
      for (shortest_distance_tmp = 0; shortest_distance_tmp < 13;
           shortest_distance_tmp++) {
        Cobs_0[shortest_distance_tmp] = -Path_col_0[shortest_distance_tmp] /
          offset_3;
      }

      exp_n(Cobs_0);
      for (shortest_distance_tmp = 0; shortest_distance_tmp < 13;
           shortest_distance_tmp++) {
        Path_col_0[shortest_distance_tmp] = Path_col[(shortest_distance_tmp << 2)
          + 3] * (Cobs_0[shortest_distance_tmp] / offset_2);
      }

      Clane[i] = sum_a(Path_col_0);
      if ((1 + i == 1) && (Path_col[3] == 1.0)) {
        Clane[0] += std::exp(-1.0 / (x_0 * x_0 * 2.0)) / (2.5066282746310002 *
          x_0);
      } else {
        if ((1 + i == 13) && (Path_col[51] == 1.0)) {
          Clane[12] += std::exp(-1.0 / (x_0 * x_0 * 2.0)) / (2.5066282746310002 *
            x_0);
        }
      }
    }

    b_x_0 = rtIsNaN(Clane[0]);
    if (!b_x_0) {
      b_idx = 1;
    } else {
      b_idx = 0;
      d_idx = 2;
      exitg1 = false;
      while ((!exitg1) && (d_idx < 14)) {
        if (!rtIsNaN(Clane[d_idx - 1])) {
          b_idx = d_idx;
          exitg1 = true;
        } else {
          d_idx++;
        }
      }
    }

    if (b_idx == 0) {
      offset_2 = Clane[0];
    } else {
      offset_2 = Clane[b_idx - 1];
      while (b_idx + 1 < 14) {
        if (offset_2 < Clane[b_idx]) {
          offset_2 = Clane[b_idx];
        }

        b_idx++;
      }
    }

    if (offset_2 != 1.0) {
      if (!b_x_0) {
        d_idx = 1;
      } else {
        d_idx = 0;
        b_colj = 2;
        exitg1 = false;
        while ((!exitg1) && (b_colj < 14)) {
          if (!rtIsNaN(Clane[b_colj - 1])) {
            d_idx = b_colj;
            exitg1 = true;
          } else {
            b_colj++;
          }
        }
      }

      if (d_idx == 0) {
        offset_2 = Clane[0];
      } else {
        offset_2 = Clane[d_idx - 1];
        while (d_idx + 1 < 14) {
          if (offset_2 < Clane[d_idx]) {
            offset_2 = Clane[d_idx];
          }

          d_idx++;
        }
      }

      for (shortest_distance_tmp = 0; shortest_distance_tmp < 13;
           shortest_distance_tmp++) {
        Clane[shortest_distance_tmp] /= offset_2;
      }
    }
  }

  for (shortest_distance_tmp = 0; shortest_distance_tmp < 11;
       shortest_distance_tmp++) {
    x[shortest_distance_tmp] = rtDW.UnitDelay6_DSTATE[shortest_distance_tmp] -
      ajj;
  }

  power_d(x, K1_0);
  for (shortest_distance_tmp = 0; shortest_distance_tmp < 11;
       shortest_distance_tmp++) {
    x[shortest_distance_tmp] = rtDW.UnitDelay6_DSTATE[11 + shortest_distance_tmp]
      - table_tmp;
  }

  power_d(x, X2);
  for (shortest_distance_tmp = 0; shortest_distance_tmp < 11;
       shortest_distance_tmp++) {
    K_11[shortest_distance_tmp] = K1_0[shortest_distance_tmp] +
      X2[shortest_distance_tmp];
  }

  sqrt_l(K_11);
  if (!rtIsNaN(K_11[0])) {
    d_idx = 1;
  } else {
    d_idx = 0;
    b_colj = 2;
    exitg1 = false;
    while ((!exitg1) && (b_colj < 12)) {
      if (!rtIsNaN(K_11[b_colj - 1])) {
        d_idx = b_colj;
        exitg1 = true;
      } else {
        b_colj++;
      }
    }
  }

  if (d_idx == 0) {
    d_idx = 1;
  } else {
    offset_2 = K_11[d_idx - 1];
    for (b_colj = d_idx; b_colj + 1 < 12; b_colj++) {
      if (offset_2 > K_11[b_colj]) {
        offset_2 = K_11[b_colj];
        d_idx = b_colj + 1;
      }
    }
  }

  jj = 12 - d_idx;
  d_ix = -d_idx;
  for (shortest_distance_tmp = 0; shortest_distance_tmp <= d_ix + 11;
       shortest_distance_tmp++) {
    LastPath_overlap_data[shortest_distance_tmp] = rtDW.UnitDelay6_DSTATE[(d_idx
      + shortest_distance_tmp) - 1];
  }

  d_ix = -d_idx;
  for (shortest_distance_tmp = 0; shortest_distance_tmp <= d_ix + 11;
       shortest_distance_tmp++) {
    LastPath_overlap_data[shortest_distance_tmp + jj] = rtDW.UnitDelay6_DSTATE
      [(d_idx + shortest_distance_tmp) + 10];
  }

  for (b_idx = 0; b_idx < 13; b_idx++) {
    for (shortest_distance_tmp = 0; shortest_distance_tmp < 11;
         shortest_distance_tmp++) {
      b_Path_dis_data[shortest_distance_tmp] = X_2[11 * b_idx +
        shortest_distance_tmp] - rtDW.UnitDelay6_DSTATE[10];
    }

    power_d(b_Path_dis_data, x);
    for (shortest_distance_tmp = 0; shortest_distance_tmp < 11;
         shortest_distance_tmp++) {
      b_Path_dis_data[shortest_distance_tmp] = Y[11 * b_idx +
        shortest_distance_tmp] - rtDW.UnitDelay6_DSTATE[21];
    }

    power_d(b_Path_dis_data, K1_0);
    for (shortest_distance_tmp = 0; shortest_distance_tmp < 11;
         shortest_distance_tmp++) {
      K_11[shortest_distance_tmp] = x[shortest_distance_tmp] +
        K1_0[shortest_distance_tmp];
    }

    sqrt_l(K_11);
    if (!rtIsNaN(K_11[0])) {
      idx = 0;
    } else {
      idx = -1;
      b_colj = 2;
      exitg1 = false;
      while ((!exitg1) && (b_colj < 12)) {
        if (!rtIsNaN(K_11[b_colj - 1])) {
          idx = b_colj - 1;
          exitg1 = true;
        } else {
          b_colj++;
        }
      }
    }

    if (idx + 1 == 0) {
      idx = 0;
    } else {
      offset_2 = K_11[idx];
      for (b_colj = idx + 1; b_colj + 1 < 12; b_colj++) {
        if (offset_2 > K_11[b_colj]) {
          offset_2 = K_11[b_colj];
          idx = b_colj;
        }
      }
    }

    Path_overlap_size[0] = idx + 1;
    if (0 <= idx) {
      memcpy(&Path_overlap_data[0], &X_2[b_idx * 11], (idx + 1) * sizeof(real_T));
    }

    for (shortest_distance_tmp = 0; shortest_distance_tmp <= idx;
         shortest_distance_tmp++) {
      Path_overlap_data[shortest_distance_tmp + Path_overlap_size[0]] = Y[11 *
        b_idx + shortest_distance_tmp];
    }

    if (12 - d_idx >= Path_overlap_size[0]) {
      idx = 13 - (d_idx + Path_overlap_size[0]);
      if (idx > 12 - d_idx) {
        idx = 1;
        b_colj = 0;
      } else {
        b_colj = 12 - d_idx;
      }

      shortest_distance_tmp = idx - 1;
      idx = b_colj - shortest_distance_tmp;
      LastPath_overlap_size_2[0] = idx;
      LastPath_overlap_size_2[1] = 2;
      for (iy = 0; iy < idx; iy++) {
        LastPath_overlap_data_0[iy] =
          LastPath_overlap_data[shortest_distance_tmp + iy] -
          Path_overlap_data[iy];
      }

      for (iy = 0; iy < idx; iy++) {
        LastPath_overlap_data_0[iy + idx] = LastPath_overlap_data
          [(shortest_distance_tmp + iy) + jj] - Path_overlap_data[iy +
          Path_overlap_size[0]];
      }

      power_egqso(LastPath_overlap_data_0, LastPath_overlap_size_2,
                  Path_overlap_data, Path_overlap_size);
      Path_overlap_size_3[0] = 2;
      Path_overlap_size_3[1] = Path_overlap_size[0];
      d_ix = Path_overlap_size[0];
      for (shortest_distance_tmp = 0; shortest_distance_tmp < d_ix;
           shortest_distance_tmp++) {
        LastPath_overlap_data_0[shortest_distance_tmp << 1] =
          Path_overlap_data[shortest_distance_tmp];
        LastPath_overlap_data_0[1 + (shortest_distance_tmp << 1)] =
          Path_overlap_data[shortest_distance_tmp + Path_overlap_size[0]];
      }

      sum_hx(LastPath_overlap_data_0, Path_overlap_size_3, b_Path_dis_data,
             oi_xy_size);
      sqrt_l5(b_Path_dis_data, oi_xy_size);
      d_ix = oi_xy_size[1];
      for (shortest_distance_tmp = 0; shortest_distance_tmp < d_ix;
           shortest_distance_tmp++) {
        K_11[shortest_distance_tmp] = b_Path_dis_data[oi_xy_size[0] *
          shortest_distance_tmp];
      }

      idx = oi_xy_size[1];
      b_colj = oi_xy_size[1];
      if (0 <= idx - 1) {
        memcpy(&Path_dis_data[0], &K_11[0], idx * sizeof(real_T));
      }
    } else {
      i = 12 - d_idx;
      LastPath_overlap_size_1[0] = i;
      LastPath_overlap_size_1[1] = 2;
      for (shortest_distance_tmp = 0; shortest_distance_tmp < i;
           shortest_distance_tmp++) {
        LastPath_overlap_data_0[shortest_distance_tmp] =
          LastPath_overlap_data[shortest_distance_tmp] -
          Path_overlap_data[shortest_distance_tmp];
      }

      for (shortest_distance_tmp = 0; shortest_distance_tmp < i;
           shortest_distance_tmp++) {
        LastPath_overlap_data_0[shortest_distance_tmp + i] =
          LastPath_overlap_data[shortest_distance_tmp + jj] -
          Path_overlap_data[shortest_distance_tmp + Path_overlap_size[0]];
      }

      power_egqso(LastPath_overlap_data_0, LastPath_overlap_size_1,
                  Path_overlap_data, Path_overlap_size);
      Path_overlap_size_2[0] = 2;
      Path_overlap_size_2[1] = Path_overlap_size[0];
      d_ix = Path_overlap_size[0];
      for (shortest_distance_tmp = 0; shortest_distance_tmp < d_ix;
           shortest_distance_tmp++) {
        LastPath_overlap_data_0[shortest_distance_tmp << 1] =
          Path_overlap_data[shortest_distance_tmp];
        LastPath_overlap_data_0[1 + (shortest_distance_tmp << 1)] =
          Path_overlap_data[shortest_distance_tmp + Path_overlap_size[0]];
      }

      sum_hx(LastPath_overlap_data_0, Path_overlap_size_2, b_Path_dis_data,
             oi_xy_size);
      sqrt_l5(b_Path_dis_data, oi_xy_size);
      d_ix = oi_xy_size[1];
      for (shortest_distance_tmp = 0; shortest_distance_tmp < d_ix;
           shortest_distance_tmp++) {
        b_Path_dis_data_0[shortest_distance_tmp] = b_Path_dis_data[oi_xy_size[0]
          * shortest_distance_tmp];
      }

      idx = oi_xy_size[1];
      b_colj = oi_xy_size[1];
      if (0 <= idx - 1) {
        memcpy(&Path_dis_data[0], &b_Path_dis_data_0[0], idx * sizeof(real_T));
      }
    }

    if (b_colj > 1) {
      idx = b_colj;
    } else {
      idx = 1;
    }

    if (mod((real_T)idx) == 0.0) {
      if (b_colj > 1) {
        idx = b_colj - 1;
      } else {
        idx = 0;
      }

      oi_xy_size[1] = idx;
      d_ix = idx - 1;
      for (shortest_distance_tmp = 0; shortest_distance_tmp <= d_ix;
           shortest_distance_tmp++) {
        b_Path_dis_data[shortest_distance_tmp] = 4.0;
      }
    } else {
      if (b_colj > 1) {
        idx = b_colj;
      } else {
        idx = 1;
      }

      oi_xy_size[1] = idx;
      d_ix = idx - 1;
      for (shortest_distance_tmp = 0; shortest_distance_tmp <= d_ix;
           shortest_distance_tmp++) {
        b_Path_dis_data[shortest_distance_tmp] = 4.0;
      }
    }

    b_Path_dis_data[0] = 1.0;
    b_Path_dis_data[oi_xy_size[1] - 1] = 1.0;
    if (3 > oi_xy_size[1] - 2) {
      i = 1;
      n = 1;
      iy = 0;
    } else {
      i = 3;
      n = 2;
      iy = oi_xy_size[1] - 2;
    }

    idx = div_nde_s32_floor((int8_T)iy - i, n);
    for (shortest_distance_tmp = 0; shortest_distance_tmp <= idx;
         shortest_distance_tmp++) {
      l_data[shortest_distance_tmp] = (int8_T)((int8_T)((int8_T)(n * (int8_T)
        shortest_distance_tmp) + i) - 1);
    }

    for (shortest_distance_tmp = 0; shortest_distance_tmp <= idx;
         shortest_distance_tmp++) {
      b_Path_dis_data[l_data[shortest_distance_tmp]] = 2.0;
    }

    end_heading = 0.0;
    for (shortest_distance_tmp = 0; shortest_distance_tmp < oi_xy_size[1];
         shortest_distance_tmp++) {
      end_heading += b_Path_dis_data[shortest_distance_tmp] *
        Path_dis_data[shortest_distance_tmp];
    }

    if (!(b_colj > 1)) {
      b_colj = 1;
    }

    Cobs_0[b_idx] = offset[b_idx] / 11.0 * end_heading / 3.0 / (offset[b_idx] *
      (real_T)b_colj / 11.0);
  }

  b_x_0 = rtIsNaN(Cobs_0[0]);
  if (!b_x_0) {
    d_idx = 1;
  } else {
    d_idx = 0;
    b_colj = 2;
    exitg1 = false;
    while ((!exitg1) && (b_colj < 14)) {
      if (!rtIsNaN(Cobs_0[b_colj - 1])) {
        d_idx = b_colj;
        exitg1 = true;
      } else {
        b_colj++;
      }
    }
  }

  if (d_idx == 0) {
    end_heading = Cobs_0[0];
  } else {
    end_heading = Cobs_0[d_idx - 1];
    while (d_idx + 1 < 14) {
      if (end_heading < Cobs_0[d_idx]) {
        end_heading = Cobs_0[d_idx];
      }

      d_idx++;
    }
  }

  if (!(end_heading == 0.0)) {
    if (!b_x_0) {
      idx = 1;
    } else {
      idx = 0;
      d_idx = 2;
      exitg1 = false;
      while ((!exitg1) && (d_idx < 14)) {
        if (!rtIsNaN(Cobs_0[d_idx - 1])) {
          idx = d_idx;
          exitg1 = true;
        } else {
          d_idx++;
        }
      }
    }

    if (idx == 0) {
      x_0 = Cobs_0[0];
    } else {
      x_0 = Cobs_0[idx - 1];
      while (idx + 1 < 14) {
        if (x_0 < Cobs_0[idx]) {
          x_0 = Cobs_0[idx];
        }

        idx++;
      }
    }

    for (shortest_distance_tmp = 0; shortest_distance_tmp < 13;
         shortest_distance_tmp++) {
      Cobs_0[shortest_distance_tmp] /= x_0;
    }
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

  for (shortest_distance_tmp = 0; shortest_distance_tmp < 13;
       shortest_distance_tmp++) {
    rtb_J_out_d[shortest_distance_tmp] += (30.0 * Cobs[shortest_distance_tmp] +
      10.0 * Cobs_0[shortest_distance_tmp]) + 30.0 * Clane[shortest_distance_tmp];
  }

  // End of MATLAB Function: '<S2>/DynamicPathPlanning1'

  // MATLAB Function: '<S2>/MATLAB Function1' incorporates:
  //   Inport: '<Root>/Path_flag'

  if (!rtIsNaN(rtb_J_out_d[0])) {
    idx = 1;
  } else {
    idx = 0;
    b_idx = 2;
    exitg1 = false;
    while ((!exitg1) && (b_idx < 14)) {
      if (!rtIsNaN(rtb_J_out_d[b_idx - 1])) {
        idx = b_idx;
        exitg1 = true;
      } else {
        b_idx++;
      }
    }
  }

  if (idx == 0) {
    idx = 1;
  } else {
    offset_2 = rtb_J_out_d[idx - 1];
    for (b_idx = idx; b_idx + 1 < 14; b_idx++) {
      if (offset_2 > rtb_J_out_d[b_idx]) {
        offset_2 = rtb_J_out_d[b_idx];
        idx = b_idx + 1;
      }
    }
  }

  if (rtU.Path_flag == 1.0) {
    b_idx = idx - 1;
  } else {
    b_idx = 6;
  }

  for (i = 0; i < 11; i++) {
    shortest_distance_tmp = 6 * b_idx + 1;
    d_idx = 6 * b_idx + 2;
    total_length = a[i] * a[i];
    iy = 6 * b_idx + 3;
    n = 6 * b_idx + 4;
    b_colj = 6 * b_idx + 5;

    // Update for UnitDelay: '<S2>/Unit Delay5'
    rtDW.UnitDelay5_DSTATE[i] = ((((rtb_XP_i[shortest_distance_tmp] * a[i] +
      rtb_XP_i[6 * b_idx]) + rtb_XP_i[d_idx] * total_length) + rtb_XP_i[iy] *
      rt_powd_snf(a[i], 3.0)) + rtb_XP_i[n] * rt_powd_snf(a[i], 4.0)) +
      rtb_XP_i[b_colj] * rt_powd_snf(a[i], 5.0);
    rtDW.UnitDelay5_DSTATE[i + 11] = ((((rtb_YP_g[shortest_distance_tmp] * a[i]
      + rtb_YP_g[6 * b_idx]) + rtb_YP_g[d_idx] * total_length) + rtb_YP_g[iy] *
      rt_powd_snf(a[i], 3.0)) + rtb_YP_g[n] * rt_powd_snf(a[i], 4.0)) +
      rtb_YP_g[b_colj] * rt_powd_snf(a[i], 5.0);
    x[i] = total_length;
    b_Path_dis_data[i] = rt_powd_snf(a[i], 3.0);
    K1_0[i] = rt_powd_snf(a[i], 4.0);
    K_11[i] = rt_powd_snf(a[i], 5.0);
    X2[i] = total_length;
    Y2[i] = rt_powd_snf(a[i], 3.0);
    K2[i] = rt_powd_snf(a[i], 4.0);
    K_12[i] = rt_powd_snf(a[i], 5.0);
  }

  for (shortest_distance_tmp = 0; shortest_distance_tmp < 6;
       shortest_distance_tmp++) {
    // MATLAB Function: '<S2>/MATLAB Function1'
    d_idx = 6 * b_idx + shortest_distance_tmp;

    // Outport: '<Root>/XP_final' incorporates:
    //   MATLAB Function: '<S2>/MATLAB Function1'

    rtY.XP_final[shortest_distance_tmp] = rtb_XP_i[d_idx];

    // Outport: '<Root>/YP_final' incorporates:
    //   MATLAB Function: '<S2>/MATLAB Function1'

    rtY.YP_final[shortest_distance_tmp] = rtb_YP_g[d_idx];

    // Outport: '<Root>/XP_final_1' incorporates:
    //   MATLAB Function: '<S2>/MATLAB Function1'

    rtY.XP_final_1[shortest_distance_tmp] = rtb_XP[d_idx];

    // Outport: '<Root>/YP_final_1' incorporates:
    //   MATLAB Function: '<S2>/MATLAB Function1'

    rtY.YP_final_1[shortest_distance_tmp] = rtb_YP[d_idx];
  }

  // Outport: '<Root>/X_UKF_SLAM'
  for (i = 0; i < 5; i++) {
    rtY.X_UKF_SLAM[i] = rtb_X[i];
  }

  // End of Outport: '<Root>/X_UKF_SLAM'

  // SignalConversion: '<S21>/TmpSignal ConversionAt SFunction Inport1' incorporates:
  //   Gain: '<S2>/Gain2'
  //   MATLAB Function: '<S2>/MATLAB Function'

  rtb_TmpSignalConversionAtSFun_0 = 0.017453292519943295 * rtb_Gain1;

  // MATLAB Function: '<S2>/MATLAB Function' incorporates:
  //   MATLAB Function: '<S2>/EndPointDecision'
  //   MATLAB Function: '<S2>/EndPointDecision1'
  //   MATLAB Function: '<S2>/MATLAB Function1'
  //   SignalConversion: '<S21>/TmpSignal ConversionAt SFunction Inport1'

  if (rtb_Gain_p <= head_err) {
    rtb_Look_ahead_S0 = rtb_Gain_p / head_err;
    rtb_Gain1 = ((((rtb_XP_i[6 * b_idx + 1] * rtb_Look_ahead_S0 + rtb_XP_i[6 *
                    b_idx]) + rtb_XP_i[6 * b_idx + 2] * (rtb_Look_ahead_S0 *
      rtb_Look_ahead_S0)) + rtb_XP_i[6 * b_idx + 3] * rt_powd_snf
                  (rtb_Look_ahead_S0, 3.0)) + rtb_XP_i[6 * b_idx + 4] *
                 rt_powd_snf(rtb_Look_ahead_S0, 4.0)) + rtb_XP_i[6 * b_idx + 5] *
      rt_powd_snf(rtb_Look_ahead_S0, 5.0);
    rtb_Look_ahead_S0 = ((((rtb_YP_g[6 * b_idx + 1] * rtb_Look_ahead_S0 +
      rtb_YP_g[6 * b_idx]) + rtb_YP_g[6 * b_idx + 2] * (rtb_Look_ahead_S0 *
      rtb_Look_ahead_S0)) + rtb_YP_g[6 * b_idx + 3] * rt_powd_snf
                          (rtb_Look_ahead_S0, 3.0)) + rtb_YP_g[6 * b_idx + 4] *
                         rt_powd_snf(rtb_Look_ahead_S0, 4.0)) + rtb_YP_g[6 *
      b_idx + 5] * rt_powd_snf(rtb_Look_ahead_S0, 5.0);
  } else if ((rtb_Gain_p > head_err) && (rtb_Gain_p <= head_err +
              rtb_Look_ahead_S0)) {
    rtb_Look_ahead_S0 = (rtb_Gain_p - head_err) / rtb_Look_ahead_S0;
    head_err = rtb_Look_ahead_S0 * rtb_Look_ahead_S0;
    rtb_Gain1 = ((((rtb_XP[6 * b_idx + 1] * rtb_Look_ahead_S0 + rtb_XP[6 * b_idx])
                   + rtb_XP[6 * b_idx + 2] * head_err) + rtb_XP[6 * b_idx + 3] *
                  rt_powd_snf(rtb_Look_ahead_S0, 3.0)) + rtb_XP[6 * b_idx + 4] *
                 rt_powd_snf(rtb_Look_ahead_S0, 4.0)) + rtb_XP[6 * b_idx + 5] *
      rt_powd_snf(rtb_Look_ahead_S0, 5.0);
    rtb_Look_ahead_S0 = ((((rtb_YP[6 * b_idx + 1] * rtb_Look_ahead_S0 + rtb_YP[6
      * b_idx]) + rtb_YP[6 * b_idx + 2] * head_err) + rtb_YP[6 * b_idx + 3] *
                          rt_powd_snf(rtb_Look_ahead_S0, 3.0)) + rtb_YP[6 *
                         b_idx + 4] * rt_powd_snf(rtb_Look_ahead_S0, 4.0)) +
      rtb_YP[6 * b_idx + 5] * rt_powd_snf(rtb_Look_ahead_S0, 5.0);
  } else {
    rtb_Gain1 = ((((rtb_XP[6 * b_idx + 1] + rtb_XP[6 * b_idx]) + rtb_XP[6 *
                   b_idx + 2]) + rtb_XP[6 * b_idx + 3]) + rtb_XP[6 * b_idx + 4])
      + rtb_XP[6 * b_idx + 5];
    rtb_Look_ahead_S0 = ((((rtb_YP[6 * b_idx + 1] + rtb_YP[6 * b_idx]) + rtb_YP
      [6 * b_idx + 2]) + rtb_YP[6 * b_idx + 3]) + rtb_YP[6 * b_idx + 4]) +
      rtb_YP[6 * b_idx + 5];
  }

  rtb_Gain1 -= rtb_X[0];
  head_err = rtb_Look_ahead_S0 - rtb_X[1];
  total_length = std::sin(-rtb_TmpSignalConversionAtSFun_0);
  rtb_Look_ahead_S0 = std::cos(-rtb_TmpSignalConversionAtSFun_0);

  // Outport: '<Root>/Vehicle_Target_x' incorporates:
  //   MATLAB Function: '<S2>/MATLAB Function'

  rtY.Vehicle_Target_x = rtb_Look_ahead_S0 * rtb_Gain1 + -total_length *
    head_err;

  // Outport: '<Root>/Vehicle_Target_y' incorporates:
  //   MATLAB Function: '<S2>/MATLAB Function'

  rtY.Vehicle_Target_y = total_length * rtb_Gain1 + rtb_Look_ahead_S0 * head_err;

  // Outport: '<Root>/J_minind' incorporates:
  //   MATLAB Function: '<S2>/MATLAB Function1'

  rtY.J_minind = idx;

  // Outport: '<Root>/J_finalind' incorporates:
  //   MATLAB Function: '<S2>/MATLAB Function1'

  rtY.J_finalind = b_idx + 1;

  // Outport: '<Root>/End_x' incorporates:
  //   MATLAB Function: '<S2>/EndPointDecision'

  rtY.End_x = ajj;

  // Outport: '<Root>/End_y' incorporates:
  //   MATLAB Function: '<S2>/EndPointDecision'

  rtY.End_y = table_tmp;

  // Outport: '<Root>/forward_length' incorporates:
  //   MATLAB Function: '<S2>/Forward_Length_Decision'

  rtY.forward_length = rtb_Gain_p;

  // Update for Memory: '<S2>/Memory1' incorporates:
  //   Constant: '<S2>/Constant7'

  rtDW.Memory1_PreviousInput = 1.0;

  // Update for Memory: '<S2>/Memory' incorporates:
  //   Constant: '<S2>/Constant8'

  rtDW.Memory_PreviousInput = 188.0;

  // MATLAB Function: '<S2>/Final_Static_Path'
  if (rtb_Add > 0.0) {
    // Update for UnitDelay: '<S2>/Unit Delay'
    rtDW.UnitDelay_DSTATE = rtDW.dist;
  }

  // Update for UnitDelay: '<S3>/Unit Delay1'
  for (i = 0; i < 5; i++) {
    rtDW.UnitDelay1_DSTATE[i] = rtb_X[i];
  }

  // End of Update for UnitDelay: '<S3>/Unit Delay1'

  // Update for UnitDelay: '<S3>/Unit Delay35' incorporates:
  //   Inport: '<Root>/SLAM_counter'
  //   MATLAB Function: '<S3>/SLAM_Check'

  rtDW.UnitDelay35_DSTATE[0] = SLAM_X_out;
  rtDW.UnitDelay35_DSTATE[1] = SLAM_Y_out;
  rtDW.UnitDelay35_DSTATE[2] = SLAM_Heading_out;
  rtDW.UnitDelay35_DSTATE[3] = rtU.SLAM_counter;

  // Update for UnitDelay: '<S3>/Unit Delay37'
  memcpy(&rtDW.UnitDelay37_DSTATE[0], &rtb_Q_last_o[0], 25U * sizeof(real_T));

  // Update for UnitDelay: '<S3>/Unit Delay36'
  memcpy(&rtDW.UnitDelay36_DSTATE[0], &rtb_R_last_o[0], 25U * sizeof(real_T));

  // Update for UnitDelay: '<S3>/Unit Delay34'
  for (i = 0; i < 5; i++) {
    rtDW.UnitDelay34_DSTATE[i] = rtb_X[i];
  }

  // End of Update for UnitDelay: '<S3>/Unit Delay34'

  // Update for UnitDelay: '<S3>/Unit Delay33'
  memcpy(&rtDW.UnitDelay33_DSTATE[0], &p0[0], 25U * sizeof(real_T));

  // MATLAB Function: '<S2>/MATLAB Function1'
  for (shortest_distance_tmp = 0; shortest_distance_tmp < 11;
       shortest_distance_tmp++) {
    // Update for UnitDelay: '<S2>/Unit Delay6'
    rtDW.UnitDelay6_DSTATE[shortest_distance_tmp] = ((((rtb_XP[6 * b_idx + 1] *
      a[shortest_distance_tmp] + rtb_XP[6 * b_idx]) + rtb_XP[6 * b_idx + 2] *
      x[shortest_distance_tmp]) + rtb_XP[6 * b_idx + 3] *
      b_Path_dis_data[shortest_distance_tmp]) + rtb_XP[6 * b_idx + 4] *
      K1_0[shortest_distance_tmp]) + rtb_XP[6 * b_idx + 5] *
      K_11[shortest_distance_tmp];
    rtDW.UnitDelay6_DSTATE[shortest_distance_tmp + 11] = ((((rtb_YP[6 * b_idx +
      1] * a[shortest_distance_tmp] + rtb_YP[6 * b_idx]) + rtb_YP[6 * b_idx + 2]
      * X2[shortest_distance_tmp]) + rtb_YP[6 * b_idx + 3] *
      Y2[shortest_distance_tmp]) + rtb_YP[6 * b_idx + 4] *
      K2[shortest_distance_tmp]) + rtb_YP[6 * b_idx + 5] *
      K_12[shortest_distance_tmp];
  }

  // Update for UnitDelay: '<S2>/Unit Delay8' incorporates:
  //   MATLAB Function: '<S2>/DangerousArea'

  rtDW.UnitDelay8_DSTATE = Length_1;

  // Update for UnitDelay: '<S2>/Unit Delay12' incorporates:
  //   MATLAB Function: '<S2>/DangerousArea'

  rtDW.UnitDelay12_DSTATE = target_k;

  // Update for UnitDelay: '<S2>/Unit Delay9'
  rtDW.UnitDelay9_DSTATE[0] = rtb_num_lane_direction_f[0];

  // Update for UnitDelay: '<S2>/Unit Delay10'
  rtDW.UnitDelay10_DSTATE[0] = rtb_H_y_out[0];

  // Update for UnitDelay: '<S2>/Unit Delay9'
  rtDW.UnitDelay9_DSTATE[1] = rtb_num_lane_direction_f[1];

  // Update for UnitDelay: '<S2>/Unit Delay10'
  rtDW.UnitDelay10_DSTATE[1] = rtb_H_y_out[1];

  // Update for UnitDelay: '<S2>/Unit Delay9'
  rtDW.UnitDelay9_DSTATE[2] = rtb_num_lane_direction_f[2];

  // Update for UnitDelay: '<S2>/Unit Delay10'
  rtDW.UnitDelay10_DSTATE[2] = rtb_H_y_out[2];

  // Update for UnitDelay: '<S2>/Unit Delay9'
  rtDW.UnitDelay9_DSTATE[3] = rtb_num_lane_direction_f[3];

  // Update for UnitDelay: '<S2>/Unit Delay10'
  rtDW.UnitDelay10_DSTATE[3] = rtb_H_y_out[3];
}

// Model initialize function
void MM_DPP_1ModelClass::initialize()
{
  // Registration code

  // initialize non-finites
  rt_InitInfAndNaN(sizeof(real_T));

  // InitializeConditions for UnitDelay: '<S3>/Unit Delay37'
  memcpy(&rtDW.UnitDelay37_DSTATE[0], &rtConstP.UnitDelay37_InitialCondition[0],
         25U * sizeof(real_T));

  // InitializeConditions for UnitDelay: '<S3>/Unit Delay36'
  memcpy(&rtDW.UnitDelay36_DSTATE[0], &rtConstP.UnitDelay36_InitialCondition[0],
         25U * sizeof(real_T));

  // InitializeConditions for UnitDelay: '<S3>/Unit Delay33'
  memcpy(&rtDW.UnitDelay33_DSTATE[0], &rtConstP.UnitDelay33_InitialCondition[0],
         25U * sizeof(real_T));
}

// Constructor
MM_DPP_1ModelClass::MM_DPP_1ModelClass()
{
}

// Destructor
MM_DPP_1ModelClass::~MM_DPP_1ModelClass()
{
  // Currently there is no destructor body generated.
}

// Real-Time Model get method
RT_MODEL * MM_DPP_1ModelClass::getRTM()
{
  return (&rtM);
}

//
// File trailer for generated code.
//
// [EOF]
//
