//
// File: MM_TP.cpp
//
// Code generated for Simulink model 'MM_TP'.
//
// Model version                  : 1.28
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Thu Sep  5 15:35:30 2019
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
  int32_T jmax;
  int32_T jj;
  real_T ajj;
  int32_T colj;
  int32_T ix;
  int32_T iy;
  int8_T ii_data[5];
  real_T K1[4];
  real_T x[11];
  real_T b_a;
  int8_T I[25];
  real_T oi_xy_data[376];
  real_T dist_op_data[188];
  real_T total_length;
  real_T Forward_Static_Path_id_0_data[188];
  int16_T v_data[188];
  real_T diffheading;
  real_T count_1;
  real_T target_k;
  real_T Length_1;
  real_T x_0;
  real_T OBXY_m[8];
  real_T c;
  real_T delta_offset;
  real_T offset_2;
  real_T offset_3;
  real_T offset_4;
  real_T offset_5;
  real_T offset_6;
  real_T offset[13];
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
  real_T x_endpoint9;
  real_T x_endpoint10;
  real_T x_endpoint11;
  real_T x_endpoint12;
  real_T x_endpoint13;
  real_T X_2[143];
  real_T Y[143];
  real_T K[143];
  real_T K_1[143];
  real_T L_path[13];
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
  real_T sigma;
  real_T Clane[13];
  real_T LastPath_overlap_data[22];
  real_T Path_overlap_data[22];
  real_T Path_dis_data[121];
  real_T YP1[6];
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
  real_T e_maxval[13];
  real_T g_maxval[13];
  real_T LastPath[22];
  real_T XY_difflen_0[11];
  real_T X_NV[11];
  int32_T c_data[188];
  int32_T d_data[188];
  int32_T e_data[188];
  real_T table[376];
  real_T shortest_distance[188];
  int8_T settled[188];
  uint8_T pidx_data[188];
  uint8_T zz_data[188];
  real_T tmp_path_data[188];
  uint8_T nidx_data[188];
  uint8_T c_data_0[188];
  int32_T ii_data_0[188];
  int32_T idx;
  int32_T b_idx;
  int32_T n;
  boolean_T x_data[188];
  boolean_T b_x[188];
  uint8_T f_ii_data[188];
  real_T rtb_Add;
  real_T rtb_Gain1;
  real_T rtb_Gain_p;
  real_T rtb_X[5];
  real_T rtb_XP_final_g[6];
  real_T rtb_YP_final_o[6];
  real_T rtb_UnitDelay34[5];
  real_T rtb_X_state[5];
  real_T rtb_Oi_near_l[2];
  real_T rtb_num_lane_direction_f[4];
  real_T rtb_H_y_out[4];
  real_T rtb_J_out[13];
  real_T rtb_Q_last_o[25];
  real_T rtb_R_last_o[25];
  real_T rtb_X_AUG[55];
  real_T rtb_K[25];
  real_T rtb_V_boundingbox[400];
  real_T rtb_Forward_Static_Path_x_h[188];
  real_T rtb_Forward_Static_Path_y_p[188];
  real_T rtb_Forward_Static_Path_id[188];
  int32_T i;
  real_T tmp[4];
  real_T sigma_0[4];
  real_T rtb_X_AUG_0[10];
  real_T LastPath_overlap_data_0[22];
  int32_T loop_ub;
  real_T p_sqrt_data_0[25];
  real_T rtb_X_state_0[2];
  real_T rtb_num_lane_direction_b[4];
  int16_T tmp_0;
  real_T rtb_Oi_near_o[8];
  real_T rtb_Oi_near_o_0[8];
  real_T OBXY_EL_0[8];
  real_T b_Path_dis_data_0[121];
  real_T rtb_YP_final_l[78];
  real_T YP1_0[78];
  real_T Length_1_0[26];
  int32_T oi_xy_size[2];
  int32_T Path_overlap_size[2];
  int32_T Path_overlap_size_0[2];
  int32_T LastPath_overlap_size[2];
  int32_T Path_overlap_size_1[2];
  int32_T LastPath_overlap_size_0[2];
  boolean_T b_x_0;
  int32_T Path_RES_1_size_idx_0;
  real_T yy_idx_0;
  int32_T xy_ends_POS_size_idx_0;
  int32_T Path_RES_0_size_idx_1;
  real_T sigma_tmp;
  int32_T total_length_tmp;
  real_T Path_vehFRY_tmp;
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
    jmax = 1;
    exitg1 = false;
    while ((!exitg1) && (jmax < 189)) {
      if (rtConstP.Constant3_Value[jmax - 1] == 1.0) {
        idx++;
        ii_data_0[idx - 1] = jmax;
        if (idx >= 188) {
          exitg1 = true;
        } else {
          jmax++;
        }
      } else {
        jmax++;
      }
    }

    if (1 > idx) {
      idx = 0;
    }

    for (i = 0; i < idx; i++) {
      pidx_data[i] = (uint8_T)ii_data_0[i];
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

    for (i = 0; i < idx; i++) {
      zz_data[i] = (uint8_T)ii_data_0[i];
    }

    do {
      exitg2 = 0;
      i = zz_data[0] - 1;
      if (settled[i] == 0) {
        for (total_length_tmp = 0; total_length_tmp < 188; total_length_tmp++) {
          table[total_length_tmp] = table[188 + total_length_tmp];
        }

        colj = pidx_data[0] + 187;
        table[colj] = 0.0;
        iy = 0;
        ix = 0;
        for (b_idx = 0; b_idx < 188; b_idx++) {
          b_x_0 = (rtConstP.Constant3_Value[pidx_data[0] - 1] ==
                   rtConstP.Constant5_Value[188 + b_idx]);
          if (b_x_0) {
            iy++;
            c_data_0[ix] = (uint8_T)(b_idx + 1);
            ix++;
          }
        }

        for (jmax = 0; jmax < iy; jmax++) {
          for (total_length_tmp = 0; total_length_tmp < 188; total_length_tmp++)
          {
            b_x[total_length_tmp] = (rtConstP.Constant5_Value[c_data_0[jmax] +
              375] == rtConstP.Constant3_Value[total_length_tmp]);
          }

          b_idx = -1;
          idx = 1;
          exitg1 = false;
          while ((!exitg1) && (idx < 189)) {
            if (b_x[idx - 1]) {
              b_idx++;
              ii_data_0[b_idx] = idx;
              if (b_idx + 1 >= 188) {
                exitg1 = true;
              } else {
                idx++;
              }
            } else {
              idx++;
            }
          }

          if (!(settled[ii_data_0[0] - 1] != 0)) {
            rtb_X_state_0[0] = rtConstP.Constant3_Value[colj] -
              rtConstP.Constant3_Value[ii_data_0[0] + 187];
            rtb_X_state_0[1] = rtConstP.Constant3_Value[pidx_data[0] + 375] -
              rtConstP.Constant3_Value[ii_data_0[0] + 375];
            power_n(rtb_X_state_0, rtb_Oi_near_l);
            ajj = std::sqrt(sum_e(rtb_Oi_near_l));
            if ((table[ii_data_0[0] - 1] == 0.0) || (table[ii_data_0[0] - 1] >
                 table[pidx_data[0] - 1] + ajj)) {
              table[ii_data_0[0] + 187] = table[pidx_data[0] - 1] + ajj;
              for (total_length_tmp = 0; total_length_tmp < 188;
                   total_length_tmp++) {
                b_x[total_length_tmp] = (rtDW.path[(188 * total_length_tmp +
                  pidx_data[0]) - 1] != 0.0);
              }

              idx = 0;
              b_idx = 1;
              exitg1 = false;
              while ((!exitg1) && (b_idx < 189)) {
                if (b_x[b_idx - 1]) {
                  idx++;
                  f_ii_data[idx - 1] = (uint8_T)b_idx;
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
                n = 0;
              } else {
                n = idx;
              }

              loop_ub = n - 1;
              if (0 <= loop_ub) {
                memset(&tmp_path_data[0], 0, (loop_ub + 1) * sizeof(real_T));
              }

              for (b_idx = 0; b_idx < n; b_idx++) {
                tmp_path_data[b_idx] = rtDW.path[((f_ii_data[b_idx] - 1) * 188 +
                  pidx_data[0]) - 1];
              }

              b_idx = ii_data_0[0] - 1;
              for (total_length_tmp = 0; total_length_tmp < n; total_length_tmp
                   ++) {
                rtDW.path[b_idx + 188 * total_length_tmp] =
                  tmp_path_data[total_length_tmp];
              }

              rtDW.path[b_idx + 188 * n] =
                rtConstP.Constant5_Value[c_data_0[jmax] + 375];
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

        for (total_length_tmp = 0; total_length_tmp < idx; total_length_tmp++) {
          nidx_data[total_length_tmp] = (uint8_T)ii_data_0[total_length_tmp];
        }

        if (idx <= 2) {
          if (idx == 1) {
            sigma = table[ii_data_0[0] + 187];
          } else if (table[ii_data_0[0] + 187] > table[ii_data_0[1] + 187]) {
            sigma = table[ii_data_0[1] + 187];
          } else if (rtIsNaN(table[ii_data_0[0] + 187])) {
            if (!rtIsNaN(table[ii_data_0[1] + 187])) {
              sigma = table[ii_data_0[1] + 187];
            } else {
              sigma = table[ii_data_0[0] + 187];
            }
          } else {
            sigma = table[ii_data_0[0] + 187];
          }
        } else {
          if (!rtIsNaN(table[ii_data_0[0] + 187])) {
            b_idx = 1;
          } else {
            b_idx = 0;
            jmax = 2;
            exitg1 = false;
            while ((!exitg1) && (jmax <= idx)) {
              if (!rtIsNaN(table[ii_data_0[jmax - 1] + 187])) {
                b_idx = jmax;
                exitg1 = true;
              } else {
                jmax++;
              }
            }
          }

          if (b_idx == 0) {
            sigma = table[ii_data_0[0] + 187];
          } else {
            sigma = table[ii_data_0[b_idx - 1] + 187];
            while (b_idx + 1 <= idx) {
              if (sigma > table[ii_data_0[b_idx] + 187]) {
                sigma = table[ii_data_0[b_idx] + 187];
              }

              b_idx++;
            }
          }
        }

        for (total_length_tmp = 0; total_length_tmp < idx; total_length_tmp++) {
          x_data[total_length_tmp] = (table[ii_data_0[total_length_tmp] + 187] ==
            sigma);
        }

        b_idx = 0;
        jmax = 1;
        exitg1 = false;
        while ((!exitg1) && (jmax <= idx)) {
          if (x_data[jmax - 1]) {
            b_idx++;
            ii_data_0[b_idx - 1] = jmax;
            if (b_idx >= idx) {
              exitg1 = true;
            } else {
              jmax++;
            }
          } else {
            jmax++;
          }
        }

        if (idx == 1) {
          if (b_idx == 0) {
            idx = 0;
          }
        } else if (1 > b_idx) {
          idx = 0;
        } else {
          idx = b_idx;
        }

        if (idx == 0) {
          exitg2 = 1;
        } else {
          pidx_data[0] = nidx_data[ii_data_0[0] - 1];
          i = nidx_data[ii_data_0[0] - 1] - 1;
          shortest_distance[i] = table[nidx_data[ii_data_0[0] - 1] + 187];
          settled[i] = 1;
        }
      } else {
        exitg2 = 1;
      }
    } while (exitg2 == 0);

    for (total_length_tmp = 0; total_length_tmp < 188; total_length_tmp++) {
      b_x[total_length_tmp] = (rtDW.path[(188 * total_length_tmp + zz_data[0]) -
        1] != 0.0);
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
      n = 0;
    } else {
      n = b_idx;
    }

    if (1 > n) {
      n = 0;
    }

    rtDW.dist = shortest_distance[i];
    rtDW.SFunction_DIMS3_c = n;
    for (i = 0; i < n; i++) {
      rtDW.path_2[i] = rtDW.path[(188 * i + zz_data[0]) - 1];
    }

    // End of MATLAB Function: '<S13>/Dijkstra'
  }

  // End of Outputs for SubSystem: '<S2>/Enabled Subsystem'

  // MATLAB Function: '<S2>/Final_Static_Path' incorporates:
  //   Constant: '<S2>/Constant6'

  if (!rtDW.path_out1_not_empty) {
    if (rtb_Add > 0.0) {
      rtDW.path_out1.size = rtDW.SFunction_DIMS3_c;
      for (i = 0; i < rtDW.SFunction_DIMS3_c; i++) {
        rtDW.path_out1.data[i] = rtDW.path_2[i];
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
    for (i = 0; i < rtDW.SFunction_DIMS3_c; i++) {
      rtDW.path_out1.data[i] = rtDW.path_2[i];
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

  b_idx = 0;
  if (rtU.SLAM_counter != rtDW.UnitDelay35_DSTATE[3]) {
    b_idx = 1;

    // Update for UnitDelay: '<S3>/Unit Delay38'
    rtDW.UnitDelay38_DSTATE = 0.0;
  } else {
    // Update for UnitDelay: '<S3>/Unit Delay38'
    rtDW.UnitDelay38_DSTATE++;
  }

  if (rtU.SLAM_fault == 1.0) {
    b_idx = 0;
  }

  if (b_idx == 0) {
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
    for (n = 0; n < 5; n++) {
      rtb_Q_last_o[n + 5 * n] = b[n];
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
    for (n = 0; n < 5; n++) {
      rtb_R_last_o[n + 5 * n] = rtb_X_state[n];
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
  colj = 0;
  n = 1;
  exitg1 = false;
  while ((!exitg1) && (n <= 5)) {
    jj = (colj + n) - 1;
    ajj = 0.0;
    if (!(n - 1 < 1)) {
      ix = colj;
      iy = colj;
      for (jmax = 1; jmax < n; jmax++) {
        ajj += p_sqrt_data[ix] * p_sqrt_data[iy];
        ix++;
        iy++;
      }
    }

    ajj = p_sqrt_data[jj] - ajj;
    if (ajj > 0.0) {
      ajj = std::sqrt(ajj);
      p_sqrt_data[jj] = ajj;
      if (n < 5) {
        if (n - 1 != 0) {
          jmax = jj + 5;
          iy = ((4 - n) * 5 + colj) + 6;
          for (total_length_tmp = colj + 6; total_length_tmp <= iy;
               total_length_tmp += 5) {
            i = colj;
            target_k = 0.0;
            ix = (total_length_tmp + n) - 2;
            for (loop_ub = total_length_tmp; loop_ub <= ix; loop_ub++) {
              target_k += p_sqrt_data[loop_ub - 1] * p_sqrt_data[i];
              i++;
            }

            p_sqrt_data[jmax] += -target_k;
            jmax += 5;
          }
        }

        ajj = 1.0 / ajj;
        iy = ((4 - n) * 5 + jj) + 6;
        for (jmax = jj + 5; jmax + 1 <= iy; jmax += 5) {
          p_sqrt_data[jmax] *= ajj;
        }

        colj += 5;
      }

      n++;
    } else {
      p_sqrt_data[jj] = ajj;
      idx = n;
      exitg1 = true;
    }
  }

  if (idx == 0) {
    jmax = 5;
  } else {
    jmax = idx - 1;
  }

  for (n = 1; n <= jmax; n++) {
    for (ix = n; ix < jmax; ix++) {
      p_sqrt_data[ix + 5 * (n - 1)] = 0.0;
    }
  }

  if (1 > jmax) {
    jmax = 0;
    loop_ub = 0;
  } else {
    loop_ub = jmax;
  }

  for (i = 0; i < loop_ub; i++) {
    for (total_length_tmp = 0; total_length_tmp < jmax; total_length_tmp++) {
      p_sqrt_data_0[total_length_tmp + jmax * i] = p_sqrt_data[5 * i +
        total_length_tmp];
    }
  }

  for (i = 0; i < loop_ub; i++) {
    for (total_length_tmp = 0; total_length_tmp < jmax; total_length_tmp++) {
      iy = jmax * i;
      p_sqrt_data[total_length_tmp + iy] = p_sqrt_data_0[iy + total_length_tmp];
    }
  }

  memset(&rtb_X_AUG[0], 0, 55U * sizeof(real_T));
  if (idx != 0) {
    for (jmax = 0; jmax < 5; jmax++) {
      temp_dia[jmax] = std::abs(rtDW.UnitDelay33_DSTATE[5 * jmax + jmax]);
    }

    idx = 0;
    jmax = 1;
    exitg1 = false;
    while ((!exitg1) && (jmax < 6)) {
      if (temp_dia[jmax - 1] < 1.0E-10) {
        idx++;
        ii_data[idx - 1] = (int8_T)jmax;
        if (idx >= 5) {
          exitg1 = true;
        } else {
          jmax++;
        }
      } else {
        jmax++;
      }
    }

    if (!(1 > idx)) {
      for (i = 0; i < idx; i++) {
        temp_dia[ii_data[i] - 1] = 1.0E-10;
      }
    }

    memset(&p0[0], 0, 25U * sizeof(real_T));
    for (idx = 0; idx < 5; idx++) {
      p0[idx + 5 * idx] = temp_dia[idx];
    }

    jmax = 0;
    ix = 0;
    idx = 1;
    exitg1 = false;
    while ((!exitg1) && (idx < 6)) {
      colj = (ix + idx) - 1;
      ajj = 0.0;
      if (!(idx - 1 < 1)) {
        n = ix;
        iy = ix;
        for (jj = 1; jj < idx; jj++) {
          ajj += p0[n] * p0[iy];
          n++;
          iy++;
        }
      }

      ajj = p0[colj] - ajj;
      if (ajj > 0.0) {
        ajj = std::sqrt(ajj);
        p0[colj] = ajj;
        if (idx < 5) {
          if (idx - 1 != 0) {
            jj = colj + 5;
            n = ((4 - idx) * 5 + ix) + 6;
            for (total_length_tmp = ix + 6; total_length_tmp <= n;
                 total_length_tmp += 5) {
              i = ix;
              target_k = 0.0;
              iy = (total_length_tmp + idx) - 2;
              for (loop_ub = total_length_tmp; loop_ub <= iy; loop_ub++) {
                target_k += p0[loop_ub - 1] * p0[i];
                i++;
              }

              p0[jj] += -target_k;
              jj += 5;
            }
          }

          ajj = 1.0 / ajj;
          n = ((4 - idx) * 5 + colj) + 6;
          for (jj = colj + 5; jj + 1 <= n; jj += 5) {
            p0[jj] *= ajj;
          }

          ix += 5;
        }

        idx++;
      } else {
        p0[colj] = ajj;
        jmax = idx;
        exitg1 = true;
      }
    }

    if (jmax == 0) {
      n = 5;
    } else {
      n = jmax - 1;
    }

    for (jmax = 0; jmax < n; jmax++) {
      for (idx = jmax + 1; idx < n; idx++) {
        p0[idx + 5 * jmax] = 0.0;
      }
    }

    jmax = 5;
    loop_ub = 5;
    memcpy(&p_sqrt_data[0], &p0[0], 25U * sizeof(real_T));
  }

  for (i = 0; i < jmax; i++) {
    for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
      p_sqrt_data_0[total_length_tmp + loop_ub * i] = p_sqrt_data[jmax *
        total_length_tmp + i] * 2.23606797749979;
    }
  }

  for (i = 0; i < jmax; i++) {
    for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
      iy = loop_ub * i;
      p_sqrt_data[total_length_tmp + iy] = p_sqrt_data_0[iy + total_length_tmp];
    }
  }

  for (i = 0; i < 5; i++) {
    rtb_X_AUG[i] = rtb_UnitDelay34[i];
  }

  for (ix = 0; ix < 5; ix++) {
    jj = loop_ub - 1;
    for (i = 0; i <= jj; i++) {
      temp_dia[i] = p_sqrt_data[loop_ub * ix + i];
    }

    i = ix + 2;
    for (total_length_tmp = 0; total_length_tmp < 5; total_length_tmp++) {
      rtb_X_AUG[total_length_tmp + 5 * (i - 1)] =
        rtb_UnitDelay34[total_length_tmp] + temp_dia[total_length_tmp];
    }
  }

  for (idx = 0; idx < 5; idx++) {
    jj = loop_ub - 1;
    for (i = 0; i <= jj; i++) {
      temp_dia[i] = p_sqrt_data[loop_ub * idx + i];
    }

    i = idx + 7;
    for (total_length_tmp = 0; total_length_tmp < 5; total_length_tmp++) {
      rtb_X_AUG[total_length_tmp + 5 * (i - 1)] =
        rtb_UnitDelay34[total_length_tmp] - temp_dia[total_length_tmp];
    }
  }

  // End of MATLAB Function: '<S3>/SLAM_Generate_sigma_pt_UKF'

  // MATLAB Function: '<S3>/SLAM_UKF' incorporates:
  //   Constant: '<Root>/[Para] D_GC'
  //   Constant: '<S1>/Constant25'
  //   MATLAB Function: '<S3>/SLAM_Check'
  //   SignalConversion: '<S6>/TmpSignal ConversionAt SFunction Inport5'

  sigma_tmp = 0.01 * rtb_Gain_p * 3.8;
  for (jmax = 0; jmax < 11; jmax++) {
    rtb_X_AUG[5 * jmax] = (rtb_X_AUG[5 * jmax + 4] * 0.01 * std::cos(rtb_X_AUG[5
      * jmax + 2]) + rtb_X_AUG[5 * jmax]) + std::cos(rtb_X_AUG[5 * jmax + 2] +
      1.5707963267948966) * sigma_tmp;
    rtb_X_AUG[1 + 5 * jmax] = (rtb_X_AUG[5 * jmax + 4] * 0.01 * std::sin
      (rtb_X_AUG[5 * jmax + 2]) + rtb_X_AUG[5 * jmax + 1]) + std::sin(rtb_X_AUG
      [5 * jmax + 2] + 1.5707963267948966) * sigma_tmp;
    rtb_X_AUG[2 + 5 * jmax] += rtb_X_AUG[5 * jmax + 3] * 0.01;
  }

  for (i = 0; i < 10; i++) {
    rtb_X_AUG_0[i] = rtb_X_AUG[(1 + i) * 5];
  }

  rtb_X[0] = rtb_X_AUG[0] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (i = 0; i < 10; i++) {
    rtb_X_AUG_0[i] = rtb_X_AUG[(1 + i) * 5 + 1];
  }

  rtb_X[1] = rtb_X_AUG[1] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (i = 0; i < 10; i++) {
    rtb_X_AUG_0[i] = rtb_X_AUG[(1 + i) * 5 + 2];
  }

  rtb_X[2] = rtb_X_AUG[2] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (i = 0; i < 10; i++) {
    rtb_X_AUG_0[i] = rtb_X_AUG[(1 + i) * 5 + 3];
  }

  rtb_X[3] = rtb_X_AUG[3] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (i = 0; i < 10; i++) {
    rtb_X_AUG_0[i] = rtb_X_AUG[(1 + i) * 5 + 4];
  }

  rtb_X[4] = rtb_X_AUG[4] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (i = 0; i < 5; i++) {
    rtb_Gain1 = rtb_X_AUG[i] - rtb_X[i];
    rtb_UnitDelay34[i] = rtb_Gain1;
    temp_dia[i] = rtb_Gain1;
  }

  for (i = 0; i < 5; i++) {
    for (total_length_tmp = 0; total_length_tmp < 5; total_length_tmp++) {
      p_sqrt_data[i + 5 * total_length_tmp] = rtb_UnitDelay34[i] *
        temp_dia[total_length_tmp];
    }
  }

  for (i = 0; i < 5; i++) {
    for (total_length_tmp = 0; total_length_tmp < 5; total_length_tmp++) {
      p0[total_length_tmp + 5 * i] = p_sqrt_data[5 * i + total_length_tmp] * 2.0;
    }
  }

  for (ix = 0; ix < 10; ix++) {
    for (i = 0; i < 5; i++) {
      rtb_Gain1 = rtb_X_AUG[(ix + 1) * 5 + i] - rtb_X[i];
      rtb_UnitDelay34[i] = rtb_Gain1;
      temp_dia[i] = rtb_Gain1;
    }

    for (i = 0; i < 5; i++) {
      for (total_length_tmp = 0; total_length_tmp < 5; total_length_tmp++) {
        p_sqrt_data[i + 5 * total_length_tmp] = rtb_UnitDelay34[i] *
          temp_dia[total_length_tmp];
      }
    }

    for (i = 0; i < 5; i++) {
      for (total_length_tmp = 0; total_length_tmp < 5; total_length_tmp++) {
        jmax = 5 * i + total_length_tmp;
        p0[total_length_tmp + 5 * i] = p_sqrt_data[jmax] * 0.1 + p0[jmax];
      }
    }
  }

  for (i = 0; i < 25; i++) {
    p0[i] += rtb_Q_last_o[i];
  }

  if (rtb_X[2] < 0.0) {
    rtb_X[2] += 6.2831853071795862;
  } else {
    if (rtb_X[2] >= 6.2831853071795862) {
      rtb_X[2] -= 6.2831853071795862;
    }
  }

  if (b_idx > 0) {
    for (i = 0; i < 25; i++) {
      p_sqrt_data[i] = p0[i] + rtb_R_last_o[i];
    }

    invNxN(p_sqrt_data, p_sqrt_data_0);
    for (i = 0; i < 5; i++) {
      for (total_length_tmp = 0; total_length_tmp < 5; total_length_tmp++) {
        b_idx = i + 5 * total_length_tmp;
        rtb_K[b_idx] = 0.0;
        for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 5;
             Path_RES_0_size_idx_1++) {
          rtb_K[b_idx] = p0[5 * Path_RES_0_size_idx_1 + i] * p_sqrt_data_0[5 *
            total_length_tmp + Path_RES_0_size_idx_1] + rtb_K[5 *
            total_length_tmp + i];
        }
      }

      rtb_UnitDelay34[i] = rtb_X_state[i] - rtb_X[i];
    }

    for (i = 0; i < 5; i++) {
      sigma = 0.0;
      for (total_length_tmp = 0; total_length_tmp < 5; total_length_tmp++) {
        sigma += rtb_K[5 * total_length_tmp + i] *
          rtb_UnitDelay34[total_length_tmp];
      }

      rtb_X[i] += sigma;
    }

    for (i = 0; i < 25; i++) {
      I[i] = 0;
    }

    for (idx = 0; idx < 5; idx++) {
      I[idx + 5 * idx] = 1;
    }

    for (i = 0; i < 5; i++) {
      for (total_length_tmp = 0; total_length_tmp < 5; total_length_tmp++) {
        b_idx = 5 * i + total_length_tmp;
        p_sqrt_data[total_length_tmp + 5 * i] = (real_T)I[b_idx] - rtb_K[b_idx];
      }
    }

    for (i = 0; i < 5; i++) {
      for (total_length_tmp = 0; total_length_tmp < 5; total_length_tmp++) {
        b_idx = total_length_tmp + 5 * i;
        rtb_K[b_idx] = 0.0;
        for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 5;
             Path_RES_0_size_idx_1++) {
          rtb_K[b_idx] = p_sqrt_data[5 * Path_RES_0_size_idx_1 +
            total_length_tmp] * p0[5 * i + Path_RES_0_size_idx_1] + rtb_K[5 * i
            + total_length_tmp];
        }
      }
    }

    for (i = 0; i < 5; i++) {
      for (total_length_tmp = 0; total_length_tmp < 5; total_length_tmp++) {
        p0[total_length_tmp + 5 * i] = rtb_K[5 * i + total_length_tmp];
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

    for (i = 0; i < 2; i++) {
      K1[i] = 0.0;
      K1[i] += p0[i + 18] * rtb_H_y_out[0];
      K1[i] += p0[i + 23] * rtb_H_y_out[1];
      K1[i + 2] = 0.0;
      K1[i + 2] += p0[i + 18] * rtb_H_y_out[2];
      K1[i + 2] += p0[i + 23] * rtb_H_y_out[3];
      rtb_X_state_0[i] = rtb_X_state[3 + i] - rtb_X[3 + i];
    }

    rtb_X[3] += K1[0] * rtb_X_state_0[0] + K1[2] * rtb_X_state_0[1];
    rtb_X[4] += K1[1] * rtb_X_state_0[0] + K1[3] * rtb_X_state_0[1];
    rtb_num_lane_direction_f[0] = 1.0 - K1[0];
    rtb_num_lane_direction_f[1] = 0.0 - K1[1];
    rtb_num_lane_direction_f[2] = 0.0 - K1[2];
    rtb_num_lane_direction_f[3] = 1.0 - K1[3];
    for (i = 0; i < 2; i++) {
      rtb_num_lane_direction_b[i] = 0.0;
      rtb_num_lane_direction_b[i] += rtb_num_lane_direction_f[i] * p0[18];
      rtb_num_lane_direction_b[i] += rtb_num_lane_direction_f[i + 2] * p0[19];
      rtb_num_lane_direction_b[i + 2] = 0.0;
      rtb_num_lane_direction_b[i + 2] += rtb_num_lane_direction_f[i] * p0[23];
      rtb_num_lane_direction_b[i + 2] += rtb_num_lane_direction_f[i + 2] * p0[24];
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
  iy = 0;
  do {
    exitg2 = 0;
    i = (int32_T)rtU.BB_num - 1;
    if (iy <= i) {
      sigma = (1.0 + (real_T)iy) * 2.0;
      for (ix = 0; ix < 4; ix++) {
        diffheading = std::sin(rtb_Gain1);
        c = std::cos(rtb_Gain1);
        rtb_V_boundingbox[((int32_T)(sigma + -1.0) + 100 * ix) - 1] =
          (rtU.BB_all_XY[((int32_T)(sigma + -1.0) + 100 * ix) - 1] * c +
           rtU.BB_all_XY[(100 * ix + (int32_T)sigma) - 1] * -diffheading) +
          rtb_X[0];
        rtb_V_boundingbox[((int32_T)sigma + 100 * ix) - 1] = (rtU.BB_all_XY
          [((int32_T)(sigma + -1.0) + 100 * ix) - 1] * diffheading +
          rtU.BB_all_XY[(100 * ix + (int32_T)sigma) - 1] * c) + rtb_X[1];
      }

      iy++;
    } else {
      exitg2 = 1;
    }
  } while (exitg2 == 0);

  // MATLAB Function: '<S3>/SLAM_UKF_MM' incorporates:
  //   Constant: '<S3>/Constant4'

  for (ix = 0; ix < 188; ix++) {
    if (rtConstP.pooled2[940 + ix] == (rtInf)) {
      table[ix] = rtConstP.pooled2[188 + ix];
      table[188 + ix] = rtb_X[1];
    } else if (rtConstP.pooled2[940 + ix] == 0.0) {
      table[ix] = rtb_X[0];
      table[188 + ix] = rtConstP.pooled2[376 + ix];
    } else {
      rtb_Gain_p = -1.0 / rtConstP.pooled2[940 + ix];
      ajj = rtb_X[1] - rtb_Gain_p * rtb_X[0];
      rtb_Gain1 = rtConstP.pooled2[940 + ix] - rtb_Gain_p;
      table[ix] = (ajj - rtConstP.pooled2[1128 + ix]) / rtb_Gain1;
      table[188 + ix] = (rtConstP.pooled2[940 + ix] * ajj - rtConstP.pooled2
                         [1128 + ix] * rtb_Gain_p) / rtb_Gain1;
    }

    sigma = table[ix] - rtb_X[0];
    b_a = table[188 + ix] - rtb_X[1];
    shortest_distance[ix] = std::sqrt(sigma * sigma + b_a * b_a);
  }

  rtb_X_state_0[0] = rtb_X[0];
  rtb_X_state_0[1] = rtb_X[1];
  MM(rtb_X[2] * 180.0 / 3.1415926535897931, rtb_X_state_0, table,
     shortest_distance, rtConstP.pooled2, &c, &ajj, rtb_Oi_near_l, &rtb_Gain_p,
     &diffheading, &count_1, rtb_num_lane_direction_f, &total_length);

  // End of MATLAB Function: '<S3>/SLAM_UKF_MM'

  // Gain: '<S2>/Gain3'
  rtb_Gain1 = 57.295779513082323 * rtb_X[2];

  // MATLAB Function: '<S2>/MM' incorporates:
  //   Gain: '<S2>/Gain'

  idx = rtDW.SFunction_DIMS4_h[0];
  oi_xy_size[0] = rtDW.SFunction_DIMS4_h[0];
  oi_xy_size[1] = 2;
  loop_ub = (rtDW.SFunction_DIMS4_h[0] << 1) - 1;
  if (0 <= loop_ub) {
    memset(&oi_xy_data[0], 0, (loop_ub + 1) * sizeof(real_T));
  }

  n = rtDW.SFunction_DIMS4_h[0];
  if (0 <= idx - 1) {
    memset(&dist_op_data[0], 0, idx * sizeof(real_T));
  }

  for (ix = 0; ix < rtDW.SFunction_DIMS4_h[0]; ix++) {
    if (rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 5 + ix] == (rtInf)) {
      oi_xy_data[ix] = rtDW.Static_Path_0[ix + rtDW.SFunction_DIMS4_h[0]];
      oi_xy_data[ix + idx] = rtb_X[1];
    } else if (rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 5 + ix] == 0.0) {
      oi_xy_data[ix] = rtb_X[0];
      oi_xy_data[ix + idx] = rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] << 1)
        + ix];
    } else {
      rtb_Gain_p = -1.0 / rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 5 + ix];
      ajj = rtb_X[1] - rtb_Gain_p * rtb_X[0];
      diffheading = rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 5 + ix] -
        rtb_Gain_p;
      oi_xy_data[ix] = (ajj - rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 6 +
                        ix]) / diffheading;
      oi_xy_data[ix + idx] = (rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 5 +
        ix] * ajj - rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 6 + ix] *
        rtb_Gain_p) / diffheading;
    }
  }

  for (idx = 0; idx < oi_xy_size[0]; idx++) {
    sigma = oi_xy_data[idx] - rtb_X[0];
    b_a = oi_xy_data[idx + oi_xy_size[0]] - rtb_X[1];
    dist_op_data[idx] = std::sqrt(sigma * sigma + b_a * b_a);
  }

  rtb_X_state_0[0] = rtb_X[0];
  rtb_X_state_0[1] = rtb_X[1];
  MM_f(0.017453292519943295 * rtb_Gain1 * 180.0 / 3.1415926535897931,
       rtb_X_state_0, oi_xy_data, oi_xy_size, dist_op_data, &n,
       rtDW.Static_Path_0, rtDW.SFunction_DIMS4_h, &c, &ajj, rtb_Oi_near_l,
       &rtb_Gain_p, &diffheading, &count_1, rtb_num_lane_direction_f,
       &total_length);
  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    x_data[total_length_tmp] = (rtDW.Static_Path_0[total_length_tmp] == c);
  }

  idx = 0;
  b_x_0 = x_data[0];
  for (jmax = 1; jmax < rtDW.SFunction_DIMS4_h[0]; jmax++) {
    if ((int32_T)b_x_0 < (int32_T)x_data[jmax]) {
      b_x_0 = x_data[jmax];
      idx = jmax;
    }
  }

  // MATLAB Function: '<S2>/Forward_Seg_0' incorporates:
  //   MATLAB Function: '<S2>/MM'

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  if (0 <= loop_ub - 1) {
    memcpy(&dist_op_data[0], &rtDW.Static_Path_0[0], loop_ub * sizeof(real_T));
  }

  if (rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 3 +
                          rtDW.SFunction_DIMS4_h[0]) - 1] ==
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0]]) {
    n = (rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] << 2) +
          rtDW.SFunction_DIMS4_h[0]) - 1] ==
         rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] << 1]);
  } else {
    n = 0;
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    x_data[total_length_tmp] = (dist_op_data[total_length_tmp] == c);
  }

  b_idx = 1;
  b_x_0 = x_data[0];
  for (jmax = 2; jmax <= rtDW.SFunction_DIMS4_h[0]; jmax++) {
    if ((int32_T)b_x_0 < (int32_T)x_data[jmax - 1]) {
      b_x_0 = x_data[jmax - 1];
      b_idx = jmax;
    }
  }

  sigma = rtb_Oi_near_l[0] - rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 3 +
    b_idx) - 1];
  b_a = rtb_Oi_near_l[1] - rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] << 2)
    + b_idx) - 1];
  total_length = std::sqrt(sigma * sigma + b_a * b_a);
  colj = b_idx;
  iy = 0;
  jmax = 0;
  ix = 0;
  exitg1 = false;
  while ((!exitg1) && (ix <= rtDW.SFunction_DIMS4_h[0] - 1)) {
    if (total_length > 15.0) {
      jmax = colj;
      exitg1 = true;
    } else {
      total_length_tmp = b_idx + ix;
      Path_RES_0_size_idx_1 = total_length_tmp + 1;
      if (Path_RES_0_size_idx_1 <= rtDW.SFunction_DIMS4_h[0]) {
        total_length += rtDW.Static_Path_0[total_length_tmp +
          (rtDW.SFunction_DIMS4_h[0] << 3)];
        colj = Path_RES_0_size_idx_1;
        iy = 1;
        ix++;
      } else if (n == 1) {
        total_length_tmp -= rtDW.SFunction_DIMS4_h[0];
        total_length += rtDW.Static_Path_0[total_length_tmp +
          (rtDW.SFunction_DIMS4_h[0] << 3)];
        colj = total_length_tmp + 1;
        iy = 2;
        ix++;
      } else {
        jmax = colj;
        iy = 3;
        exitg1 = true;
      }
    }
  }

  n = rtDW.SFunction_DIMS4_h[0] - 1;
  if (0 <= n) {
    memset(&Forward_Static_Path_id_0_data[0], 0, (n + 1) * sizeof(real_T));
  }

  if ((iy == 1) || (iy == 0)) {
    if (b_idx > jmax) {
      jj = 1;
      n = 0;
    } else {
      jj = b_idx;
      n = jmax;
    }

    loop_ub = n - jj;
    for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++)
    {
      Forward_Static_Path_id_0_data[total_length_tmp] = dist_op_data[(jj +
        total_length_tmp) - 1];
    }

    if (b_idx > jmax) {
      b_idx = 1;
      jmax = 0;
    }

    b_idx = (jmax - b_idx) + 1;
  } else if (iy == 2) {
    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      iy = 0;
      ix = 0;
    } else {
      iy = b_idx - 1;
      ix = rtDW.SFunction_DIMS4_h[0];
    }

    n = ((rtDW.SFunction_DIMS4_h[0] - b_idx) + jmax) + 1;
    if (1 > n) {
      tmp_0 = 0;
    } else {
      tmp_0 = (int16_T)n;
    }

    colj = tmp_0;
    loop_ub = tmp_0 - 1;
    for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++)
    {
      v_data[total_length_tmp] = (int16_T)total_length_tmp;
    }

    if (1 > jmax) {
      total_length_tmp = 0;
    } else {
      total_length_tmp = jmax;
    }

    loop_ub = total_length_tmp - 1;
    jj = ix - iy;
    for (total_length_tmp = 0; total_length_tmp < jj; total_length_tmp++) {
      table[total_length_tmp] = dist_op_data[iy + total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++)
    {
      table[(total_length_tmp + ix) - iy] = dist_op_data[total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp < colj; total_length_tmp++) {
      Forward_Static_Path_id_0_data[v_data[total_length_tmp]] =
        table[total_length_tmp];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_idx = 1;
      n = 2;
    } else {
      n = rtDW.SFunction_DIMS4_h[0] + 2;
    }

    if (2 > jmax) {
      iy = 1;
      jmax = 0;
    } else {
      iy = 2;
    }

    b_idx = ((n - b_idx) + jmax) - iy;
  } else {
    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      ix = 1;
      n = 0;
    } else {
      ix = b_idx;
      n = rtDW.SFunction_DIMS4_h[0];
    }

    loop_ub = n - ix;
    for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++)
    {
      Forward_Static_Path_id_0_data[total_length_tmp] = dist_op_data[(ix +
        total_length_tmp) - 1];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_idx = 1;
      jmax = 1;
    } else {
      jmax = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    b_idx = jmax - b_idx;
  }

  if (1 > b_idx) {
    b_idx = 0;
  }

  b_idx--;
  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    x_data[total_length_tmp] = (rtDW.Static_Path_0[total_length_tmp] ==
      Forward_Static_Path_id_0_data[b_idx]);
  }

  b_idx = 0;
  b_x_0 = x_data[0];
  for (jmax = 1; jmax < rtDW.SFunction_DIMS4_h[0]; jmax++) {
    if ((int32_T)b_x_0 < (int32_T)x_data[jmax]) {
      b_x_0 = x_data[jmax];
      b_idx = jmax;
    }
  }

  // MATLAB Function: '<S2>/Forward_Length_Decision' incorporates:
  //   Constant: '<S2>/Constant23'
  //   Inport: '<Root>/Look_ahead_time'
  //   Inport: '<Root>/Path_flag'
  //   Inport: '<Root>/Speed_mps'
  //   MATLAB Function: '<S2>/Forward_Seg_0'
  //   MATLAB Function: '<S2>/MM'
  //   UnitDelay: '<S2>/Unit Delay1'
  //   UnitDelay: '<S2>/Unit Delay2'

  diffheading = std::abs(rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 7 +
    b_idx] - rtb_Gain1);
  if (diffheading > 180.0) {
    diffheading = std::abs(diffheading - 360.0);
  }

  if (rtDW.UnitDelay2_DSTATE == 1.0) {
    if (diffheading > 10.0) {
      total_length_tmp = 1;
    } else {
      total_length_tmp = (ajj > rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] *
                          10 + idx] / 4.0);
    }
  } else if (rtDW.UnitDelay1_DSTATE_e == 1.0) {
    total_length_tmp = 0;
  } else {
    total_length_tmp = (diffheading > 45.0);
  }

  if (total_length_tmp == 1) {
    diffheading = rtU.Speed_mps * rtU.Look_ahead_time + 3.0;

    // Update for UnitDelay: '<S2>/Unit Delay2' incorporates:
    //   Inport: '<Root>/Look_ahead_time'
    //   Inport: '<Root>/Speed_mps'

    rtDW.UnitDelay2_DSTATE = 1.0;
  } else {
    if (rtU.Path_flag == 0.0) {
      diffheading = rtU.Speed_mps * rtU.Look_ahead_time + 3.0;
    } else {
      diffheading = 15.0;
    }

    // Update for UnitDelay: '<S2>/Unit Delay2' incorporates:
    //   Constant: '<S2>/Constant23'
    //   Inport: '<Root>/Look_ahead_time'
    //   Inport: '<Root>/Speed_mps'

    rtDW.UnitDelay2_DSTATE = 0.0;
  }

  // End of MATLAB Function: '<S2>/Forward_Length_Decision'

  // MATLAB Function: '<S2>/Forward_Seg' incorporates:
  //   MATLAB Function: '<S2>/MM'

  xy_ends_POS_size_idx_0 = rtDW.SFunction_DIMS4_h[0];
  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    rtDW.xy_ends_POS_data[total_length_tmp] =
      rtDW.Static_Path_0[total_length_tmp + rtDW.SFunction_DIMS4_h[0]];
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    rtDW.xy_ends_POS_data[total_length_tmp + xy_ends_POS_size_idx_0] =
      rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] << 1) + total_length_tmp];
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    rtDW.xy_ends_POS_data[total_length_tmp + (xy_ends_POS_size_idx_0 << 1)] =
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 3 + total_length_tmp];
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    rtDW.xy_ends_POS_data[total_length_tmp + xy_ends_POS_size_idx_0 * 3] =
      rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] << 2) + total_length_tmp];
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  if (0 <= loop_ub - 1) {
    memcpy(&dist_op_data[0], &rtDW.Static_Path_0[0], loop_ub * sizeof(real_T));
  }

  if (rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 3 +
                          rtDW.SFunction_DIMS4_h[0]) - 1] ==
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0]]) {
    n = (rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] << 2) +
          rtDW.SFunction_DIMS4_h[0]) - 1] ==
         rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] << 1]);
  } else {
    n = 0;
  }

  ix = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < ix; total_length_tmp++) {
    x_data[total_length_tmp] = (dist_op_data[total_length_tmp] == c);
  }

  b_idx = 1;
  b_x_0 = x_data[0];
  for (jmax = 2; jmax <= ix; jmax++) {
    if ((int32_T)b_x_0 < (int32_T)x_data[jmax - 1]) {
      b_x_0 = x_data[jmax - 1];
      b_idx = jmax;
    }
  }

  sigma = rtb_Oi_near_l[0] - rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 3 +
    b_idx) - 1];
  b_a = rtb_Oi_near_l[1] - rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] << 2)
    + b_idx) - 1];
  total_length = std::sqrt(sigma * sigma + b_a * b_a);
  colj = b_idx;
  iy = 0;
  jmax = 0;
  ix = 0;
  exitg1 = false;
  while ((!exitg1) && (ix <= rtDW.SFunction_DIMS4_h[0] - 1)) {
    if (total_length > diffheading) {
      jmax = colj;
      exitg1 = true;
    } else {
      total_length_tmp = b_idx + ix;
      Path_RES_0_size_idx_1 = total_length_tmp + 1;
      if (Path_RES_0_size_idx_1 <= rtDW.SFunction_DIMS4_h[0]) {
        total_length += rtDW.Static_Path_0[total_length_tmp +
          (rtDW.SFunction_DIMS4_h[0] << 3)];
        colj = Path_RES_0_size_idx_1;
        iy = 1;
        ix++;
      } else if (n == 1) {
        total_length_tmp -= rtDW.SFunction_DIMS4_h[0];
        total_length += rtDW.Static_Path_0[total_length_tmp +
          (rtDW.SFunction_DIMS4_h[0] << 3)];
        colj = total_length_tmp + 1;
        iy = 2;
        ix++;
      } else {
        jmax = colj;
        iy = 3;
        exitg1 = true;
      }
    }
  }

  ix = rtDW.SFunction_DIMS4_h[0];
  if (0 <= ix - 1) {
    memset(&Forward_Static_Path_id_0_data[0], 0, ix * sizeof(real_T));
  }

  if ((iy == 1) || (iy == 0)) {
    if (b_idx > jmax) {
      jj = 0;
      n = 0;
    } else {
      jj = b_idx - 1;
      n = jmax;
    }

    Path_RES_0_size_idx_1 = n - jj;
    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp] =
        rtDW.xy_ends_POS_data[jj + total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(jj + total_length_tmp) + xy_ends_POS_size_idx_0];
    }

    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + (Path_RES_0_size_idx_1 <<
        1)] = rtDW.xy_ends_POS_data[(jj + total_length_tmp) +
        (xy_ends_POS_size_idx_0 << 1)];
    }

    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + Path_RES_0_size_idx_1 *
        3] = rtDW.xy_ends_POS_data[(jj + total_length_tmp) +
        xy_ends_POS_size_idx_0 * 3];
    }

    if (b_idx > jmax) {
      iy = 1;
      total_length_tmp = 0;
    } else {
      iy = b_idx;
      total_length_tmp = jmax;
    }

    loop_ub = total_length_tmp - iy;
    for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++)
    {
      Forward_Static_Path_id_0_data[total_length_tmp] = dist_op_data[(iy +
        total_length_tmp) - 1];
    }

    if (b_idx > jmax) {
      b_idx = 1;
      jmax = 0;
    }

    b_idx = (jmax - b_idx) + 1;
  } else if (iy == 2) {
    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      n = 0;
      colj = 0;
    } else {
      n = b_idx - 1;
      colj = rtDW.SFunction_DIMS4_h[0];
    }

    if (1 > jmax) {
      loop_ub = 0;
    } else {
      loop_ub = jmax;
    }

    iy = colj - n;
    Path_RES_0_size_idx_1 = iy + loop_ub;
    for (total_length_tmp = 0; total_length_tmp < iy; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp] = rtDW.xy_ends_POS_data[n
        + total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp < iy; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(n + total_length_tmp) + xy_ends_POS_size_idx_0];
    }

    for (total_length_tmp = 0; total_length_tmp < iy; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + (Path_RES_0_size_idx_1 <<
        1)] = rtDW.xy_ends_POS_data[(n + total_length_tmp) +
        (xy_ends_POS_size_idx_0 << 1)];
    }

    for (total_length_tmp = 0; total_length_tmp < iy; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + Path_RES_0_size_idx_1 *
        3] = rtDW.xy_ends_POS_data[(n + total_length_tmp) +
        xy_ends_POS_size_idx_0 * 3];
    }

    for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[(total_length_tmp + colj) - n] =
        rtDW.xy_ends_POS_data[total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[((total_length_tmp + colj) - n) +
        Path_RES_0_size_idx_1] = rtDW.xy_ends_POS_data[total_length_tmp +
        xy_ends_POS_size_idx_0];
    }

    for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[((total_length_tmp + colj) - n) +
        (Path_RES_0_size_idx_1 << 1)] = rtDW.xy_ends_POS_data
        [(xy_ends_POS_size_idx_0 << 1) + total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[((total_length_tmp + colj) - n) +
        Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[xy_ends_POS_size_idx_0 * 3 + total_length_tmp];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      iy = 0;
      n = 0;
    } else {
      iy = b_idx - 1;
      n = rtDW.SFunction_DIMS4_h[0];
    }

    ix = ((rtDW.SFunction_DIMS4_h[0] - b_idx) + jmax) + 1;
    if (1 > ix) {
      tmp_0 = 0;
    } else {
      tmp_0 = (int16_T)ix;
    }

    colj = tmp_0;
    loop_ub = tmp_0 - 1;
    for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++)
    {
      v_data[total_length_tmp] = (int16_T)total_length_tmp;
    }

    if (1 > jmax) {
      total_length_tmp = 0;
    } else {
      total_length_tmp = jmax;
    }

    loop_ub = total_length_tmp - 1;
    jj = n - iy;
    for (total_length_tmp = 0; total_length_tmp < jj; total_length_tmp++) {
      table[total_length_tmp] = dist_op_data[iy + total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++)
    {
      table[(total_length_tmp + n) - iy] = dist_op_data[total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp < colj; total_length_tmp++) {
      Forward_Static_Path_id_0_data[v_data[total_length_tmp]] =
        table[total_length_tmp];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_idx = 1;
      ix = 1;
    } else {
      ix = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    if (1 > jmax) {
      jmax = 0;
    }

    b_idx = (ix - b_idx) + jmax;
  } else {
    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      n = 0;
      iy = 0;
    } else {
      n = b_idx - 1;
      iy = rtDW.SFunction_DIMS4_h[0];
    }

    Path_RES_0_size_idx_1 = iy - n;
    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp] = rtDW.xy_ends_POS_data[n
        + total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(n + total_length_tmp) + xy_ends_POS_size_idx_0];
    }

    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + (Path_RES_0_size_idx_1 <<
        1)] = rtDW.xy_ends_POS_data[(n + total_length_tmp) +
        (xy_ends_POS_size_idx_0 << 1)];
    }

    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + Path_RES_0_size_idx_1 *
        3] = rtDW.xy_ends_POS_data[(n + total_length_tmp) +
        xy_ends_POS_size_idx_0 * 3];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      ix = 1;
      n = 0;
    } else {
      ix = b_idx;
      n = rtDW.SFunction_DIMS4_h[0];
    }

    loop_ub = n - ix;
    for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++)
    {
      Forward_Static_Path_id_0_data[total_length_tmp] = dist_op_data[(ix +
        total_length_tmp) - 1];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_idx = 1;
      jmax = 1;
    } else {
      jmax = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    b_idx = jmax - b_idx;
  }

  if (1 > b_idx) {
    b_idx = 0;
  }

  rtDW.SFunction_DIMS4_f = b_idx;
  if (0 <= b_idx - 1) {
    memcpy(&shortest_distance[0], &Forward_Static_Path_id_0_data[0], b_idx *
           sizeof(real_T));
  }

  jmax = Path_RES_0_size_idx_1 + 1;
  loop_ub = (jmax << 1) - 1;
  if (0 <= loop_ub) {
    memset(&rtDW.Forward_Static_Path_data[0], 0, (loop_ub + 1) * sizeof(real_T));
  }

  loop_ub = Path_RES_0_size_idx_1 - 1;
  if (0 <= loop_ub) {
    memcpy(&rtDW.Forward_Static_Path_data[0], &rtDW.Static_Path_ends_POS_data[0],
           (loop_ub + 1) * sizeof(real_T));
  }

  for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++) {
    rtDW.Forward_Static_Path_data[total_length_tmp + jmax] =
      rtDW.Static_Path_ends_POS_data[total_length_tmp + Path_RES_0_size_idx_1];
  }

  total_length_tmp = Path_RES_0_size_idx_1 - 1;
  rtDW.Forward_Static_Path_data[Path_RES_0_size_idx_1] =
    rtDW.Static_Path_ends_POS_data[(Path_RES_0_size_idx_1 << 1) +
    total_length_tmp];
  rtDW.Forward_Static_Path_data[Path_RES_0_size_idx_1 + jmax] =
    rtDW.Static_Path_ends_POS_data[Path_RES_0_size_idx_1 * 3 + total_length_tmp];
  rtDW.SFunction_DIMS2_h = jmax;
  loop_ub = jmax - 1;
  if (0 <= loop_ub) {
    memcpy(&rtb_Forward_Static_Path_x_h[0], &rtDW.Forward_Static_Path_data[0],
           (loop_ub + 1) * sizeof(real_T));
  }

  rtDW.SFunction_DIMS3_k = jmax;
  loop_ub = jmax - 1;
  for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++) {
    rtb_Forward_Static_Path_y_p[total_length_tmp] =
      rtDW.Forward_Static_Path_data[total_length_tmp + jmax];
  }

  rtDW.SFunction_DIMS6[0] = rtDW.SFunction_DIMS4_h[0];
  rtDW.SFunction_DIMS6[1] = 1;

  // MATLAB Function: '<S2>/EndPointDecision'
  xy_ends_POS_size_idx_0 = 20000;
  Path_RES_0_size_idx_1 = 2;
  memset(&rtDW.Path_RES_0_data[0], 0, 40000U * sizeof(real_T));
  memset(&rtDW.Path_RES_0_1[0], 0, 40000U * sizeof(real_T));
  rtb_Gain_p = 0.0;
  count_1 = 0.0;
  iy = 0;
  target_k = std::floor(diffheading / 0.1);
  sigma = rtb_Forward_Static_Path_x_h[1] - rtb_Forward_Static_Path_x_h[0];
  b_a = rtb_Forward_Static_Path_y_p[1] - rtb_Forward_Static_Path_y_p[0];
  Length_1 = std::sqrt(sigma * sigma + b_a * b_a);
  ajj = rt_atan2d_snf(rtb_Forward_Static_Path_y_p[1] -
                      rtb_Forward_Static_Path_y_p[0],
                      rtb_Forward_Static_Path_x_h[1] -
                      rtb_Forward_Static_Path_x_h[0]);
  if (Length_1 > 0.1) {
    Length_1 = rt_roundd_snf(Length_1 / 0.1);
    for (n = 0; n < (int32_T)Length_1; n++) {
      count_1 = ((1.0 + (real_T)n) - 1.0) * 0.1;
      rtDW.Path_RES_0_1[n] = count_1 * std::cos(ajj) +
        rtb_Forward_Static_Path_x_h[0];
      rtDW.Path_RES_0_1[20000 + n] = count_1 * std::sin(ajj) +
        rtb_Forward_Static_Path_y_p[0];
      count_1 = 1.0 + (real_T)n;
    }
  } else {
    rtDW.Path_RES_0_1[0] = rtb_Forward_Static_Path_x_h[0];
    rtDW.Path_RES_0_1[20000] = rtb_Forward_Static_Path_y_p[0];
    count_1 = 1.0;
  }

  if (1.0 > count_1) {
    jj = 0;
  } else {
    jj = (int32_T)count_1;
  }

  Path_RES_1_size_idx_0 = jj;
  if (0 <= jj - 1) {
    memcpy(&rtDW.Path_RES_1_data[0], &rtDW.Path_RES_0_1[0], jj * sizeof(real_T));
  }

  for (total_length_tmp = 0; total_length_tmp < jj; total_length_tmp++) {
    rtDW.Path_RES_1_data[total_length_tmp + jj] =
      rtDW.Path_RES_0_1[total_length_tmp + 20000];
  }

  jmax = jj;
  for (total_length_tmp = 0; total_length_tmp < jj; total_length_tmp++) {
    rtDW.rtb_X_data[total_length_tmp] = rtb_X[0] -
      rtDW.Path_RES_1_data[total_length_tmp];
  }

  power_ec(rtDW.rtb_X_data, &jj, rtDW.tmp_data, &n);
  for (total_length_tmp = 0; total_length_tmp < jj; total_length_tmp++) {
    rtDW.rtb_X_data[total_length_tmp] = rtb_X[1] -
      rtDW.Path_RES_1_data[total_length_tmp + jj];
  }

  power_ec(rtDW.rtb_X_data, &jj, rtDW.tmp_data_c, &jmax);
  for (total_length_tmp = 0; total_length_tmp < n; total_length_tmp++) {
    rtDW.ob_distance_data[total_length_tmp] = rtDW.tmp_data[total_length_tmp] +
      rtDW.tmp_data_c[total_length_tmp];
  }

  if (n <= 2) {
    if (n == 1) {
      b_idx = 0;
    } else if (rtDW.ob_distance_data[0] > rtDW.ob_distance_data[1]) {
      b_idx = 1;
    } else if (rtIsNaN(rtDW.ob_distance_data[0])) {
      if (!rtIsNaN(rtDW.ob_distance_data[1])) {
        total_length_tmp = 2;
      } else {
        total_length_tmp = 1;
      }

      b_idx = total_length_tmp - 1;
    } else {
      b_idx = 0;
    }
  } else {
    if (!rtIsNaN(rtDW.ob_distance_data[0])) {
      b_idx = 0;
    } else {
      b_idx = -1;
      jmax = 2;
      exitg1 = false;
      while ((!exitg1) && (jmax <= n)) {
        if (!rtIsNaN(rtDW.ob_distance_data[jmax - 1])) {
          b_idx = jmax - 1;
          exitg1 = true;
        } else {
          jmax++;
        }
      }
    }

    if (b_idx + 1 == 0) {
      b_idx = 0;
    } else {
      sigma = rtDW.ob_distance_data[b_idx];
      for (jmax = b_idx + 1; jmax < n; jmax++) {
        if (sigma > rtDW.ob_distance_data[jmax]) {
          sigma = rtDW.ob_distance_data[jmax];
          b_idx = jmax;
        }
      }
    }
  }

  Length_1 = count_1 - (real_T)(b_idx + 1);
  if (rtDW.SFunction_DIMS2_h - 2 >= 1) {
    for (ix = 1; ix - 1 <= rtDW.SFunction_DIMS2_h - 3; ix++) {
      if (iy == 0) {
        b_a = rtb_Forward_Static_Path_x_h[ix + 1] -
          rtb_Forward_Static_Path_x_h[ix];
        sigma = rtb_Forward_Static_Path_y_p[ix + 1] -
          rtb_Forward_Static_Path_y_p[ix];
        ajj = std::sqrt(b_a * b_a + sigma * sigma);
        count_1 = rt_atan2d_snf(rtb_Forward_Static_Path_y_p[ix + 1] -
          rtb_Forward_Static_Path_y_p[ix], rtb_Forward_Static_Path_x_h[ix + 1] -
          rtb_Forward_Static_Path_x_h[ix]);
        if (ajj >= 0.1) {
          ajj = rt_roundd_snf(ajj / 0.1);
          for (n = 0; n < (int32_T)ajj; n++) {
            x_endpoint2 = ((1.0 + (real_T)n) - 1.0) * 0.1;
            jmax = (int32_T)((1.0 + (real_T)n) + rtb_Gain_p);
            rtDW.Path_RES_0_data[jmax - 1] = x_endpoint2 * std::cos(count_1) +
              rtb_Forward_Static_Path_x_h[ix];
            rtDW.Path_RES_0_data[jmax + 19999] = x_endpoint2 * std::sin(count_1)
              + rtb_Forward_Static_Path_y_p[ix];
          }

          rtb_Gain_p += ajj;
        } else {
          rtDW.Path_RES_0_data[(int32_T)(1.0 + rtb_Gain_p) - 1] =
            rtb_Forward_Static_Path_x_h[ix];
          rtDW.Path_RES_0_data[(int32_T)(1.0 + rtb_Gain_p) + 19999] =
            rtb_Forward_Static_Path_y_p[ix];
          rtb_Gain_p++;
        }

        if (rtb_Gain_p > target_k - Length_1) {
          iy = 1;
        }
      }
    }
  } else {
    xy_ends_POS_size_idx_0 = 0;
    Path_RES_0_size_idx_1 = 0;
  }

  count_1 = (real_T)(b_idx + 1) + target_k;
  if ((xy_ends_POS_size_idx_0 == 0) || (Path_RES_0_size_idx_1 == 0)) {
    if (count_1 <= jj) {
      if (b_idx + 1 > count_1) {
        b_idx = 0;
      }

      total_length_tmp = b_idx + (int32_T)target_k;
      ajj = rtDW.Path_RES_1_data[total_length_tmp - 1];
      count_1 = rtDW.Path_RES_1_data[(total_length_tmp + jj) - 1];
      rtb_Gain_p = target_k * 0.1;
    } else {
      if (b_idx + 1 > jj) {
        b_idx = 0;
        iy = 0;
      } else {
        iy = jj;
      }

      n = iy - b_idx;
      total_length_tmp = n + b_idx;
      ajj = rtDW.Path_RES_1_data[total_length_tmp - 1];
      count_1 = rtDW.Path_RES_1_data[(total_length_tmp + jj) - 1];
      if (n == 0) {
        n = 0;
      } else {
        if (!(n > 2)) {
          n = 2;
        }
      }

      rtb_Gain_p = (real_T)n * 0.1;
    }
  } else {
    if (b_idx + 1 > jj) {
      b_idx = 0;
      jmax = 0;
    } else {
      jmax = jj;
    }

    if (1.0 > rtb_Gain_p) {
      iy = 0;
    } else {
      iy = (int32_T)rtb_Gain_p;
    }

    loop_ub = jmax - b_idx;
    if (!(loop_ub == 0)) {
      n = 2;
      ix = loop_ub;
    } else {
      if (!(iy == 0)) {
        n = Path_RES_0_size_idx_1;
      } else {
        n = 2;
      }

      ix = 0;
    }

    if (!(iy == 0)) {
      jj = iy;
    } else {
      jj = 0;
    }

    for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
      rtDW.Path_RES_0_1[total_length_tmp] = rtDW.Path_RES_1_data[b_idx +
        total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
      rtDW.Path_RES_0_1[total_length_tmp + loop_ub] = rtDW.Path_RES_1_data
        [(b_idx + total_length_tmp) + Path_RES_1_size_idx_0];
    }

    loop_ub = Path_RES_0_size_idx_1 - 1;
    for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++)
    {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < iy;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_0_data_k[Path_RES_0_size_idx_1 + iy * total_length_tmp] =
          rtDW.Path_RES_0_data[xy_ends_POS_size_idx_0 * total_length_tmp +
          Path_RES_0_size_idx_1];
      }
    }

    jmax = ix + jj;
    for (total_length_tmp = 0; total_length_tmp < n; total_length_tmp++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < ix;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_data[Path_RES_0_size_idx_1 + jmax * total_length_tmp] =
          rtDW.Path_RES_0_1[ix * total_length_tmp + Path_RES_0_size_idx_1];
      }
    }

    for (total_length_tmp = 0; total_length_tmp < n; total_length_tmp++) {
      for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < jj;
           Path_RES_0_size_idx_1++) {
        rtDW.Path_RES_data[(Path_RES_0_size_idx_1 + ix) + jmax *
          total_length_tmp] = rtDW.Path_RES_0_data_k[jj * total_length_tmp +
          Path_RES_0_size_idx_1];
      }
    }

    if (target_k - Length_1 <= rtb_Gain_p) {
      ajj = rtDW.Path_RES_data[(int32_T)target_k - 1];
      count_1 = rtDW.Path_RES_data[((int32_T)target_k + jmax) - 1];
      rtb_Gain_p = target_k * 0.1;
    } else {
      rtb_Gain_p += Length_1;
      total_length_tmp = (int32_T)rtb_Gain_p;
      ajj = rtDW.Path_RES_data[total_length_tmp - 1];
      count_1 = rtDW.Path_RES_data[(total_length_tmp + jmax) - 1];
      rtb_Gain_p *= 0.1;
    }
  }

  // MATLAB Function: '<S2>/Forward_Seg1' incorporates:
  //   MATLAB Function: '<S2>/EndPointDecision'
  //   MATLAB Function: '<S2>/Forward_Seg'

  xy_ends_POS_size_idx_0 = rtDW.SFunction_DIMS4_h[0];
  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    rtDW.xy_ends_POS_data[total_length_tmp] =
      rtDW.Static_Path_0[total_length_tmp + rtDW.SFunction_DIMS4_h[0]];
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    rtDW.xy_ends_POS_data[total_length_tmp + xy_ends_POS_size_idx_0] =
      rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] << 1) + total_length_tmp];
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    rtDW.xy_ends_POS_data[total_length_tmp + (xy_ends_POS_size_idx_0 << 1)] =
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 3 + total_length_tmp];
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    rtDW.xy_ends_POS_data[total_length_tmp + xy_ends_POS_size_idx_0 * 3] =
      rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] << 2) + total_length_tmp];
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  if (0 <= loop_ub - 1) {
    memcpy(&dist_op_data[0], &rtDW.Static_Path_0[0], loop_ub * sizeof(real_T));
  }

  if (rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 3 +
                          rtDW.SFunction_DIMS4_h[0]) - 1] ==
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0]]) {
    n = (rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] << 2) +
          rtDW.SFunction_DIMS4_h[0]) - 1] ==
         rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] << 1]);
  } else {
    n = 0;
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    x_data[total_length_tmp] = (shortest_distance[rtDW.SFunction_DIMS4_f - 1] ==
      dist_op_data[total_length_tmp]);
  }

  b_idx = 1;
  b_x_0 = x_data[0];
  for (jmax = 2; jmax <= rtDW.SFunction_DIMS4_h[0]; jmax++) {
    if ((int32_T)b_x_0 < (int32_T)x_data[jmax - 1]) {
      b_x_0 = x_data[jmax - 1];
      b_idx = jmax;
    }
  }

  sigma = ajj - rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 3 + b_idx) - 1];
  b_a = count_1 - rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] << 2) + b_idx)
    - 1];
  total_length = std::sqrt(sigma * sigma + b_a * b_a);
  colj = b_idx;
  iy = 0;
  jmax = 0;
  ix = 0;
  exitg1 = false;
  while ((!exitg1) && (ix <= rtDW.SFunction_DIMS4_h[0] - 1)) {
    if (total_length > 15.0) {
      jmax = colj;
      exitg1 = true;
    } else {
      total_length_tmp = b_idx + ix;
      Path_RES_0_size_idx_1 = total_length_tmp + 1;
      if (Path_RES_0_size_idx_1 <= rtDW.SFunction_DIMS4_h[0]) {
        total_length += rtDW.Static_Path_0[total_length_tmp +
          (rtDW.SFunction_DIMS4_h[0] << 3)];
        colj = Path_RES_0_size_idx_1;
        iy = 1;
        ix++;
      } else if (n == 1) {
        total_length_tmp -= rtDW.SFunction_DIMS4_h[0];
        total_length += rtDW.Static_Path_0[total_length_tmp +
          (rtDW.SFunction_DIMS4_h[0] << 3)];
        colj = total_length_tmp + 1;
        iy = 2;
        ix++;
      } else {
        jmax = colj;
        iy = 3;
        exitg1 = true;
      }
    }
  }

  n = rtDW.SFunction_DIMS4_h[0] - 1;
  if (0 <= n) {
    memset(&Forward_Static_Path_id_0_data[0], 0, (n + 1) * sizeof(real_T));
  }

  if ((iy == 1) || (iy == 0)) {
    if (b_idx > jmax) {
      jj = 0;
      n = 0;
    } else {
      jj = b_idx - 1;
      n = jmax;
    }

    Path_RES_0_size_idx_1 = n - jj;
    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp] =
        rtDW.xy_ends_POS_data[jj + total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(jj + total_length_tmp) + xy_ends_POS_size_idx_0];
    }

    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + (Path_RES_0_size_idx_1 <<
        1)] = rtDW.xy_ends_POS_data[(jj + total_length_tmp) +
        (xy_ends_POS_size_idx_0 << 1)];
    }

    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + Path_RES_0_size_idx_1 *
        3] = rtDW.xy_ends_POS_data[(jj + total_length_tmp) +
        xy_ends_POS_size_idx_0 * 3];
    }

    if (b_idx > jmax) {
      iy = 1;
      total_length_tmp = 0;
    } else {
      iy = b_idx;
      total_length_tmp = jmax;
    }

    loop_ub = total_length_tmp - iy;
    for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++)
    {
      Forward_Static_Path_id_0_data[total_length_tmp] = dist_op_data[(iy +
        total_length_tmp) - 1];
    }

    if (b_idx > jmax) {
      b_idx = 1;
      jmax = 0;
    }

    b_idx = (jmax - b_idx) + 1;
  } else if (iy == 2) {
    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      n = 0;
      colj = 0;
    } else {
      n = b_idx - 1;
      colj = rtDW.SFunction_DIMS4_h[0];
    }

    if (1 > jmax) {
      loop_ub = 0;
    } else {
      loop_ub = jmax;
    }

    iy = colj - n;
    Path_RES_0_size_idx_1 = iy + loop_ub;
    for (total_length_tmp = 0; total_length_tmp < iy; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp] = rtDW.xy_ends_POS_data[n
        + total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp < iy; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(n + total_length_tmp) + xy_ends_POS_size_idx_0];
    }

    for (total_length_tmp = 0; total_length_tmp < iy; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + (Path_RES_0_size_idx_1 <<
        1)] = rtDW.xy_ends_POS_data[(n + total_length_tmp) +
        (xy_ends_POS_size_idx_0 << 1)];
    }

    for (total_length_tmp = 0; total_length_tmp < iy; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + Path_RES_0_size_idx_1 *
        3] = rtDW.xy_ends_POS_data[(n + total_length_tmp) +
        xy_ends_POS_size_idx_0 * 3];
    }

    for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[(total_length_tmp + colj) - n] =
        rtDW.xy_ends_POS_data[total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[((total_length_tmp + colj) - n) +
        Path_RES_0_size_idx_1] = rtDW.xy_ends_POS_data[total_length_tmp +
        xy_ends_POS_size_idx_0];
    }

    for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[((total_length_tmp + colj) - n) +
        (Path_RES_0_size_idx_1 << 1)] = rtDW.xy_ends_POS_data
        [(xy_ends_POS_size_idx_0 << 1) + total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[((total_length_tmp + colj) - n) +
        Path_RES_0_size_idx_1 * 3] =
        rtDW.xy_ends_POS_data[xy_ends_POS_size_idx_0 * 3 + total_length_tmp];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      iy = 0;
      n = 0;
    } else {
      iy = b_idx - 1;
      n = rtDW.SFunction_DIMS4_h[0];
    }

    ix = ((rtDW.SFunction_DIMS4_h[0] - b_idx) + jmax) + 1;
    if (1 > ix) {
      tmp_0 = 0;
    } else {
      tmp_0 = (int16_T)ix;
    }

    colj = tmp_0;
    loop_ub = tmp_0 - 1;
    for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++)
    {
      v_data[total_length_tmp] = (int16_T)total_length_tmp;
    }

    if (1 > jmax) {
      total_length_tmp = 0;
    } else {
      total_length_tmp = jmax;
    }

    loop_ub = total_length_tmp - 1;
    jj = n - iy;
    for (total_length_tmp = 0; total_length_tmp < jj; total_length_tmp++) {
      table[total_length_tmp] = dist_op_data[iy + total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++)
    {
      table[(total_length_tmp + n) - iy] = dist_op_data[total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp < colj; total_length_tmp++) {
      Forward_Static_Path_id_0_data[v_data[total_length_tmp]] =
        table[total_length_tmp];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_idx = 1;
      ix = 1;
    } else {
      ix = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    if (1 > jmax) {
      jmax = 0;
    }

    b_idx = (ix - b_idx) + jmax;
  } else {
    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      n = 0;
      iy = 0;
    } else {
      n = b_idx - 1;
      iy = rtDW.SFunction_DIMS4_h[0];
    }

    Path_RES_0_size_idx_1 = iy - n;
    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp] = rtDW.xy_ends_POS_data[n
        + total_length_tmp];
    }

    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + Path_RES_0_size_idx_1] =
        rtDW.xy_ends_POS_data[(n + total_length_tmp) + xy_ends_POS_size_idx_0];
    }

    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + (Path_RES_0_size_idx_1 <<
        1)] = rtDW.xy_ends_POS_data[(n + total_length_tmp) +
        (xy_ends_POS_size_idx_0 << 1)];
    }

    for (total_length_tmp = 0; total_length_tmp < Path_RES_0_size_idx_1;
         total_length_tmp++) {
      rtDW.Static_Path_ends_POS_data[total_length_tmp + Path_RES_0_size_idx_1 *
        3] = rtDW.xy_ends_POS_data[(n + total_length_tmp) +
        xy_ends_POS_size_idx_0 * 3];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      ix = 1;
      n = 0;
    } else {
      ix = b_idx;
      n = rtDW.SFunction_DIMS4_h[0];
    }

    loop_ub = n - ix;
    for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++)
    {
      Forward_Static_Path_id_0_data[total_length_tmp] = dist_op_data[(ix +
        total_length_tmp) - 1];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_idx = 1;
      jmax = 1;
    } else {
      jmax = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    b_idx = jmax - b_idx;
  }

  if (1 > b_idx) {
    b_idx = 0;
  }

  jmax = Path_RES_0_size_idx_1 + 1;
  loop_ub = (jmax << 1) - 1;
  if (0 <= loop_ub) {
    memset(&rtDW.Forward_Static_Path_data_m[0], 0, (loop_ub + 1) * sizeof(real_T));
  }

  loop_ub = Path_RES_0_size_idx_1 - 1;
  if (0 <= loop_ub) {
    memcpy(&rtDW.Forward_Static_Path_data_m[0], &rtDW.Static_Path_ends_POS_data
           [0], (loop_ub + 1) * sizeof(real_T));
  }

  for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++) {
    rtDW.Forward_Static_Path_data_m[total_length_tmp + jmax] =
      rtDW.Static_Path_ends_POS_data[total_length_tmp + Path_RES_0_size_idx_1];
  }

  total_length_tmp = Path_RES_0_size_idx_1 - 1;
  rtDW.Forward_Static_Path_data_m[Path_RES_0_size_idx_1] =
    rtDW.Static_Path_ends_POS_data[(Path_RES_0_size_idx_1 << 1) +
    total_length_tmp];
  rtDW.Forward_Static_Path_data_m[Path_RES_0_size_idx_1 + jmax] =
    rtDW.Static_Path_ends_POS_data[Path_RES_0_size_idx_1 * 3 + total_length_tmp];
  rtDW.SFunction_DIMS2 = jmax;
  loop_ub = jmax - 1;
  if (0 <= loop_ub) {
    memcpy(&rtb_Forward_Static_Path_x_h[0], &rtDW.Forward_Static_Path_data_m[0],
           (loop_ub + 1) * sizeof(real_T));
  }

  rtDW.SFunction_DIMS3 = jmax;
  loop_ub = jmax - 1;
  for (total_length_tmp = 0; total_length_tmp <= loop_ub; total_length_tmp++) {
    rtb_Forward_Static_Path_y_p[total_length_tmp] =
      rtDW.Forward_Static_Path_data_m[total_length_tmp + jmax];
  }

  rtDW.SFunction_DIMS4 = b_idx;
  if (0 <= b_idx - 1) {
    memcpy(&rtb_Forward_Static_Path_id[0], &Forward_Static_Path_id_0_data[0],
           b_idx * sizeof(real_T));
  }

  // End of MATLAB Function: '<S2>/Forward_Seg1'

  // MATLAB Function: '<S2>/DangerousArea' incorporates:
  //   MATLAB Function: '<S2>/EndPointDecision'
  //   UnitDelay: '<S2>/Unit Delay10'
  //   UnitDelay: '<S2>/Unit Delay12'
  //   UnitDelay: '<S2>/Unit Delay8'
  //   UnitDelay: '<S2>/Unit Delay9'

  c = rtDW.UnitDelay8_DSTATE;
  rtb_num_lane_direction_f[0] = rtDW.UnitDelay9_DSTATE[0];
  rtb_H_y_out[0] = rtDW.UnitDelay10_DSTATE[0];
  rtb_num_lane_direction_f[1] = rtDW.UnitDelay9_DSTATE[1];
  rtb_H_y_out[1] = rtDW.UnitDelay10_DSTATE[1];
  rtb_num_lane_direction_f[2] = rtDW.UnitDelay9_DSTATE[2];
  rtb_H_y_out[2] = rtDW.UnitDelay10_DSTATE[2];
  rtb_num_lane_direction_f[3] = rtDW.UnitDelay9_DSTATE[3];
  rtb_H_y_out[3] = rtDW.UnitDelay10_DSTATE[3];
  colj = 0;
  x_0 = rtb_X[0];
  total_length = rtb_X[1];
  loop_ub = rtDW.SFunction_DIMS4_f * rtDW.SFunction_DIMS4_h[1] - 1;
  if (0 <= loop_ub) {
    memset(&rtDW.Forward_Static_Path_0_data[0], 0, (loop_ub + 1) * sizeof(real_T));
  }

  for (ix = 0; ix < rtDW.SFunction_DIMS4_f; ix++) {
    loop_ub = rtDW.SFunction_DIMS4_h[0];
    for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
      x_data[total_length_tmp] = (shortest_distance[ix] ==
        rtDW.Static_Path_0[total_length_tmp]);
    }

    b_idx = 0;
    b_x_0 = x_data[0];
    for (jmax = 1; jmax < rtDW.SFunction_DIMS4_h[0]; jmax++) {
      if ((int32_T)b_x_0 < (int32_T)x_data[jmax]) {
        b_x_0 = x_data[jmax];
        b_idx = jmax;
      }
    }

    loop_ub = rtDW.SFunction_DIMS4_h[1];
    for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
      rtDW.Forward_Static_Path_0_data[ix + rtDW.SFunction_DIMS4_f *
        total_length_tmp] = rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] *
        total_length_tmp + b_idx];
    }
  }

  iy = 0;
  exitg1 = false;
  while ((!exitg1) && (iy <= i)) {
    sigma = (1.0 + (real_T)iy) * 2.0;
    for (total_length_tmp = 0; total_length_tmp < 4; total_length_tmp++) {
      OBXY_m[total_length_tmp << 1] = rtb_V_boundingbox[((int32_T)(sigma + -1.0)
        + 100 * total_length_tmp) - 1];
      OBXY_m[1 + (total_length_tmp << 1)] = rtb_V_boundingbox[(100 *
        total_length_tmp + (int32_T)sigma) - 1];
    }

    n = 0;
    exitg3 = false;
    while ((!exitg3) && (n <= rtDW.SFunction_DIMS4_f - 1)) {
      sigma = rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f << 2) + n]
        - rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f << 1) + n];
      Length_1 = rtDW.Forward_Static_Path_0_data[n + rtDW.SFunction_DIMS4_f] -
        rtDW.Forward_Static_Path_0_data[rtDW.SFunction_DIMS4_f * 3 + n];
      c = (rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f << 2) + n] -
           rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f << 1) + n]) *
        -rtDW.Forward_Static_Path_0_data[n + rtDW.SFunction_DIMS4_f] +
        (rtDW.Forward_Static_Path_0_data[rtDW.SFunction_DIMS4_f * 3 + n] -
         rtDW.Forward_Static_Path_0_data[n + rtDW.SFunction_DIMS4_f]) *
        rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f << 1) + n];
      yy_idx_0 = Length_1 * Length_1;
      b_a = std::sqrt(sigma * sigma + yy_idx_0);
      sigma_0[0] = (sigma * OBXY_m[0] + Length_1 * OBXY_m[1]) + c;
      sigma_0[1] = (sigma * OBXY_m[2] + Length_1 * OBXY_m[3]) + c;
      sigma_0[2] = (sigma * OBXY_m[4] + Length_1 * OBXY_m[5]) + c;
      sigma_0[3] = (sigma * OBXY_m[6] + Length_1 * OBXY_m[7]) + c;
      abs_g(sigma_0, rtb_num_lane_direction_b);
      K1[0] = rtb_num_lane_direction_b[0] / b_a;
      K1[1] = rtb_num_lane_direction_b[1] / b_a;
      K1[2] = rtb_num_lane_direction_b[2] / b_a;
      K1[3] = rtb_num_lane_direction_b[3] / b_a;
      b_a = sigma * Length_1;
      x_endpoint2 = sigma * sigma + yy_idx_0;
      target_k = sigma * c;
      rtb_num_lane_direction_f[0] = ((yy_idx_0 * OBXY_m[0] - b_a * OBXY_m[1]) -
        target_k) / x_endpoint2;
      rtb_num_lane_direction_f[1] = ((yy_idx_0 * OBXY_m[2] - b_a * OBXY_m[3]) -
        target_k) / x_endpoint2;
      rtb_num_lane_direction_f[2] = ((yy_idx_0 * OBXY_m[4] - b_a * OBXY_m[5]) -
        target_k) / x_endpoint2;
      rtb_num_lane_direction_f[3] = ((yy_idx_0 * OBXY_m[6] - b_a * OBXY_m[7]) -
        target_k) / x_endpoint2;
      b_a = -sigma * Length_1;
      target_k = sigma * sigma;
      x_endpoint2 = sigma * sigma + yy_idx_0;
      delta_offset = Length_1 * c;
      rtb_H_y_out[0] = ((b_a * OBXY_m[0] + target_k * OBXY_m[1]) - delta_offset)
        / x_endpoint2;
      rtb_H_y_out[1] = ((b_a * OBXY_m[2] + target_k * OBXY_m[3]) - delta_offset)
        / x_endpoint2;
      rtb_H_y_out[2] = ((b_a * OBXY_m[4] + target_k * OBXY_m[5]) - delta_offset)
        / x_endpoint2;
      rtb_H_y_out[3] = ((b_a * OBXY_m[6] + target_k * OBXY_m[7]) - delta_offset)
        / x_endpoint2;
      rtb_Oi_near_l[0] = ((yy_idx_0 * x_0 - sigma * Length_1 * total_length) -
                          sigma * c) / (sigma * sigma + yy_idx_0);
      yy_idx_0 = ((-sigma * Length_1 * x_0 + sigma * sigma * total_length) -
                  Length_1 * c) / (sigma * sigma + yy_idx_0);
      b_x_0 = rtIsNaN(rtb_num_lane_direction_f[0]);
      if (!b_x_0) {
        b_idx = 1;
      } else {
        b_idx = 0;
        jmax = 2;
        exitg4 = false;
        while ((!exitg4) && (jmax < 5)) {
          if (!rtIsNaN(rtb_num_lane_direction_f[jmax - 1])) {
            b_idx = jmax;
            exitg4 = true;
          } else {
            jmax++;
          }
        }
      }

      if (b_idx == 0) {
        sigma = rtb_num_lane_direction_f[0];
      } else {
        sigma = rtb_num_lane_direction_f[b_idx - 1];
        while (b_idx + 1 < 5) {
          if (sigma > rtb_num_lane_direction_f[b_idx]) {
            sigma = rtb_num_lane_direction_f[b_idx];
          }

          b_idx++;
        }
      }

      if (rtb_Oi_near_l[0] < ajj) {
        c = ajj;
      } else if (rtIsNaN(rtb_Oi_near_l[0])) {
        if (!rtIsNaN(ajj)) {
          c = ajj;
        } else {
          c = rtb_Oi_near_l[0];
        }
      } else {
        c = rtb_Oi_near_l[0];
      }

      guard1 = false;
      if (sigma <= c) {
        if (!b_x_0) {
          b_idx = 1;
        } else {
          b_idx = 0;
          jmax = 2;
          exitg4 = false;
          while ((!exitg4) && (jmax < 5)) {
            if (!rtIsNaN(rtb_num_lane_direction_f[jmax - 1])) {
              b_idx = jmax;
              exitg4 = true;
            } else {
              jmax++;
            }
          }
        }

        if (b_idx == 0) {
          sigma = rtb_num_lane_direction_f[0];
        } else {
          sigma = rtb_num_lane_direction_f[b_idx - 1];
          while (b_idx + 1 < 5) {
            if (sigma < rtb_num_lane_direction_f[b_idx]) {
              sigma = rtb_num_lane_direction_f[b_idx];
            }

            b_idx++;
          }
        }

        if (rtb_Oi_near_l[0] > ajj) {
          c = ajj;
        } else if (rtIsNaN(rtb_Oi_near_l[0])) {
          if (!rtIsNaN(ajj)) {
            c = ajj;
          } else {
            c = rtb_Oi_near_l[0];
          }
        } else {
          c = rtb_Oi_near_l[0];
        }

        if (sigma >= c) {
          b_x_0 = rtIsNaN(rtb_H_y_out[0]);
          if (!b_x_0) {
            b_idx = 1;
          } else {
            b_idx = 0;
            jj = 2;
            exitg4 = false;
            while ((!exitg4) && (jj < 5)) {
              if (!rtIsNaN(rtb_H_y_out[jj - 1])) {
                b_idx = jj;
                exitg4 = true;
              } else {
                jj++;
              }
            }
          }

          if (b_idx == 0) {
            sigma = rtb_H_y_out[0];
          } else {
            sigma = rtb_H_y_out[b_idx - 1];
            while (b_idx + 1 < 5) {
              if (sigma > rtb_H_y_out[b_idx]) {
                sigma = rtb_H_y_out[b_idx];
              }

              b_idx++;
            }
          }

          if (yy_idx_0 < count_1) {
            c = count_1;
          } else if (rtIsNaN(yy_idx_0)) {
            if (!rtIsNaN(count_1)) {
              c = count_1;
            } else {
              c = yy_idx_0;
            }
          } else {
            c = yy_idx_0;
          }

          if (sigma <= c) {
            if (!b_x_0) {
              b_idx = 1;
            } else {
              b_idx = 0;
              jj = 2;
              exitg4 = false;
              while ((!exitg4) && (jj < 5)) {
                if (!rtIsNaN(rtb_H_y_out[jj - 1])) {
                  b_idx = jj;
                  exitg4 = true;
                } else {
                  jj++;
                }
              }
            }

            if (b_idx == 0) {
              sigma = rtb_H_y_out[0];
            } else {
              sigma = rtb_H_y_out[b_idx - 1];
              while (b_idx + 1 < 5) {
                if (sigma < rtb_H_y_out[b_idx]) {
                  sigma = rtb_H_y_out[b_idx];
                }

                b_idx++;
              }
            }

            if (yy_idx_0 > count_1) {
              yy_idx_0 = count_1;
            } else {
              if (rtIsNaN(yy_idx_0) && (!rtIsNaN(count_1))) {
                yy_idx_0 = count_1;
              }
            }

            if (sigma >= yy_idx_0) {
              if (!rtIsNaN(K1[0])) {
                b_idx = 1;
              } else {
                b_idx = 0;
                jmax = 2;
                exitg4 = false;
                while ((!exitg4) && (jmax < 5)) {
                  if (!rtIsNaN(K1[jmax - 1])) {
                    b_idx = jmax;
                    exitg4 = true;
                  } else {
                    jmax++;
                  }
                }
              }

              if (b_idx == 0) {
                sigma = K1[0];
              } else {
                sigma = K1[b_idx - 1];
                while (b_idx + 1 < 5) {
                  if (sigma > K1[b_idx]) {
                    sigma = K1[b_idx];
                  }

                  b_idx++;
                }
              }

              if (sigma <=
                  rtDW.Forward_Static_Path_0_data[rtDW.SFunction_DIMS4_f * 10 +
                  n] / 2.0) {
                c = 1.0;
                colj = 1;
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
        c = rtDW.UnitDelay8_DSTATE;
        rtb_num_lane_direction_f[0] = rtDW.UnitDelay9_DSTATE[0];
        rtb_H_y_out[0] = rtDW.UnitDelay10_DSTATE[0];
        rtb_num_lane_direction_f[1] = rtDW.UnitDelay9_DSTATE[1];
        rtb_H_y_out[1] = rtDW.UnitDelay10_DSTATE[1];
        rtb_num_lane_direction_f[2] = rtDW.UnitDelay9_DSTATE[2];
        rtb_H_y_out[2] = rtDW.UnitDelay10_DSTATE[2];
        rtb_num_lane_direction_f[3] = rtDW.UnitDelay9_DSTATE[3];
        rtb_H_y_out[3] = rtDW.UnitDelay10_DSTATE[3];
        n++;
      }
    }

    if (colj == 1) {
      exitg1 = true;
    } else {
      iy++;
    }
  }

  if (c == 1.0) {
    sigma_0[0] = ajj - rtb_num_lane_direction_f[0];
    sigma_0[1] = ajj - rtb_num_lane_direction_f[1];
    sigma_0[2] = ajj - rtb_num_lane_direction_f[2];
    sigma_0[3] = ajj - rtb_num_lane_direction_f[3];
    power_j(sigma_0, rtb_num_lane_direction_b);
    sigma_0[0] = count_1 - rtb_H_y_out[0];
    sigma_0[1] = count_1 - rtb_H_y_out[1];
    sigma_0[2] = count_1 - rtb_H_y_out[2];
    sigma_0[3] = count_1 - rtb_H_y_out[3];
    power_j(sigma_0, tmp);
    K1[0] = rtb_num_lane_direction_b[0] + tmp[0];
    K1[1] = rtb_num_lane_direction_b[1] + tmp[1];
    K1[2] = rtb_num_lane_direction_b[2] + tmp[2];
    K1[3] = rtb_num_lane_direction_b[3] + tmp[3];
    if (!rtIsNaN(K1[0])) {
      b_idx = 1;
    } else {
      b_idx = 0;
      jmax = 2;
      exitg1 = false;
      while ((!exitg1) && (jmax < 5)) {
        if (!rtIsNaN(K1[jmax - 1])) {
          b_idx = jmax;
          exitg1 = true;
        } else {
          jmax++;
        }
      }
    }

    if (b_idx == 0) {
      yy_idx_0 = K1[0];
    } else {
      yy_idx_0 = K1[b_idx - 1];
      while (b_idx + 1 < 5) {
        if (yy_idx_0 > K1[b_idx]) {
          yy_idx_0 = K1[b_idx];
        }

        b_idx++;
      }
    }

    if (std::sqrt(yy_idx_0) > diffheading + 3.0) {
      c = 0.0;
    }
  }

  if (c == 1.0) {
    diffheading = 100.0;

    // MATLAB Function: '<S2>/DynamicPathPlanning'
    total_length = 5.0;
  } else {
    diffheading = rtDW.UnitDelay12_DSTATE - 1.0;
    if (rtDW.UnitDelay12_DSTATE - 1.0 < 0.0) {
      diffheading = 0.0;
    }

    // MATLAB Function: '<S2>/DynamicPathPlanning' incorporates:
    //   UnitDelay: '<S2>/Unit Delay12'

    total_length = (100.0 - diffheading) * 5.0 / 100.0 + 5.0;
  }

  // SignalConversion: '<S11>/TmpSignal ConversionAt SFunction Inport6' incorporates:
  //   Gain: '<S2>/Gain1'
  //   MATLAB Function: '<S2>/DynamicPathPlanning'

  x_0 = 0.017453292519943295 * rtb_Gain1;

  // MATLAB Function: '<S2>/DynamicPathPlanning' incorporates:
  //   Constant: '<S2>/Constant16'
  //   Inport: '<Root>/Path_flag'
  //   MATLAB Function: '<S2>/EndPointDecision'
  //   MATLAB Function: '<S2>/MM'
  //   SignalConversion: '<S11>/TmpSignal ConversionAt SFunction Inport6'
  //   UnitDelay: '<S2>/Unit Delay5'

  if (!rtDW.LastPath_not_empty) {
    memset(&LastPath[0], 0, 22U * sizeof(real_T));
    rtDW.LastPath_not_empty = true;
  } else {
    memcpy(&LastPath[0], &rtDW.UnitDelay5_DSTATE[0], 22U * sizeof(real_T));
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    x_data[total_length_tmp] = (shortest_distance[rtDW.SFunction_DIMS4_f - 1] ==
      rtDW.Static_Path_0[total_length_tmp]);
  }

  jmax = rtDW.SFunction_DIMS4_h[0] - 1;
  iy = 0;
  for (n = 0; n <= jmax; n++) {
    if (x_data[n]) {
      iy++;
    }
  }

  ix = 0;
  for (n = 0; n <= jmax; n++) {
    if (x_data[n]) {
      p_data[ix] = n + 1;
      ix++;
    }
  }

  for (total_length_tmp = 0; total_length_tmp < iy; total_length_tmp++) {
    dist_op_data[total_length_tmp] = rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h
      [0] * 7 + p_data[total_length_tmp]) - 1] * 3.1415926535897931;
  }

  for (total_length_tmp = 0; total_length_tmp < iy; total_length_tmp++) {
    Forward_Static_Path_id_0_data[total_length_tmp] =
      dist_op_data[total_length_tmp] / 180.0;
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    x_data[total_length_tmp] = (shortest_distance[rtDW.SFunction_DIMS4_f - 1] ==
      rtDW.Static_Path_0[total_length_tmp]);
  }

  n = 0;
  for (b_idx = 0; b_idx < rtDW.SFunction_DIMS4_h[0]; b_idx++) {
    if (x_data[b_idx]) {
      q_data[n] = b_idx + 1;
      n++;
    }
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (total_length_tmp = 0; total_length_tmp < loop_ub; total_length_tmp++) {
    x_data[total_length_tmp] = (shortest_distance[rtDW.SFunction_DIMS4_f - 1] ==
      rtDW.Static_Path_0[total_length_tmp]);
  }

  ix = 0;
  for (b_idx = 0; b_idx < rtDW.SFunction_DIMS4_h[0]; b_idx++) {
    if (x_data[b_idx]) {
      s_data[ix] = b_idx + 1;
      ix++;
    }
  }

  delta_offset = rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 10 + s_data[0])
    - 1] / 4.0;
  offset_2 = delta_offset * 2.0;
  offset_3 = delta_offset * 3.0;
  offset_4 = delta_offset * 4.0;
  offset_5 = delta_offset * 5.0;
  offset_6 = delta_offset * 6.0;
  offset[0] = offset_6;
  offset[1] = offset_5;
  offset[2] = offset_4;
  offset[3] = offset_3;
  offset[4] = offset_2;
  offset[5] = delta_offset;
  offset[6] = 0.0;
  offset[7] = delta_offset;
  offset[8] = offset_2;
  offset[9] = offset_3;
  offset[10] = offset_4;
  offset[11] = offset_5;
  offset[12] = offset_6;
  x_endpoint6 = std::cos(Forward_Static_Path_id_0_data[0] + 1.5707963267948966);
  Length_1 = x_endpoint6 * offset_6 + ajj;
  y_endpoint6 = std::sin(Forward_Static_Path_id_0_data[0] + 1.5707963267948966);
  target_k = y_endpoint6 * offset_6 + count_1;
  x_endpoint2 = x_endpoint6 * offset_5 + ajj;
  y_endpoint2 = y_endpoint6 * offset_5 + count_1;
  x_endpoint3 = x_endpoint6 * offset_4 + ajj;
  y_endpoint3 = y_endpoint6 * offset_4 + count_1;
  x_endpoint4 = x_endpoint6 * offset_3 + ajj;
  y_endpoint4 = y_endpoint6 * offset_3 + count_1;
  x_endpoint5 = x_endpoint6 * offset_2 + ajj;
  y_endpoint5 = y_endpoint6 * offset_2 + count_1;
  x_endpoint6 = x_endpoint6 * delta_offset + ajj;
  y_endpoint6 = y_endpoint6 * delta_offset + count_1;
  x_endpoint13 = std::cos(Forward_Static_Path_id_0_data[0] - 1.5707963267948966);
  x_endpoint8 = x_endpoint13 * delta_offset + ajj;
  sigma = std::sin(Forward_Static_Path_id_0_data[0] - 1.5707963267948966);
  delta_offset = sigma * delta_offset + count_1;
  x_endpoint9 = x_endpoint13 * offset_2 + ajj;
  offset_2 = sigma * offset_2 + count_1;
  x_endpoint10 = x_endpoint13 * offset_3 + ajj;
  offset_3 = sigma * offset_3 + count_1;
  x_endpoint11 = x_endpoint13 * offset_4 + ajj;
  offset_4 = sigma * offset_4 + count_1;
  x_endpoint12 = x_endpoint13 * offset_5 + ajj;
  offset_5 = sigma * offset_5 + count_1;
  x_endpoint13 = x_endpoint13 * offset_6 + ajj;
  offset_6 = sigma * offset_6 + count_1;
  G2splines(rtb_X[0], rtb_X[1], x_0, rtDW.Static_Path_0[idx +
            rtDW.SFunction_DIMS4_h[0] * 13], Length_1, target_k,
            Forward_Static_Path_id_0_data[0], rtDW.Static_Path_0[(q_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Gain_p, x,
            b_Path_dis_data, rtb_YP_final_o, YP1, XY_difflen_0, X_NV, &L_path[0]);
  G2splines(rtb_X[0], rtb_X[1], x_0, rtDW.Static_Path_0[idx +
            rtDW.SFunction_DIMS4_h[0] * 13], x_endpoint2, y_endpoint2,
            Forward_Static_Path_id_0_data[0], rtDW.Static_Path_0[(q_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Gain_p, X2, Y2, XP2, YP2,
            K2, K_12, &L_path[1]);
  G2splines(rtb_X[0], rtb_X[1], x_0, rtDW.Static_Path_0[idx +
            rtDW.SFunction_DIMS4_h[0] * 13], x_endpoint3, y_endpoint3,
            Forward_Static_Path_id_0_data[0], rtDW.Static_Path_0[(q_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Gain_p, X3, Y3, XP3, YP3,
            K3, K_13, &L_path[2]);
  G2splines(rtb_X[0], rtb_X[1], x_0, rtDW.Static_Path_0[idx +
            rtDW.SFunction_DIMS4_h[0] * 13], x_endpoint4, y_endpoint4,
            Forward_Static_Path_id_0_data[0], rtDW.Static_Path_0[(q_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Gain_p, X4, Y4, XP4, YP4,
            K4, K_14, &L_path[3]);
  G2splines(rtb_X[0], rtb_X[1], x_0, rtDW.Static_Path_0[idx +
            rtDW.SFunction_DIMS4_h[0] * 13], x_endpoint5, y_endpoint5,
            Forward_Static_Path_id_0_data[0], rtDW.Static_Path_0[(q_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Gain_p, X5, Y5, XP5, YP5,
            K5, K_15, &L_path[4]);
  G2splines(rtb_X[0], rtb_X[1], x_0, rtDW.Static_Path_0[idx +
            rtDW.SFunction_DIMS4_h[0] * 13], x_endpoint6, y_endpoint6,
            Forward_Static_Path_id_0_data[0], rtDW.Static_Path_0[(q_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Gain_p, X6, Y6, XP6, YP6,
            K6, K_16, &L_path[5]);
  G2splines(rtb_X[0], rtb_X[1], x_0, rtDW.Static_Path_0[idx +
            rtDW.SFunction_DIMS4_h[0] * 13], ajj, count_1,
            Forward_Static_Path_id_0_data[0], rtDW.Static_Path_0[(q_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Gain_p, X7, Y7, XP7, YP7,
            K7, K_17, &L_path[6]);
  G2splines(rtb_X[0], rtb_X[1], x_0, rtDW.Static_Path_0[idx +
            rtDW.SFunction_DIMS4_h[0] * 13], x_endpoint8, delta_offset,
            Forward_Static_Path_id_0_data[0], rtDW.Static_Path_0[(q_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Gain_p, X8, Y8, XP8, YP8,
            K8, K_18, &L_path[7]);
  G2splines(rtb_X[0], rtb_X[1], x_0, rtDW.Static_Path_0[idx +
            rtDW.SFunction_DIMS4_h[0] * 13], x_endpoint9, offset_2,
            Forward_Static_Path_id_0_data[0], rtDW.Static_Path_0[(q_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Gain_p, X9, Y9, XP9, YP9,
            K9, K_19, &L_path[8]);
  G2splines(rtb_X[0], rtb_X[1], x_0, rtDW.Static_Path_0[idx +
            rtDW.SFunction_DIMS4_h[0] * 13], x_endpoint10, offset_3,
            Forward_Static_Path_id_0_data[0], rtDW.Static_Path_0[(q_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Gain_p, X10, Y10, XP10,
            YP10, K10, K_110, &L_path[9]);
  G2splines(rtb_X[0], rtb_X[1], x_0, rtDW.Static_Path_0[idx +
            rtDW.SFunction_DIMS4_h[0] * 13], x_endpoint11, offset_4,
            Forward_Static_Path_id_0_data[0], rtDW.Static_Path_0[(q_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Gain_p, X11, Y11, XP11,
            YP11, K11, K_111, &L_path[10]);
  G2splines(rtb_X[0], rtb_X[1], x_0, rtDW.Static_Path_0[idx +
            rtDW.SFunction_DIMS4_h[0] * 13], x_endpoint12, offset_5,
            Forward_Static_Path_id_0_data[0], rtDW.Static_Path_0[(q_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Gain_p, X12, Y12, XP12,
            YP12, K12, K_112, &L_path[11]);
  G2splines(rtb_X[0], rtb_X[1], x_0, rtDW.Static_Path_0[idx +
            rtDW.SFunction_DIMS4_h[0] * 13], x_endpoint13, offset_6,
            Forward_Static_Path_id_0_data[0], rtDW.Static_Path_0[(q_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], rtb_Gain_p, X13, Y13, XP13,
            YP13, K13, K_113, &L_path[12]);
  for (total_length_tmp = 0; total_length_tmp < 11; total_length_tmp++) {
    X_2[total_length_tmp] = x[total_length_tmp];
    X_2[total_length_tmp + 11] = X2[total_length_tmp];
    X_2[total_length_tmp + 22] = X3[total_length_tmp];
    X_2[total_length_tmp + 33] = X4[total_length_tmp];
    X_2[total_length_tmp + 44] = X5[total_length_tmp];
    X_2[total_length_tmp + 55] = X6[total_length_tmp];
    X_2[total_length_tmp + 66] = X7[total_length_tmp];
    X_2[total_length_tmp + 77] = X8[total_length_tmp];
    X_2[total_length_tmp + 88] = X9[total_length_tmp];
    X_2[total_length_tmp + 99] = X10[total_length_tmp];
    X_2[total_length_tmp + 110] = X11[total_length_tmp];
    X_2[total_length_tmp + 121] = X12[total_length_tmp];
    X_2[total_length_tmp + 132] = X13[total_length_tmp];
    Y[total_length_tmp] = b_Path_dis_data[total_length_tmp];
    Y[total_length_tmp + 11] = Y2[total_length_tmp];
    Y[total_length_tmp + 22] = Y3[total_length_tmp];
    Y[total_length_tmp + 33] = Y4[total_length_tmp];
    Y[total_length_tmp + 44] = Y5[total_length_tmp];
    Y[total_length_tmp + 55] = Y6[total_length_tmp];
    Y[total_length_tmp + 66] = Y7[total_length_tmp];
    Y[total_length_tmp + 77] = Y8[total_length_tmp];
    Y[total_length_tmp + 88] = Y9[total_length_tmp];
    Y[total_length_tmp + 99] = Y10[total_length_tmp];
    Y[total_length_tmp + 110] = Y11[total_length_tmp];
    Y[total_length_tmp + 121] = Y12[total_length_tmp];
    Y[total_length_tmp + 132] = Y13[total_length_tmp];
    K[total_length_tmp] = XY_difflen_0[total_length_tmp];
    K[total_length_tmp + 11] = K2[total_length_tmp];
    K[total_length_tmp + 22] = K3[total_length_tmp];
    K[total_length_tmp + 33] = K4[total_length_tmp];
    K[total_length_tmp + 44] = K5[total_length_tmp];
    K[total_length_tmp + 55] = K6[total_length_tmp];
    K[total_length_tmp + 66] = K7[total_length_tmp];
    K[total_length_tmp + 77] = K8[total_length_tmp];
    K[total_length_tmp + 88] = K9[total_length_tmp];
    K[total_length_tmp + 99] = K10[total_length_tmp];
    K[total_length_tmp + 110] = K11[total_length_tmp];
    K[total_length_tmp + 121] = K12[total_length_tmp];
    K[total_length_tmp + 132] = K13[total_length_tmp];
    K_1[total_length_tmp] = X_NV[total_length_tmp];
    K_1[total_length_tmp + 11] = K_12[total_length_tmp];
    K_1[total_length_tmp + 22] = K_13[total_length_tmp];
    K_1[total_length_tmp + 33] = K_14[total_length_tmp];
    K_1[total_length_tmp + 44] = K_15[total_length_tmp];
    K_1[total_length_tmp + 55] = K_16[total_length_tmp];
    K_1[total_length_tmp + 66] = K_17[total_length_tmp];
    K_1[total_length_tmp + 77] = K_18[total_length_tmp];
    K_1[total_length_tmp + 88] = K_19[total_length_tmp];
    K_1[total_length_tmp + 99] = K_110[total_length_tmp];
    K_1[total_length_tmp + 110] = K_111[total_length_tmp];
    K_1[total_length_tmp + 121] = K_112[total_length_tmp];
    K_1[total_length_tmp + 132] = K_113[total_length_tmp];
  }

  memset(&Path_col[0], 0, 52U * sizeof(real_T));
  for (total_length_tmp = 0; total_length_tmp < 5; total_length_tmp++) {
    Path_col[3 + ((8 + total_length_tmp) << 2)] = 1.0;
  }

  memcpy(&OBXY_EL[0], &rtb_V_boundingbox[0], 400U * sizeof(real_T));
  for (iy = 0; iy <= i; iy++) {
    sigma = (1.0 + (real_T)iy) * 2.0;
    b_idx = (int32_T)(sigma + -1.0);
    jmax = b_idx - 1;
    OBXY_EL[jmax] = ((rtb_V_boundingbox[jmax] - rtb_V_boundingbox[b_idx + 99]) *
                     0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) * 2.0
      + -1.0) - 1]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) * 2.0 +
      -1.0) - 1] - rtb_V_boundingbox[b_idx + 299]) * 0.3;
    jmax = (int32_T)sigma;
    idx = jmax - 1;
    OBXY_EL[idx] = ((rtb_V_boundingbox[idx] - rtb_V_boundingbox[jmax + 99]) *
                    0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) * 2.0)
                    - 1]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) *
      2.0) - 1] - rtb_V_boundingbox[jmax + 299]) * 0.3;
    OBXY_EL[(int32_T)(sigma + -1.0) + 99] = ((rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)iy) * 2.0 + -1.0) + 99] - rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)iy) * 2.0 + -1.0) - 1]) * 0.3 + rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)iy) * 2.0 + -1.0) + 99]) + (rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)iy) * 2.0 + -1.0) + 99] - rtb_V_boundingbox[b_idx + 199]) * 0.3;
    OBXY_EL[(int32_T)sigma + 99] = ((rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
      iy) * 2.0) + 99] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) * 2.0) -
      1]) * 0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) * 2.0) + 99]) +
      (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) * 2.0) + 99] -
       rtb_V_boundingbox[jmax + 199]) * 0.3;
    OBXY_EL[(int32_T)(sigma + -1.0) + 199] = ((rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)iy) * 2.0 + -1.0) + 199] - rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)iy) * 2.0 + -1.0) + 299]) * 0.3 + rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)iy) * 2.0 + -1.0) + 199]) + (rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)iy) * 2.0 + -1.0) + 199] - rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)iy) * 2.0 + -1.0) + 99]) * 0.3;
    OBXY_EL[(int32_T)sigma + 199] = ((rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
      iy) * 2.0) + 199] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) * 2.0)
      + 299]) * 0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) * 2.0) +
      199]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) * 2.0) + 199] -
               rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) * 2.0) + 99]) *
      0.3;
    OBXY_EL[(int32_T)(sigma + -1.0) + 299] = ((rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)iy) * 2.0 + -1.0) + 299] - rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)iy) * 2.0 + -1.0) + 199]) * 0.3 + rtb_V_boundingbox[(int32_T)((1.0
      + (real_T)iy) * 2.0 + -1.0) + 299]) + (rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)iy) * 2.0 + -1.0) + 299] - rtb_V_boundingbox[(int32_T)((1.0 +
      (real_T)iy) * 2.0 + -1.0) - 1]) * 0.3;
    OBXY_EL[(int32_T)sigma + 299] = ((rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
      iy) * 2.0) + 299] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) * 2.0)
      + 199]) * 0.3 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) * 2.0) +
      299]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) * 2.0) + 299] -
               rtb_V_boundingbox[(int32_T)((1.0 + (real_T)iy) * 2.0) - 1]) * 0.3;
  }

  for (total_length_tmp = 0; total_length_tmp < 13; total_length_tmp++) {
    for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 10;
         Path_RES_0_size_idx_1++) {
      b_idx = 11 * total_length_tmp + Path_RES_0_size_idx_1;
      x_0 = X_2[b_idx + 1] - X_2[b_idx];
      X_diff[Path_RES_0_size_idx_1 + 11 * total_length_tmp] = x_0;
      X_diff_0[Path_RES_0_size_idx_1 + 10 * total_length_tmp] = x_0;
    }

    jmax = 10 + 11 * total_length_tmp;
    X_diff[jmax] = X_diff_0[10 * total_length_tmp + 9];
    for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 10;
         Path_RES_0_size_idx_1++) {
      b_idx = 11 * total_length_tmp + Path_RES_0_size_idx_1;
      x_0 = Y[b_idx + 1] - Y[b_idx];
      Y_diff[Path_RES_0_size_idx_1 + 11 * total_length_tmp] = x_0;
      X_diff_0[Path_RES_0_size_idx_1 + 10 * total_length_tmp] = x_0;
    }

    Y_diff[jmax] = X_diff_0[10 * total_length_tmp + 9];
  }

  power_dw(X_diff, XY_difflen);
  power_dw(Y_diff, Path_vehFLY);
  for (total_length_tmp = 0; total_length_tmp < 143; total_length_tmp++) {
    Path_vehFLX[total_length_tmp] = XY_difflen[total_length_tmp] +
      Path_vehFLY[total_length_tmp];
  }

  power_dw3(Path_vehFLX, XY_difflen);
  for (total_length_tmp = 0; total_length_tmp < 143; total_length_tmp++) {
    x_0 = X_diff[total_length_tmp] / XY_difflen[total_length_tmp];
    sigma = Y_diff[total_length_tmp] / XY_difflen[total_length_tmp];
    b_a = 0.99993 * -sigma + X_2[total_length_tmp];
    Path_vehFLX[total_length_tmp] = b_a + 1.219116 * x_0;
    yy_idx_0 = 0.99993 * x_0 + Y[total_length_tmp];
    Path_vehFLY[total_length_tmp] = yy_idx_0 + 1.219116 * sigma;
    sigma_tmp = X_2[total_length_tmp] - 0.99993 * -sigma;
    Path_vehFRX[total_length_tmp] = sigma_tmp + 1.219116 * x_0;
    Path_vehFRY_tmp = Y[total_length_tmp] - 0.99993 * x_0;
    Path_vehFRY[total_length_tmp] = Path_vehFRY_tmp + 1.219116 * sigma;
    Path_vehRLX[total_length_tmp] = b_a - 4.876464 * x_0;
    Path_vehRLY[total_length_tmp] = yy_idx_0 - 4.876464 * sigma;
    Path_vehRRX[total_length_tmp] = sigma_tmp - 4.876464 * x_0;
    Path_vehRRY[total_length_tmp] = Path_vehFRY_tmp - 4.876464 * sigma;
    X_diff[total_length_tmp] = x_0;
    XY_difflen[total_length_tmp] = -sigma;
    Y_diff[total_length_tmp] = sigma;
  }

  for (ix = 0; ix < 13; ix++) {
    Path_col[ix << 2] = 0.0;
    if (!(Path_col[(ix << 2) + 3] == 1.0)) {
      n = 0;
      exitg1 = false;
      while ((!exitg1) && (n < 11)) {
        b_idx = 11 * ix + n;
        OBXY_m[0] = Path_vehFLX[b_idx];
        OBXY_m[2] = Path_vehFRX[b_idx];
        OBXY_m[4] = Path_vehRLX[b_idx];
        OBXY_m[6] = Path_vehRRX[b_idx];
        OBXY_m[1] = Path_vehFLY[b_idx];
        OBXY_m[3] = Path_vehFRY[b_idx];
        OBXY_m[5] = Path_vehRLY[b_idx];
        OBXY_m[7] = Path_vehRRY[b_idx];
        jmax = 0;
        exitg3 = false;
        while ((!exitg3) && (jmax <= i)) {
          sigma = (1.0 + (real_T)jmax) * 2.0;
          idx = (int32_T)(sigma + -1.0);
          sigma_tmp = OBXY_EL[idx + 99] - OBXY_EL[idx - 1];
          x_0 = std::sqrt(sigma_tmp * sigma_tmp + sigma_tmp * sigma_tmp);
          total_length_tmp = (int32_T)sigma;
          rtb_Oi_near_l[0] = -(OBXY_EL[total_length_tmp + 99] -
                               OBXY_EL[total_length_tmp - 1]) / x_0;
          rtb_Oi_near_l[1] = sigma_tmp / x_0;
          yy_idx_0 = OBXY_EL[total_length_tmp + 199] - OBXY_EL[(int32_T)((1.0 +
            (real_T)jmax) * 2.0) + 99];
          sigma_tmp = OBXY_EL[idx + 199] - OBXY_EL[(int32_T)((1.0 + (real_T)jmax)
            * 2.0 + -1.0) + 99];
          b_a = std::sqrt(yy_idx_0 * yy_idx_0 + sigma_tmp * sigma_tmp);
          yy_idx_0 = -yy_idx_0 / b_a;
          x_0 = sigma_tmp / b_a;
          rtb_Oi_near_o[0] = rtb_Oi_near_l[0];
          rtb_Oi_near_o[1] = yy_idx_0;
          rtb_Oi_near_o[4] = rtb_Oi_near_l[1];
          rtb_Oi_near_o[5] = x_0;
          rtb_Oi_near_o[2] = X_diff[b_idx];
          rtb_Oi_near_o[6] = Y_diff[b_idx];
          rtb_Oi_near_o[3] = XY_difflen[b_idx];
          rtb_Oi_near_o[7] = X_diff[11 * ix + n];
          rtb_Oi_near_o_0[0] = rtb_Oi_near_l[0];
          rtb_Oi_near_o_0[1] = yy_idx_0;
          rtb_Oi_near_o_0[4] = rtb_Oi_near_l[1];
          rtb_Oi_near_o_0[5] = x_0;
          rtb_Oi_near_o_0[2] = X_diff[11 * ix + n];
          rtb_Oi_near_o_0[6] = Y_diff[11 * ix + n];
          rtb_Oi_near_o_0[3] = XY_difflen[11 * ix + n];
          rtb_Oi_near_o_0[7] = X_diff[11 * ix + n];
          for (total_length_tmp = 0; total_length_tmp < 4; total_length_tmp++) {
            for (Path_RES_0_size_idx_1 = 0; Path_RES_0_size_idx_1 < 4;
                 Path_RES_0_size_idx_1++) {
              proj_veh[total_length_tmp + (Path_RES_0_size_idx_1 << 2)] = 0.0;
              proj_veh[total_length_tmp + (Path_RES_0_size_idx_1 << 2)] +=
                OBXY_m[Path_RES_0_size_idx_1 << 1] *
                rtb_Oi_near_o[total_length_tmp];
              proj_veh[total_length_tmp + (Path_RES_0_size_idx_1 << 2)] +=
                OBXY_m[(Path_RES_0_size_idx_1 << 1) + 1] *
                rtb_Oi_near_o[total_length_tmp + 4];
            }

            OBXY_EL_0[total_length_tmp << 1] = OBXY_EL[((int32_T)(sigma + -1.0)
              + 100 * total_length_tmp) - 1];
            OBXY_EL_0[1 + (total_length_tmp << 1)] = OBXY_EL[(100 *
              total_length_tmp + (int32_T)sigma) - 1];
          }

          for (idx = 0; idx < 4; idx++) {
            for (total_length_tmp = 0; total_length_tmp < 4; total_length_tmp++)
            {
              proj_ob[idx + (total_length_tmp << 2)] = 0.0;
              proj_ob[idx + (total_length_tmp << 2)] +=
                OBXY_EL_0[total_length_tmp << 1] * rtb_Oi_near_o_0[idx];
              proj_ob[idx + (total_length_tmp << 2)] += OBXY_EL_0
                [(total_length_tmp << 1) + 1] * rtb_Oi_near_o_0[idx + 4];
            }

            K1[idx] = proj_veh[idx];
          }

          x_0 = proj_veh[0];
          sigma = proj_veh[1];
          b_a = proj_veh[2];
          yy_idx_0 = proj_veh[3];
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

            sigma_tmp = x_0;
            if ((!rtIsNaN(proj_veh[(idx + 1) << 2])) && (rtIsNaN(x_0) || (x_0 <
                  proj_veh[(idx + 1) << 2]))) {
              sigma_tmp = proj_veh[(idx + 1) << 2];
            }

            x_0 = sigma_tmp;
            sigma_tmp = sigma;
            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 1])) && (rtIsNaN(sigma) ||
                 (sigma < proj_veh[((idx + 1) << 2) + 1]))) {
              sigma_tmp = proj_veh[((idx + 1) << 2) + 1];
            }

            sigma = sigma_tmp;
            sigma_tmp = b_a;
            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 2])) && (rtIsNaN(b_a) ||
                 (b_a < proj_veh[((idx + 1) << 2) + 2]))) {
              sigma_tmp = proj_veh[((idx + 1) << 2) + 2];
            }

            b_a = sigma_tmp;
            sigma_tmp = yy_idx_0;
            if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 3])) && (rtIsNaN(yy_idx_0)
                 || (yy_idx_0 < proj_veh[((idx + 1) << 2) + 3]))) {
              sigma_tmp = proj_veh[((idx + 1) << 2) + 3];
            }

            yy_idx_0 = sigma_tmp;
          }

          minmax_veh[0] = K1[0];
          minmax_veh[4] = x_0;
          minmax_veh[1] = K1[1];
          minmax_veh[5] = sigma;
          minmax_veh[2] = K1[2];
          minmax_veh[6] = b_a;
          minmax_veh[3] = K1[3];
          minmax_veh[7] = yy_idx_0;
          K1[0] = proj_ob[0];
          K1[1] = proj_ob[1];
          K1[2] = proj_ob[2];
          K1[3] = proj_ob[3];
          x_0 = proj_ob[0];
          sigma = proj_ob[1];
          b_a = proj_ob[2];
          yy_idx_0 = proj_ob[3];
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

            sigma_tmp = x_0;
            if ((!rtIsNaN(proj_ob[(idx + 1) << 2])) && (rtIsNaN(x_0) || (x_0 <
                  proj_ob[(idx + 1) << 2]))) {
              sigma_tmp = proj_ob[(idx + 1) << 2];
            }

            x_0 = sigma_tmp;
            sigma_tmp = sigma;
            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 1])) && (rtIsNaN(sigma) ||
                 (sigma < proj_ob[((idx + 1) << 2) + 1]))) {
              sigma_tmp = proj_ob[((idx + 1) << 2) + 1];
            }

            sigma = sigma_tmp;
            sigma_tmp = b_a;
            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 2])) && (rtIsNaN(b_a) ||
                 (b_a < proj_ob[((idx + 1) << 2) + 2]))) {
              sigma_tmp = proj_ob[((idx + 1) << 2) + 2];
            }

            b_a = sigma_tmp;
            sigma_tmp = yy_idx_0;
            if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 3])) && (rtIsNaN(yy_idx_0) ||
                 (yy_idx_0 < proj_ob[((idx + 1) << 2) + 3]))) {
              sigma_tmp = proj_ob[((idx + 1) << 2) + 3];
            }

            yy_idx_0 = sigma_tmp;
          }

          minmax_obj[0] = K1[0];
          minmax_obj[4] = x_0;
          minmax_obj[1] = K1[1];
          minmax_obj[5] = sigma;
          minmax_obj[2] = K1[2];
          minmax_obj[6] = b_a;
          minmax_obj[3] = K1[3];
          minmax_obj[7] = yy_idx_0;
          iy = 0;
          exitg4 = false;
          while ((!exitg4) && (iy < 4)) {
            if (minmax_veh[iy] > minmax_obj[4 + iy]) {
              Path_col[ix << 2] = 0.0;
              exitg4 = true;
            } else if (minmax_veh[4 + iy] < minmax_obj[iy]) {
              Path_col[ix << 2] = 0.0;
              exitg4 = true;
            } else {
              Path_col[ix << 2] = 1.0;
              iy++;
            }
          }

          if (Path_col[ix << 2] == 1.0) {
            Path_col[2 + (ix << 2)] = 1.0 + (real_T)jmax;
            exitg3 = true;
          } else {
            jmax++;
          }
        }

        if (Path_col[ix << 2] == 1.0) {
          Path_col[1 + (ix << 2)] = 1.0 + (real_T)n;
          exitg1 = true;
        } else {
          n++;
        }
      }
    }

    Cobs[ix] = Path_col[ix << 2];
    Cobs_0[ix] = Path_col[ix << 2];
  }

  sigma = std(Cobs_0);
  if (sigma != 0.0) {
    x_0 = sigma * sigma * 2.0;
    b_a = 2.5066282746310002 * sigma;
    for (idx = 0; idx < 13; idx++) {
      i = 1 + idx;
      for (total_length_tmp = 0; total_length_tmp < 13; total_length_tmp++) {
        e_maxval[total_length_tmp] = (i - total_length_tmp) - 1;
      }

      power_dw3x(e_maxval, Cobs_0);
      for (i = 0; i < 13; i++) {
        e_maxval[i] = -Cobs_0[i] / x_0;
      }

      exp_n(e_maxval);
      for (i = 0; i < 13; i++) {
        Cobs_0[i] = Path_col[i << 2] * (e_maxval[i] / b_a);
      }

      Cobs[idx] = sum_a(Cobs_0);
      if ((1 + idx == 1) && (Path_col[0] == 1.0)) {
        Cobs[0] += std::exp(-1.0 / (sigma * sigma * 2.0)) / (2.5066282746310002 *
          sigma);
      } else {
        if ((1 + idx == 13) && (Path_col[48] == 1.0)) {
          Cobs[12] += std::exp(-1.0 / (sigma * sigma * 2.0)) /
            (2.5066282746310002 * sigma);
        }
      }
    }

    b_x_0 = rtIsNaN(Cobs[0]);
    if (!b_x_0) {
      idx = 1;
    } else {
      idx = 0;
      jmax = 2;
      exitg1 = false;
      while ((!exitg1) && (jmax < 14)) {
        if (!rtIsNaN(Cobs[jmax - 1])) {
          idx = jmax;
          exitg1 = true;
        } else {
          jmax++;
        }
      }
    }

    if (idx == 0) {
      sigma = Cobs[0];
    } else {
      sigma = Cobs[idx - 1];
      while (idx + 1 < 14) {
        if (sigma < Cobs[idx]) {
          sigma = Cobs[idx];
        }

        idx++;
      }
    }

    if (sigma != 1.0) {
      if (!b_x_0) {
        b_idx = 1;
      } else {
        b_idx = 0;
        idx = 2;
        exitg1 = false;
        while ((!exitg1) && (idx < 14)) {
          if (!rtIsNaN(Cobs[idx - 1])) {
            b_idx = idx;
            exitg1 = true;
          } else {
            idx++;
          }
        }
      }

      if (b_idx == 0) {
        sigma = Cobs[0];
      } else {
        sigma = Cobs[b_idx - 1];
        while (b_idx + 1 < 14) {
          if (sigma < Cobs[b_idx]) {
            sigma = Cobs[b_idx];
          }

          b_idx++;
        }
      }

      for (i = 0; i < 13; i++) {
        Cobs[i] /= sigma;
      }
    }
  }

  for (i = 0; i < 13; i++) {
    Clane[i] = Path_col[(i << 2) + 3];
    Cobs_0[i] = Path_col[(i << 2) + 3];
  }

  sigma = std(Cobs_0);
  if (sigma != 0.0) {
    x_0 = sigma * sigma * 2.0;
    b_a = 2.5066282746310002 * sigma;
    for (ix = 0; ix < 13; ix++) {
      i = 1 + ix;
      for (total_length_tmp = 0; total_length_tmp < 13; total_length_tmp++) {
        e_maxval[total_length_tmp] = (i - total_length_tmp) - 1;
      }

      power_dw3x(e_maxval, Cobs_0);
      for (i = 0; i < 13; i++) {
        e_maxval[i] = -Cobs_0[i] / x_0;
      }

      exp_n(e_maxval);
      for (i = 0; i < 13; i++) {
        Cobs_0[i] = Path_col[(i << 2) + 3] * (e_maxval[i] / b_a);
      }

      Clane[ix] = sum_a(Cobs_0);
      if ((1 + ix == 1) && (Path_col[3] == 1.0)) {
        Clane[0] += std::exp(-1.0 / (sigma * sigma * 2.0)) / (2.5066282746310002
          * sigma);
      } else {
        if ((1 + ix == 13) && (Path_col[51] == 1.0)) {
          Clane[12] += std::exp(-1.0 / (sigma * sigma * 2.0)) /
            (2.5066282746310002 * sigma);
        }
      }
    }

    b_x_0 = rtIsNaN(Clane[0]);
    if (!b_x_0) {
      b_idx = 1;
    } else {
      b_idx = 0;
      idx = 2;
      exitg1 = false;
      while ((!exitg1) && (idx < 14)) {
        if (!rtIsNaN(Clane[idx - 1])) {
          b_idx = idx;
          exitg1 = true;
        } else {
          idx++;
        }
      }
    }

    if (b_idx == 0) {
      sigma = Clane[0];
    } else {
      sigma = Clane[b_idx - 1];
      while (b_idx + 1 < 14) {
        if (sigma < Clane[b_idx]) {
          sigma = Clane[b_idx];
        }

        b_idx++;
      }
    }

    if (sigma != 1.0) {
      if (!b_x_0) {
        b_idx = 1;
      } else {
        b_idx = 0;
        idx = 2;
        exitg1 = false;
        while ((!exitg1) && (idx < 14)) {
          if (!rtIsNaN(Clane[idx - 1])) {
            b_idx = idx;
            exitg1 = true;
          } else {
            idx++;
          }
        }
      }

      if (b_idx == 0) {
        sigma = Clane[0];
      } else {
        sigma = Clane[b_idx - 1];
        while (b_idx + 1 < 14) {
          if (sigma < Clane[b_idx]) {
            sigma = Clane[b_idx];
          }

          b_idx++;
        }
      }

      for (i = 0; i < 13; i++) {
        Clane[i] /= sigma;
      }
    }
  }

  for (i = 0; i < 11; i++) {
    b_Path_dis_data[i] = LastPath[i] - rtb_X[0];
  }

  power_d(b_Path_dis_data, x);
  for (i = 0; i < 11; i++) {
    b_Path_dis_data[i] = LastPath[11 + i] - rtb_X[1];
  }

  power_d(b_Path_dis_data, XY_difflen_0);
  for (i = 0; i < 11; i++) {
    b_Path_dis_data[i] = x[i] + XY_difflen_0[i];
  }

  sqrt_l(b_Path_dis_data);
  if (!rtIsNaN(b_Path_dis_data[0])) {
    b_idx = 1;
  } else {
    b_idx = 0;
    idx = 2;
    exitg1 = false;
    while ((!exitg1) && (idx < 12)) {
      if (!rtIsNaN(b_Path_dis_data[idx - 1])) {
        b_idx = idx;
        exitg1 = true;
      } else {
        idx++;
      }
    }
  }

  if (b_idx == 0) {
    b_idx = 1;
  } else {
    sigma = b_Path_dis_data[b_idx - 1];
    for (jmax = b_idx; jmax + 1 < 12; jmax++) {
      if (sigma > b_Path_dis_data[jmax]) {
        sigma = b_Path_dis_data[jmax];
        b_idx = jmax + 1;
      }
    }
  }

  jj = 12 - b_idx;
  loop_ub = -b_idx;
  for (i = 0; i <= loop_ub + 11; i++) {
    LastPath_overlap_data[i] = LastPath[(b_idx + i) - 1];
  }

  loop_ub = -b_idx;
  for (i = 0; i <= loop_ub + 11; i++) {
    LastPath_overlap_data[i + jj] = LastPath[(b_idx + i) + 10];
  }

  for (jmax = 0; jmax < 13; jmax++) {
    for (i = 0; i < 11; i++) {
      b_Path_dis_data[i] = X_2[11 * jmax + i] - LastPath[10];
    }

    power_d(b_Path_dis_data, x);
    for (i = 0; i < 11; i++) {
      b_Path_dis_data[i] = Y[11 * jmax + i] - LastPath[21];
    }

    power_d(b_Path_dis_data, XY_difflen_0);
    for (i = 0; i < 11; i++) {
      b_Path_dis_data[i] = x[i] + XY_difflen_0[i];
    }

    sqrt_l(b_Path_dis_data);
    if (!rtIsNaN(b_Path_dis_data[0])) {
      idx = 0;
    } else {
      idx = -1;
      n = 2;
      exitg1 = false;
      while ((!exitg1) && (n < 12)) {
        if (!rtIsNaN(b_Path_dis_data[n - 1])) {
          idx = n - 1;
          exitg1 = true;
        } else {
          n++;
        }
      }
    }

    if (idx + 1 == 0) {
      idx = 0;
    } else {
      sigma = b_Path_dis_data[idx];
      for (n = idx + 1; n + 1 < 12; n++) {
        if (sigma > b_Path_dis_data[n]) {
          sigma = b_Path_dis_data[n];
          idx = n;
        }
      }
    }

    Path_overlap_size[0] = idx + 1;
    if (0 <= idx) {
      memcpy(&Path_overlap_data[0], &X_2[jmax * 11], (idx + 1) * sizeof(real_T));
    }

    for (i = 0; i <= idx; i++) {
      Path_overlap_data[i + Path_overlap_size[0]] = Y[11 * jmax + i];
    }

    if (12 - b_idx >= Path_overlap_size[0]) {
      idx = 13 - (b_idx + Path_overlap_size[0]);
      if (idx > 12 - b_idx) {
        idx = 1;
        n = 0;
      } else {
        n = 12 - b_idx;
      }

      i = idx - 1;
      idx = n - i;
      LastPath_overlap_size_0[0] = idx;
      LastPath_overlap_size_0[1] = 2;
      for (total_length_tmp = 0; total_length_tmp < idx; total_length_tmp++) {
        LastPath_overlap_data_0[total_length_tmp] = LastPath_overlap_data[i +
          total_length_tmp] - Path_overlap_data[total_length_tmp];
      }

      for (total_length_tmp = 0; total_length_tmp < idx; total_length_tmp++) {
        LastPath_overlap_data_0[total_length_tmp + idx] = LastPath_overlap_data
          [(i + total_length_tmp) + jj] - Path_overlap_data[total_length_tmp +
          Path_overlap_size[0]];
      }

      power_dw3xd(LastPath_overlap_data_0, LastPath_overlap_size_0,
                  Path_overlap_data, Path_overlap_size);
      Path_overlap_size_1[0] = 2;
      Path_overlap_size_1[1] = Path_overlap_size[0];
      loop_ub = Path_overlap_size[0];
      for (i = 0; i < loop_ub; i++) {
        LastPath_overlap_data_0[i << 1] = Path_overlap_data[i];
        LastPath_overlap_data_0[1 + (i << 1)] = Path_overlap_data[i +
          Path_overlap_size[0]];
      }

      sum_ae(LastPath_overlap_data_0, Path_overlap_size_1, b_Path_dis_data,
             oi_xy_size);
      sqrt_l5(b_Path_dis_data, oi_xy_size);
      loop_ub = oi_xy_size[1];
      for (i = 0; i < loop_ub; i++) {
        x[i] = b_Path_dis_data[oi_xy_size[0] * i];
      }

      i = oi_xy_size[1];
      total_length_tmp = oi_xy_size[1];
      if (0 <= i - 1) {
        memcpy(&Path_dis_data[0], &x[0], i * sizeof(real_T));
      }
    } else {
      ix = 12 - b_idx;
      LastPath_overlap_size[0] = ix;
      LastPath_overlap_size[1] = 2;
      for (i = 0; i < ix; i++) {
        LastPath_overlap_data_0[i] = LastPath_overlap_data[i] -
          Path_overlap_data[i];
      }

      for (i = 0; i < ix; i++) {
        LastPath_overlap_data_0[i + ix] = LastPath_overlap_data[i + jj] -
          Path_overlap_data[i + Path_overlap_size[0]];
      }

      power_dw3xd(LastPath_overlap_data_0, LastPath_overlap_size,
                  Path_overlap_data, Path_overlap_size);
      Path_overlap_size_0[0] = 2;
      Path_overlap_size_0[1] = Path_overlap_size[0];
      loop_ub = Path_overlap_size[0];
      for (i = 0; i < loop_ub; i++) {
        LastPath_overlap_data_0[i << 1] = Path_overlap_data[i];
        LastPath_overlap_data_0[1 + (i << 1)] = Path_overlap_data[i +
          Path_overlap_size[0]];
      }

      sum_ae(LastPath_overlap_data_0, Path_overlap_size_0, b_Path_dis_data,
             oi_xy_size);
      sqrt_l5(b_Path_dis_data, oi_xy_size);
      loop_ub = oi_xy_size[1];
      for (i = 0; i < loop_ub; i++) {
        b_Path_dis_data_0[i] = b_Path_dis_data[oi_xy_size[0] * i];
      }

      i = oi_xy_size[1];
      total_length_tmp = oi_xy_size[1];
      if (0 <= i - 1) {
        memcpy(&Path_dis_data[0], &b_Path_dis_data_0[0], i * sizeof(real_T));
      }
    }

    if (total_length_tmp > 1) {
      i = total_length_tmp;
    } else {
      i = 1;
    }

    if (mod((real_T)i) == 0.0) {
      if (total_length_tmp > 1) {
        idx = total_length_tmp - 1;
      } else {
        idx = 0;
      }

      oi_xy_size[1] = idx;
      loop_ub = idx - 1;
      for (i = 0; i <= loop_ub; i++) {
        b_Path_dis_data[i] = 4.0;
      }
    } else {
      if (total_length_tmp > 1) {
        idx = total_length_tmp;
      } else {
        idx = 1;
      }

      oi_xy_size[1] = idx;
      loop_ub = idx - 1;
      for (i = 0; i <= loop_ub; i++) {
        b_Path_dis_data[i] = 4.0;
      }
    }

    b_Path_dis_data[0] = 1.0;
    b_Path_dis_data[oi_xy_size[1] - 1] = 1.0;
    if (3 > oi_xy_size[1] - 2) {
      colj = 1;
      iy = 1;
      ix = 0;
    } else {
      colj = 3;
      iy = 2;
      ix = oi_xy_size[1] - 2;
    }

    idx = div_nde_s32_floor((int8_T)ix - colj, iy);
    for (i = 0; i <= idx; i++) {
      l_data[i] = (int8_T)((int8_T)((int8_T)(iy * (int8_T)i) + colj) - 1);
    }

    for (i = 0; i <= idx; i++) {
      b_Path_dis_data[l_data[i]] = 2.0;
    }

    sigma = 0.0;
    for (i = 0; i < oi_xy_size[1]; i++) {
      sigma += b_Path_dis_data[i] * Path_dis_data[i];
    }

    if (!(total_length_tmp > 1)) {
      total_length_tmp = 1;
    }

    Cobs_0[jmax] = L_path[jmax] / 11.0 * sigma / 3.0 / (L_path[jmax] * (real_T)
      total_length_tmp / 11.0);
  }

  abs_a(K, XY_difflen);
  for (idx = 0; idx < 13; idx++) {
    rtb_J_out[idx] = XY_difflen[11 * idx];
    for (b_idx = 0; b_idx < 10; b_idx++) {
      x_0 = rtb_J_out[idx];
      i = (11 * idx + b_idx) + 1;
      if ((!rtIsNaN(XY_difflen[i])) && (rtIsNaN(rtb_J_out[idx]) ||
           (rtb_J_out[idx] < XY_difflen[i]))) {
        x_0 = XY_difflen[i];
      }

      rtb_J_out[idx] = x_0;
    }
  }

  abs_a(K, XY_difflen);
  for (jmax = 0; jmax < 13; jmax++) {
    e_maxval[jmax] = XY_difflen[11 * jmax];
    for (idx = 0; idx < 10; idx++) {
      x_0 = e_maxval[jmax];
      i = (11 * jmax + idx) + 1;
      if ((!rtIsNaN(XY_difflen[i])) && (rtIsNaN(e_maxval[jmax]) ||
           (e_maxval[jmax] < XY_difflen[i]))) {
        x_0 = XY_difflen[i];
      }

      e_maxval[jmax] = x_0;
    }
  }

  for (i = 0; i < 13; i++) {
    e_maxval[i] *= 10.0;
  }

  if (!rtIsNaN(e_maxval[0])) {
    b_idx = 1;
  } else {
    b_idx = 0;
    idx = 2;
    exitg1 = false;
    while ((!exitg1) && (idx < 14)) {
      if (!rtIsNaN(e_maxval[idx - 1])) {
        b_idx = idx;
        exitg1 = true;
      } else {
        idx++;
      }
    }
  }

  if (b_idx == 0) {
    yy_idx_0 = e_maxval[0];
  } else {
    yy_idx_0 = e_maxval[b_idx - 1];
    while (b_idx + 1 < 14) {
      if (yy_idx_0 < e_maxval[b_idx]) {
        yy_idx_0 = e_maxval[b_idx];
      }

      b_idx++;
    }
  }

  abs_a(K_1, XY_difflen);
  for (idx = 0; idx < 13; idx++) {
    e_maxval[idx] = XY_difflen[11 * idx];
    for (b_idx = 0; b_idx < 10; b_idx++) {
      x_0 = e_maxval[idx];
      i = (11 * idx + b_idx) + 1;
      if ((!rtIsNaN(XY_difflen[i])) && (rtIsNaN(e_maxval[idx]) || (e_maxval[idx]
            < XY_difflen[i]))) {
        x_0 = XY_difflen[i];
      }

      e_maxval[idx] = x_0;
    }
  }

  abs_a(K_1, XY_difflen);
  for (idx = 0; idx < 13; idx++) {
    g_maxval[idx] = XY_difflen[11 * idx];
    for (b_idx = 0; b_idx < 10; b_idx++) {
      x_0 = g_maxval[idx];
      if ((!rtIsNaN(XY_difflen[(11 * idx + b_idx) + 1])) && (rtIsNaN
           (g_maxval[idx]) || (g_maxval[idx] < XY_difflen[(11 * idx + b_idx) + 1])))
      {
        x_0 = XY_difflen[(11 * idx + b_idx) + 1];
      }

      g_maxval[idx] = x_0;
    }

    g_maxval[idx] *= 10.0;
  }

  if (!rtIsNaN(g_maxval[0])) {
    idx = 1;
  } else {
    idx = 0;
    b_idx = 2;
    exitg1 = false;
    while ((!exitg1) && (b_idx < 14)) {
      if (!rtIsNaN(g_maxval[b_idx - 1])) {
        idx = b_idx;
        exitg1 = true;
      } else {
        b_idx++;
      }
    }
  }

  if (idx == 0) {
    sigma = g_maxval[0];
  } else {
    sigma = g_maxval[idx - 1];
    while (idx + 1 < 14) {
      if (sigma < g_maxval[idx]) {
        sigma = g_maxval[idx];
      }

      idx++;
    }
  }

  if (!rtIsNaN(offset[0])) {
    idx = 1;
  } else {
    idx = 0;
    b_idx = 2;
    exitg1 = false;
    while ((!exitg1) && (b_idx < 14)) {
      if (!rtIsNaN(offset[b_idx - 1])) {
        idx = b_idx;
        exitg1 = true;
      } else {
        b_idx++;
      }
    }
  }

  if (idx == 0) {
    x_0 = offset[0];
  } else {
    x_0 = offset[idx - 1];
    while (idx + 1 < 14) {
      if (x_0 < offset[idx]) {
        x_0 = offset[idx];
      }

      idx++;
    }
  }

  b_x_0 = rtIsNaN(Cobs_0[0]);
  if (!b_x_0) {
    idx = 1;
  } else {
    idx = 0;
    b_idx = 2;
    exitg1 = false;
    while ((!exitg1) && (b_idx < 14)) {
      if (!rtIsNaN(Cobs_0[b_idx - 1])) {
        idx = b_idx;
        exitg1 = true;
      } else {
        b_idx++;
      }
    }
  }

  if (idx == 0) {
    b_a = Cobs_0[0];
  } else {
    b_a = Cobs_0[idx - 1];
    while (idx + 1 < 14) {
      if (b_a < Cobs_0[idx]) {
        b_a = Cobs_0[idx];
      }

      idx++;
    }
  }

  if (!(b_a == 0.0)) {
    if (!b_x_0) {
      idx = 1;
    } else {
      idx = 0;
      b_idx = 2;
      exitg1 = false;
      while ((!exitg1) && (b_idx < 14)) {
        if (!rtIsNaN(Cobs_0[b_idx - 1])) {
          idx = b_idx;
          exitg1 = true;
        } else {
          b_idx++;
        }
      }
    }

    if (idx == 0) {
      b_a = Cobs_0[0];
    } else {
      b_a = Cobs_0[idx - 1];
      while (idx + 1 < 14) {
        if (b_a < Cobs_0[idx]) {
          b_a = Cobs_0[idx];
        }

        idx++;
      }
    }

    for (i = 0; i < 13; i++) {
      Cobs_0[i] /= b_a;
    }
  }

  for (i = 0; i < 13; i++) {
    rtb_J_out[i] = (((((L_path[i] / rtb_Gain_p * 0.5 + rtb_J_out[i] * 10.0 /
                        yy_idx_0) + e_maxval[i] * 10.0 / sigma) + offset[i] /
                      x_0 * total_length) + 30.0 * Cobs[i]) + 10.0 * Cobs_0[i])
      + 30.0 * Clane[i];
  }

  if (!rtIsNaN(rtb_J_out[0])) {
    b_idx = 1;
  } else {
    b_idx = 0;
    idx = 2;
    exitg1 = false;
    while ((!exitg1) && (idx < 14)) {
      if (!rtIsNaN(rtb_J_out[idx - 1])) {
        b_idx = idx;
        exitg1 = true;
      } else {
        idx++;
      }
    }
  }

  if (b_idx == 0) {
    b_idx = 1;
  } else {
    rtb_Gain_p = rtb_J_out[b_idx - 1];
    for (idx = b_idx; idx + 1 < 14; idx++) {
      if (rtb_Gain_p > rtb_J_out[idx]) {
        rtb_Gain_p = rtb_J_out[idx];
        b_idx = idx + 1;
      }
    }
  }

  if (rtU.Path_flag == 1.0) {
    colj = b_idx - 1;
  } else {
    colj = 6;
  }

  for (i = 0; i < 6; i++) {
    rtb_YP_final_l[i] = rtb_YP_final_o[i];
    rtb_YP_final_l[6 + i] = XP2[i];
    rtb_YP_final_l[12 + i] = XP3[i];
    rtb_YP_final_l[18 + i] = XP4[i];
    rtb_YP_final_l[24 + i] = XP5[i];
    rtb_YP_final_l[30 + i] = XP6[i];
    rtb_YP_final_l[36 + i] = XP7[i];
    rtb_YP_final_l[42 + i] = XP8[i];
    rtb_YP_final_l[48 + i] = XP9[i];
    rtb_YP_final_l[54 + i] = XP10[i];
    rtb_YP_final_l[60 + i] = XP11[i];
    rtb_YP_final_l[66 + i] = XP12[i];
    rtb_YP_final_l[72 + i] = XP13[i];
    jmax = 6 * colj + i;
    rtb_XP_final_g[i] = rtb_YP_final_l[jmax];
    YP1_0[i] = YP1[i];
    YP1_0[6 + i] = YP2[i];
    YP1_0[12 + i] = YP3[i];
    YP1_0[18 + i] = YP4[i];
    YP1_0[24 + i] = YP5[i];
    YP1_0[30 + i] = YP6[i];
    YP1_0[36 + i] = YP7[i];
    YP1_0[42 + i] = YP8[i];
    YP1_0[48 + i] = YP9[i];
    YP1_0[54 + i] = YP10[i];
    YP1_0[60 + i] = YP11[i];
    YP1_0[66 + i] = YP12[i];
    YP1_0[72 + i] = YP13[i];
    rtb_YP_final_o[i] = YP1_0[jmax];
  }

  Length_1_0[0] = Length_1;
  Length_1_0[2] = x_endpoint2;
  Length_1_0[4] = x_endpoint3;
  Length_1_0[6] = x_endpoint4;
  Length_1_0[8] = x_endpoint5;
  Length_1_0[10] = x_endpoint6;
  Length_1_0[12] = ajj;
  Length_1_0[14] = x_endpoint8;
  Length_1_0[16] = x_endpoint9;
  Length_1_0[18] = x_endpoint10;
  Length_1_0[20] = x_endpoint11;
  Length_1_0[22] = x_endpoint12;
  Length_1_0[24] = x_endpoint13;
  Length_1_0[1] = target_k;
  Length_1_0[3] = y_endpoint2;
  Length_1_0[5] = y_endpoint3;
  Length_1_0[7] = y_endpoint4;
  Length_1_0[9] = y_endpoint5;
  Length_1_0[11] = y_endpoint6;
  Length_1_0[13] = count_1;
  Length_1_0[15] = delta_offset;
  Length_1_0[17] = offset_2;
  Length_1_0[19] = offset_3;
  Length_1_0[21] = offset_4;
  Length_1_0[23] = offset_5;
  Length_1_0[25] = offset_6;

  // MATLAB Function: '<S2>/EndPointDecision1' incorporates:
  //   MATLAB Function: '<S2>/DynamicPathPlanning'

  xy_ends_POS_size_idx_0 = 20000;
  Path_RES_0_size_idx_1 = 2;
  memset(&rtDW.Path_RES_0_data[0], 0, 40000U * sizeof(real_T));
  memset(&rtDW.Path_RES_0_1[0], 0, 40000U * sizeof(real_T));
  rtb_Gain_p = 0.0;
  count_1 = 0.0;
  iy = 0;
  sigma = rtb_Forward_Static_Path_x_h[1] - rtb_Forward_Static_Path_x_h[0];
  b_a = rtb_Forward_Static_Path_y_p[1] - rtb_Forward_Static_Path_y_p[0];
  Length_1 = std::sqrt(sigma * sigma + b_a * b_a);
  ajj = rt_atan2d_snf(rtb_Forward_Static_Path_y_p[1] -
                      rtb_Forward_Static_Path_y_p[0],
                      rtb_Forward_Static_Path_x_h[1] -
                      rtb_Forward_Static_Path_x_h[0]);
  if (Length_1 > 0.1) {
    Length_1 = rt_roundd_snf(Length_1 / 0.1);
    for (n = 0; n < (int32_T)Length_1; n++) {
      count_1 = ((1.0 + (real_T)n) - 1.0) * 0.1;
      rtDW.Path_RES_0_1[n] = count_1 * std::cos(ajj) +
        rtb_Forward_Static_Path_x_h[0];
      rtDW.Path_RES_0_1[20000 + n] = count_1 * std::sin(ajj) +
        rtb_Forward_Static_Path_y_p[0];
      count_1 = 1.0 + (real_T)n;
    }
  } else {
    rtDW.Path_RES_0_1[0] = rtb_Forward_Static_Path_x_h[0];
    rtDW.Path_RES_0_1[20000] = rtb_Forward_Static_Path_y_p[0];
    count_1 = 1.0;
  }

  if (1.0 > count_1) {
    jj = 0;
  } else {
    jj = (int32_T)count_1;
  }

  Path_RES_1_size_idx_0 = jj;
  if (0 <= jj - 1) {
    memcpy(&rtDW.Path_RES_1_data[0], &rtDW.Path_RES_0_1[0], jj * sizeof(real_T));
  }

  for (i = 0; i < jj; i++) {
    rtDW.Path_RES_1_data[i + jj] = rtDW.Path_RES_0_1[i + 20000];
  }

  for (i = 0; i < jj; i++) {
    rtDW.rtb_X_data[i] = Length_1_0[colj << 1] - rtDW.Path_RES_1_data[i];
  }

  power_jb(rtDW.rtb_X_data, &jj, rtDW.tmp_data, &n);
  for (i = 0; i < jj; i++) {
    rtDW.rtb_X_data[i] = Length_1_0[(colj << 1) + 1] - rtDW.Path_RES_1_data[i +
      jj];
  }

  power_jb(rtDW.rtb_X_data, &jj, rtDW.tmp_data_c, &jmax);
  for (i = 0; i < n; i++) {
    rtDW.ob_distance_data[i] = rtDW.tmp_data[i] + rtDW.tmp_data_c[i];
  }

  if (n <= 2) {
    if (n == 1) {
      idx = 0;
    } else if (rtDW.ob_distance_data[0] > rtDW.ob_distance_data[1]) {
      idx = 1;
    } else if (rtIsNaN(rtDW.ob_distance_data[0])) {
      if (!rtIsNaN(rtDW.ob_distance_data[1])) {
        i = 2;
      } else {
        i = 1;
      }

      idx = i - 1;
    } else {
      idx = 0;
    }
  } else {
    if (!rtIsNaN(rtDW.ob_distance_data[0])) {
      idx = 0;
    } else {
      idx = -1;
      jmax = 2;
      exitg1 = false;
      while ((!exitg1) && (jmax <= n)) {
        if (!rtIsNaN(rtDW.ob_distance_data[jmax - 1])) {
          idx = jmax - 1;
          exitg1 = true;
        } else {
          jmax++;
        }
      }
    }

    if (idx + 1 == 0) {
      idx = 0;
    } else {
      sigma = rtDW.ob_distance_data[idx];
      for (jmax = idx + 1; jmax < n; jmax++) {
        if (sigma > rtDW.ob_distance_data[jmax]) {
          sigma = rtDW.ob_distance_data[jmax];
          idx = jmax;
        }
      }
    }
  }

  Length_1 = count_1 - (real_T)(idx + 1);
  if (rtDW.SFunction_DIMS2 - 2 >= 1) {
    for (ix = 1; ix - 1 <= rtDW.SFunction_DIMS2 - 3; ix++) {
      if (iy == 0) {
        b_a = rtb_Forward_Static_Path_x_h[ix + 1] -
          rtb_Forward_Static_Path_x_h[ix];
        sigma = rtb_Forward_Static_Path_y_p[ix + 1] -
          rtb_Forward_Static_Path_y_p[ix];
        ajj = std::sqrt(b_a * b_a + sigma * sigma);
        count_1 = rt_atan2d_snf(rtb_Forward_Static_Path_y_p[ix + 1] -
          rtb_Forward_Static_Path_y_p[ix], rtb_Forward_Static_Path_x_h[ix + 1] -
          rtb_Forward_Static_Path_x_h[ix]);
        if (ajj >= 0.1) {
          ajj = rt_roundd_snf(ajj / 0.1);
          for (n = 0; n < (int32_T)ajj; n++) {
            x_endpoint2 = ((1.0 + (real_T)n) - 1.0) * 0.1;
            jmax = (int32_T)((1.0 + (real_T)n) + rtb_Gain_p);
            rtDW.Path_RES_0_data[jmax - 1] = x_endpoint2 * std::cos(count_1) +
              rtb_Forward_Static_Path_x_h[ix];
            rtDW.Path_RES_0_data[jmax + 19999] = x_endpoint2 * std::sin(count_1)
              + rtb_Forward_Static_Path_y_p[ix];
          }

          rtb_Gain_p += ajj;
        } else {
          rtDW.Path_RES_0_data[(int32_T)(1.0 + rtb_Gain_p) - 1] =
            rtb_Forward_Static_Path_x_h[ix];
          rtDW.Path_RES_0_data[(int32_T)(1.0 + rtb_Gain_p) + 19999] =
            rtb_Forward_Static_Path_y_p[ix];
          rtb_Gain_p++;
        }

        if (rtb_Gain_p > 150.0 - Length_1) {
          iy = 1;
        }
      }
    }
  } else {
    xy_ends_POS_size_idx_0 = 0;
    Path_RES_0_size_idx_1 = 0;
  }

  count_1 = (real_T)(idx + 1) + 150.0;
  if ((xy_ends_POS_size_idx_0 == 0) || (Path_RES_0_size_idx_1 == 0)) {
    if (count_1 <= jj) {
      if (idx + 1 > (int32_T)count_1) {
        idx = 0;
      }

      ajj = rtDW.Path_RES_1_data[idx + 149];
      count_1 = rtDW.Path_RES_1_data[(idx + jj) + 149];
      rtb_Gain_p = 15.0;
    } else {
      if (idx + 1 > jj) {
        idx = 0;
        iy = 0;
      } else {
        iy = jj;
      }

      n = iy - idx;
      total_length_tmp = n + idx;
      ajj = rtDW.Path_RES_1_data[total_length_tmp - 1];
      count_1 = rtDW.Path_RES_1_data[(total_length_tmp + jj) - 1];
      if (n == 0) {
        n = 0;
      } else {
        if (!(n > 2)) {
          n = 2;
        }
      }

      rtb_Gain_p = (real_T)n * 0.1;
    }
  } else {
    if (idx + 1 > jj) {
      idx = 0;
      jmax = 0;
    } else {
      jmax = jj;
    }

    if (1.0 > rtb_Gain_p) {
      iy = 0;
    } else {
      iy = (int32_T)rtb_Gain_p;
    }

    loop_ub = jmax - idx;
    if (!(loop_ub == 0)) {
      n = 2;
      ix = loop_ub;
    } else {
      if (!(iy == 0)) {
        n = Path_RES_0_size_idx_1;
      } else {
        n = 2;
      }

      ix = 0;
    }

    if (!(iy == 0)) {
      jj = iy;
    } else {
      jj = 0;
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Path_RES_0_1[i] = rtDW.Path_RES_1_data[idx + i];
    }

    for (i = 0; i < loop_ub; i++) {
      rtDW.Path_RES_0_1[i + loop_ub] = rtDW.Path_RES_1_data[(idx + i) +
        Path_RES_1_size_idx_0];
    }

    loop_ub = Path_RES_0_size_idx_1 - 1;
    for (i = 0; i <= loop_ub; i++) {
      for (total_length_tmp = 0; total_length_tmp < iy; total_length_tmp++) {
        rtDW.Path_RES_0_data_k[total_length_tmp + iy * i] =
          rtDW.Path_RES_0_data[xy_ends_POS_size_idx_0 * i + total_length_tmp];
      }
    }

    jmax = ix + jj;
    for (i = 0; i < n; i++) {
      for (total_length_tmp = 0; total_length_tmp < ix; total_length_tmp++) {
        rtDW.Path_RES_data[total_length_tmp + jmax * i] = rtDW.Path_RES_0_1[ix *
          i + total_length_tmp];
      }
    }

    for (i = 0; i < n; i++) {
      for (total_length_tmp = 0; total_length_tmp < jj; total_length_tmp++) {
        rtDW.Path_RES_data[(total_length_tmp + ix) + jmax * i] =
          rtDW.Path_RES_0_data_k[jj * i + total_length_tmp];
      }
    }

    if (150.0 - Length_1 <= rtb_Gain_p) {
      ajj = rtDW.Path_RES_data[149];
      count_1 = rtDW.Path_RES_data[jmax + 149];
      rtb_Gain_p = 15.0;
    } else {
      rtb_Gain_p += Length_1;
      total_length_tmp = (int32_T)rtb_Gain_p;
      ajj = rtDW.Path_RES_data[total_length_tmp - 1];
      count_1 = rtDW.Path_RES_data[(total_length_tmp + jmax) - 1];
      rtb_Gain_p *= 0.1;
    }
  }

  // MATLAB Function: '<S2>/DynamicPathPlanning1' incorporates:
  //   MATLAB Function: '<S2>/DynamicPathPlanning'
  //   MATLAB Function: '<S2>/EndPointDecision1'

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (i = 0; i < loop_ub; i++) {
    x_data[i] = (rtb_Forward_Static_Path_id[rtDW.SFunction_DIMS4 - 1] ==
                 rtDW.Static_Path_0[i]);
  }

  jmax = rtDW.SFunction_DIMS4_h[0] - 1;
  iy = 0;
  for (ix = 0; ix <= jmax; ix++) {
    if (x_data[ix]) {
      iy++;
    }
  }

  ix = 0;
  for (idx = 0; idx <= jmax; idx++) {
    if (x_data[idx]) {
      c_data[ix] = idx + 1;
      ix++;
    }
  }

  for (i = 0; i < iy; i++) {
    Forward_Static_Path_id_0_data[i] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS4_h[0] * 7 + c_data[i]) - 1] * 3.1415926535897931 /
      180.0;
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (i = 0; i < loop_ub; i++) {
    x_data[i] = (rtb_Forward_Static_Path_id[rtDW.SFunction_DIMS4 - 1] ==
                 rtDW.Static_Path_0[i]);
  }

  n = 0;
  for (idx = 0; idx < rtDW.SFunction_DIMS4_h[0]; idx++) {
    if (x_data[idx]) {
      d_data[n] = idx + 1;
      n++;
    }
  }

  loop_ub = rtDW.SFunction_DIMS4_h[0];
  for (i = 0; i < loop_ub; i++) {
    x_data[i] = (rtb_Forward_Static_Path_id[rtDW.SFunction_DIMS4 - 1] ==
                 rtDW.Static_Path_0[i]);
  }

  ix = 0;
  for (n = 0; n < rtDW.SFunction_DIMS4_h[0]; n++) {
    if (x_data[n]) {
      e_data[ix] = n + 1;
      ix++;
    }
  }

  delta_offset = rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 10 + e_data[0])
    - 1] / 4.0;
  offset_2 = delta_offset * 2.0;
  offset_3 = delta_offset * 3.0;
  offset_4 = delta_offset * 4.0;
  offset_5 = delta_offset * 5.0;
  offset_6 = delta_offset * 6.0;
  offset[0] = offset_6;
  offset[1] = offset_5;
  offset[2] = offset_4;
  offset[3] = offset_3;
  offset[4] = offset_2;
  offset[5] = delta_offset;
  offset[6] = 0.0;
  offset[7] = delta_offset;
  offset[8] = offset_2;
  offset[9] = offset_3;
  offset[10] = offset_4;
  offset[11] = offset_5;
  offset[12] = offset_6;
  if (colj + 1 < 8) {
    ajj += std::cos(Forward_Static_Path_id_0_data[0] + 1.5707963267948966) *
      offset[colj];
    count_1 += std::sin(Forward_Static_Path_id_0_data[0] + 1.5707963267948966) *
      offset[colj];
  } else {
    ajj += std::cos(Forward_Static_Path_id_0_data[0] - 1.5707963267948966) *
      offset[colj];
    count_1 += std::sin(Forward_Static_Path_id_0_data[0] - 1.5707963267948966) *
      offset[colj];
  }

  // Outport: '<Root>/YP_final_1' incorporates:
  //   MATLAB Function: '<S2>/DynamicPathPlanning'
  //   MATLAB Function: '<S2>/DynamicPathPlanning1'
  //   MATLAB Function: '<S2>/EndPointDecision1'
  //   Outport: '<Root>/XP_final_1'

  G2splines_e(Length_1_0[colj << 1], Length_1_0[1 + (colj << 1)], dist_op_data[0]
              / 180.0, rtDW.Static_Path_0[(q_data[0] + rtDW.SFunction_DIMS4_h[0]
    * 13) - 1], ajj, count_1, Forward_Static_Path_id_0_data[0],
              rtDW.Static_Path_0[(d_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) -
              1], rtb_Gain_p, x, b_Path_dis_data, rtY.XP_final_1, rtY.YP_final_1,
              XY_difflen_0, X_NV, &sigma);
  for (i = 0; i < 6; i++) {
    // Outport: '<Root>/XP_final'
    rtY.XP_final[i] = rtb_XP_final_g[i];

    // Outport: '<Root>/YP_final'
    rtY.YP_final[i] = rtb_YP_final_o[i];
  }

  // Outport: '<Root>/X_UKF_SLAM'
  for (i = 0; i < 5; i++) {
    rtY.X_UKF_SLAM[i] = rtb_X[i];
  }

  // End of Outport: '<Root>/X_UKF_SLAM'

  // SignalConversion: '<S21>/TmpSignal ConversionAt SFunction Inport1' incorporates:
  //   Gain: '<S2>/Gain2'
  //   MATLAB Function: '<S2>/MATLAB Function'

  x_0 = 0.017453292519943295 * rtb_Gain1;

  // MATLAB Function: '<S2>/MATLAB Function' incorporates:
  //   Inport: '<Root>/Look_ahead_time'
  //   Inport: '<Root>/Path_flag'
  //   Inport: '<Root>/Speed_mps'
  //   MATLAB Function: '<S2>/DynamicPathPlanning'
  //   SignalConversion: '<S21>/TmpSignal ConversionAt SFunction Inport1'

  if ((rtU.Path_flag == 1.0) && (rtU.Speed_mps < 1.5)) {
    rtb_Gain1 = 1.0;
  } else {
    rtb_Gain1 = (rtU.Speed_mps * rtU.Look_ahead_time + 3.0) / L_path[colj];
  }

  ajj = rtb_Gain1 * rtb_Gain1;
  count_1 = (((((rtb_XP_final_g[1] * rtb_Gain1 + rtb_XP_final_g[0]) + ajj *
                rtb_XP_final_g[2]) + rtb_XP_final_g[3] * rt_powd_snf(rtb_Gain1,
    3.0)) + rtb_XP_final_g[4] * rt_powd_snf(rtb_Gain1, 4.0)) + rtb_XP_final_g[5]
             * rt_powd_snf(rtb_Gain1, 5.0)) - rtb_X[0];
  rtb_Gain1 = (((((rtb_YP_final_o[1] * rtb_Gain1 + rtb_YP_final_o[0]) + ajj *
                  rtb_YP_final_o[2]) + rtb_YP_final_o[3] * rt_powd_snf(rtb_Gain1,
    3.0)) + rtb_YP_final_o[4] * rt_powd_snf(rtb_Gain1, 4.0)) + rtb_YP_final_o[5]
               * rt_powd_snf(rtb_Gain1, 5.0)) - rtb_X[1];
  ajj = std::sin(-x_0);
  rtb_Gain_p = std::cos(-x_0);

  // Outport: '<Root>/Vehicle_Target_x' incorporates:
  //   MATLAB Function: '<S2>/MATLAB Function'

  rtY.Vehicle_Target_x = rtb_Gain_p * count_1 + -ajj * rtb_Gain1;

  // Outport: '<Root>/Vehicle_Target_y' incorporates:
  //   MATLAB Function: '<S2>/MATLAB Function'

  rtY.Vehicle_Target_y = ajj * count_1 + rtb_Gain_p * rtb_Gain1;

  // Outport: '<Root>/J_minind' incorporates:
  //   MATLAB Function: '<S2>/DynamicPathPlanning'

  rtY.J_minind = b_idx;

  // Outport: '<Root>/J_finalind' incorporates:
  //   MATLAB Function: '<S2>/DynamicPathPlanning'

  rtY.J_finalind = colj + 1;

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

  // Update for UnitDelay: '<S2>/Unit Delay1' incorporates:
  //   MATLAB Function: '<S2>/DangerousArea'

  rtDW.UnitDelay1_DSTATE_e = c;

  // Update for UnitDelay: '<S2>/Unit Delay8' incorporates:
  //   MATLAB Function: '<S2>/DangerousArea'

  rtDW.UnitDelay8_DSTATE = c;

  // Update for UnitDelay: '<S2>/Unit Delay12' incorporates:
  //   MATLAB Function: '<S2>/DangerousArea'

  rtDW.UnitDelay12_DSTATE = diffheading;

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

  // MATLAB Function: '<S2>/DynamicPathPlanning'
  for (i = 0; i < 11; i++) {
    b_idx = 11 * colj + i;

    // Update for UnitDelay: '<S2>/Unit Delay5'
    rtDW.UnitDelay5_DSTATE[i] = X_2[b_idx];
    rtDW.UnitDelay5_DSTATE[11 + i] = Y[b_idx];
  }
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
