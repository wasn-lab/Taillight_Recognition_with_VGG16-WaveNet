//
// File: MM_TP.cpp
//
// Code generated for Simulink model 'MM_TP'.
//
// Model version                  : 1.370
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Thu Oct 24 16:28:00 2019
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
void MM_DPP_1ModelClass::merge(int32_T idx[113], real_T x[113], int32_T offset,
  int32_T np, int32_T nq, int32_T iwork[113], real_T xwork[113])
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
void MM_DPP_1ModelClass::sort(real_T x[113], int32_T idx[113])
{
  int32_T nNaNs;
  real_T xwork[113];
  int32_T iwork[113];
  real_T x4[4];
  int8_T idx4[4];
  int32_T ib;
  int32_T m;
  int8_T perm[4];
  int32_T i1;
  int32_T i2;
  int32_T i3;
  int32_T i4;
  x4[0] = 0.0;
  idx4[0] = 0;
  x4[1] = 0.0;
  idx4[1] = 0;
  x4[2] = 0.0;
  idx4[2] = 0;
  x4[3] = 0.0;
  idx4[3] = 0;
  memset(&idx[0], 0, 113U * sizeof(int32_T));
  memset(&xwork[0], 0, 113U * sizeof(real_T));
  nNaNs = 0;
  ib = 0;
  for (m = 0; m < 113; m++) {
    if (rtIsNaN(x[m])) {
      idx[112 - nNaNs] = m + 1;
      xwork[112 - nNaNs] = x[m];
      nNaNs++;
    } else {
      ib++;
      idx4[ib - 1] = (int8_T)(m + 1);
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

    for (m = 113; m - 112 <= ib; m++) {
      i1 = perm[m - 113] - 1;
      i2 = (m - nNaNs) - ib;
      idx[i2] = idx4[i1];
      x[i2] = x4[i1];
    }
  }

  m = nNaNs >> 1;
  for (ib = 1; ib <= m; ib++) {
    i2 = (ib - nNaNs) + 112;
    i1 = idx[i2];
    idx[i2] = idx[113 - ib];
    idx[113 - ib] = i1;
    x[i2] = xwork[113 - ib];
    x[113 - ib] = xwork[i2];
  }

  if ((nNaNs & 1U) != 0U) {
    x[(m - nNaNs) + 113] = xwork[(m - nNaNs) + 113];
  }

  if (113 - nNaNs > 1) {
    memset(&iwork[0], 0, 113U * sizeof(int32_T));
    ib = (113 - nNaNs) >> 2;
    m = 4;
    while (ib > 1) {
      if ((ib & 1U) != 0U) {
        ib--;
        i1 = m * ib;
        i2 = 113 - (nNaNs + i1);
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

    if (113 - nNaNs > m) {
      merge(idx, x, 0, m, 113 - (nNaNs + m), iwork, xwork);
    }
  }
}

// Function for MATLAB Function: '<S3>/SLAM_UKF_MM'
void MM_DPP_1ModelClass::rel_dist_xy(const real_T ref_xy[2], const real_T pt_xy
  [226], real_T dist[113])
{
  int32_T k;
  real_T a;
  real_T temp;
  for (k = 0; k < 113; k++) {
    a = pt_xy[k] - ref_xy[0];
    temp = a * a;
    a = pt_xy[113 + k] - ref_xy[1];
    temp += a * a;
    dist[k] = std::sqrt(temp);
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
  oi_xy[226], const real_T dist_op[113], const real_T Map_data[2599], real_T
  *seg_id_near, real_T *op_distance, real_T oi_near[2], real_T *note, real_T
  *seg_direction, real_T *head_err, real_T num_lane_direction[4], real_T
  *seg_heading)
{
  real_T op_distance_n;
  real_T C;
  int32_T b_index[226];
  real_T SEG_GPS_HEAD[226];
  real_T dist_ini[113];
  real_T dist_end[113];
  int32_T iidx[113];
  boolean_T x[113];
  int8_T ii_data[113];
  int32_T idx;
  int32_T b_ii;
  boolean_T ex;
  int32_T d_k;
  real_T c_a;
  real_T d_a;
  real_T oi_xy_0[2];
  boolean_T exitg1;
  memcpy(&dist_ini[0], &dist_op[0], 113U * sizeof(real_T));
  sort(dist_ini, iidx);
  for (b_ii = 0; b_ii < 113; b_ii++) {
    b_index[b_ii] = iidx[b_ii];
    b_index[b_ii + 113] = 0;
    SEG_GPS_HEAD[b_ii] = Map_data[b_ii];
    SEG_GPS_HEAD[113 + b_ii] = Map_data[791 + b_ii];
  }

  for (idx = 0; idx < 113; idx++) {
    op_distance_n = Map_data[b_index[idx] + 112] - oi_xy[b_index[idx] - 1];
    C = Map_data[b_index[idx] + 225] - oi_xy[b_index[idx] + 112];
    c_a = Map_data[b_index[idx] + 338] - oi_xy[b_index[idx] - 1];
    d_a = Map_data[b_index[idx] + 451] - oi_xy[b_index[idx] + 112];
    if (std::sqrt(op_distance_n * op_distance_n + C * C) <= Map_data[b_index[idx]
        + 903]) {
      b_index[113 + idx] = (std::sqrt(c_a * c_a + d_a * d_a) <=
                            Map_data[b_index[idx] + 903]);
    } else {
      b_index[113 + idx] = 0;
    }

    x[idx] = (b_index[113 + idx] == 1);
  }

  idx = 0;
  b_ii = 1;
  exitg1 = false;
  while ((!exitg1) && (b_ii < 114)) {
    if (x[b_ii - 1]) {
      idx++;
      ii_data[idx - 1] = (int8_T)b_ii;
      if (idx >= 113) {
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
    rel_dist_xy(X_pos, &Map_data[113], dist_ini);
    rel_dist_xy(X_pos, &Map_data[339], dist_end);
    if (!rtIsNaN(dist_ini[0])) {
      idx = 0;
    } else {
      idx = -1;
      b_ii = 2;
      exitg1 = false;
      while ((!exitg1) && (b_ii < 114)) {
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
      for (b_ii = idx + 1; b_ii + 1 < 114; b_ii++) {
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
      while ((!exitg1) && (d_k < 114)) {
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
      for (d_k = b_ii + 1; d_k + 1 < 114; d_k++) {
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
    oi_near[1] = oi_xy[idx + 113];
    if ((idx + 1 > 1) && (idx + 1 < 113)) {
      oi_xy_0[0] = oi_xy[idx];
      oi_xy_0[1] = oi_xy[idx + 113];
      op_distance_n = rel_dist_xy_c(X_pos, oi_xy_0);
      oi_xy_0[0] = Map_data[idx + 113];
      oi_xy_0[1] = Map_data[idx + 226];
      if (op_distance_n < rel_dist_xy_c(X_pos, oi_xy_0)) {
        oi_xy_0[0] = Map_data[idx + 339];
        oi_xy_0[1] = Map_data[idx + 452];
        if (op_distance_n < rel_dist_xy_c(X_pos, oi_xy_0)) {
          *note = 0.0;
        }
      }
    }

    for (b_ii = 0; b_ii < 113; b_ii++) {
      x[b_ii] = (SEG_GPS_HEAD[b_ii] == Map_data[idx]);
    }

    idx = -1;
    ex = x[0];
    for (b_ii = 0; b_ii < 112; b_ii++) {
      if ((int32_T)ex < (int32_T)x[b_ii + 1]) {
        ex = x[b_ii + 1];
        idx = b_ii;
      }
    }

    *head_err = std::abs(heading - SEG_GPS_HEAD[idx + 114]);
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
    oi_near[1] = oi_xy[b_index[ii_data[0] - 1] + 112];
    for (b_ii = 0; b_ii < 113; b_ii++) {
      x[b_ii] = (Map_data[b_index[ii_data[0] - 1] - 1] == SEG_GPS_HEAD[b_ii]);
    }

    idx = -1;
    ex = x[0];
    for (b_ii = 0; b_ii < 112; b_ii++) {
      if ((int32_T)ex < (int32_T)x[b_ii + 1]) {
        ex = x[b_ii + 1];
        idx = b_ii;
      }
    }

    *head_err = std::abs(heading - SEG_GPS_HEAD[idx + 114]);
    if (*head_err <= 0.78539816339744828) {
      *seg_direction = 1.0;
    } else if ((*head_err >= 0.78539816339744828) && (*head_err <= 90.0)) {
      *seg_direction = 0.0;
    } else {
      *seg_direction = 2.0;
    }

    rel_dist_xy(X_pos, &Map_data[113], dist_ini);
    rel_dist_xy(X_pos, &Map_data[339], dist_end);
    if (!rtIsNaN(dist_ini[0])) {
      idx = 0;
    } else {
      idx = -1;
      b_ii = 2;
      exitg1 = false;
      while ((!exitg1) && (b_ii < 114)) {
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
      for (b_ii = idx + 1; b_ii + 1 < 114; b_ii++) {
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
      while ((!exitg1) && (d_k < 114)) {
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
      for (d_k = b_ii + 1; d_k + 1 < 114; d_k++) {
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

      for (b_ii = 0; b_ii < 113; b_ii++) {
        x[b_ii] = (Map_data[b_index[ii_data[0] - 1] - 1] == SEG_GPS_HEAD[b_ii]);
      }

      b_ii = -1;
      ex = x[0];
      for (d_k = 0; d_k < 112; d_k++) {
        if ((int32_T)ex < (int32_T)x[d_k + 1]) {
          ex = x[d_k + 1];
          b_ii = d_k;
        }
      }

      *head_err = std::abs(heading - SEG_GPS_HEAD[b_ii + 114]);
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
      oi_near[1] = oi_xy[idx + 113];
    }
  }

  for (b_ii = 0; b_ii < 113; b_ii++) {
    dist_ini[b_ii] = Map_data[791 + b_ii] * 3.1415926535897931 / 180.0;
    x[b_ii] = (Map_data[b_ii] == *seg_id_near);
  }

  idx = 0;
  ex = x[0];
  for (b_ii = 0; b_ii < 112; b_ii++) {
    if ((int32_T)ex < (int32_T)x[b_ii + 1]) {
      ex = x[b_ii + 1];
      idx = b_ii + 1;
    }
  }

  op_distance_n = oi_near[1] - Map_data[565 + idx] * oi_near[0];
  if (Map_data[565 + idx] < 0.0) {
    C = (-Map_data[565 + idx] * X_pos[0] - op_distance_n) + X_pos[1];
    if (dist_ini[idx] > 4.71238898038469) {
      if (!(dist_ini[idx] < 6.2831853071795862)) {
        C = -C;
      }
    } else {
      C = -C;
    }
  } else if (Map_data[565 + idx] == 0.0) {
    if (oi_near[1] < X_pos[1]) {
      C = -1.0;
    } else {
      C = 1.0;
    }
  } else {
    C = (Map_data[565 + idx] * X_pos[0] + op_distance_n) - X_pos[1];
    if (dist_ini[idx] > 3.1415926535897931) {
      if (!(dist_ini[idx] < 4.71238898038469)) {
        C = -C;
      }
    } else {
      C = -C;
    }
  }

  num_lane_direction[0] = Map_data[565 + idx];
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
  real_T c_x_data[113];
  real_T xwork_data[113];
  int32_T iwork_data[113];
  int32_T n;
  real_T x4[4];
  int8_T idx4[4];
  int32_T ib;
  int32_T wOffset;
  int32_T itmp;
  int8_T perm[4];
  int32_T i1;
  int32_T i3;
  int32_T i4;
  int32_T c_x_size;
  int8_T b_x_idx_0;
  int8_T b_idx_0;
  b_x_idx_0 = (int8_T)*x_size;
  b_idx_0 = (int8_T)*x_size;
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
    idx4[0] = 0;
    x4[1] = 0.0;
    idx4[1] = 0;
    x4[2] = 0.0;
    idx4[2] = 0;
    x4[3] = 0.0;
    idx4[3] = 0;
    b_idx_0 = (int8_T)*x_size;
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
        idx4[ib - 1] = (int8_T)(wOffset + 1);
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
  real_T vwork_data[113];
  int32_T vstride;
  int32_T iidx_data[113];
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

  vwork_size = (int8_T)b;
  *idx_size = (int8_T)*x_size;
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
  real_T z1_data[113];
  int32_T loop_ub;
  int8_T a_idx_0;
  a_idx_0 = (int8_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&z1_data[0], &y_data[0], a_idx_0 * sizeof(real_T));
  }

  for (loop_ub = 0; loop_ub < a_idx_0; loop_ub++) {
    z1_data[loop_ub] = a_data[loop_ub] * a_data[loop_ub];
  }

  *y_size = (int8_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&y_data[0], &z1_data[0], a_idx_0 * sizeof(real_T));
  }
}

// Function for MATLAB Function: '<S2>/MM'
void MM_DPP_1ModelClass::power_lz(const real_T a_data[], const int32_T *a_size,
  real_T y_data[], int32_T *y_size)
{
  real_T z1_data[113];
  int32_T loop_ub;
  int8_T a_idx_0;
  a_idx_0 = (int8_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&z1_data[0], &y_data[0], a_idx_0 * sizeof(real_T));
  }

  for (loop_ub = 0; loop_ub < a_idx_0; loop_ub++) {
    z1_data[loop_ub] = std::sqrt(a_data[loop_ub]);
  }

  *y_size = (int8_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&y_data[0], &z1_data[0], a_idx_0 * sizeof(real_T));
  }
}

// Function for MATLAB Function: '<S2>/MM'
void MM_DPP_1ModelClass::rel_dist_xy_d(const real_T ref_xy[2], const real_T
  pt_xy_data[], const int32_T pt_xy_size[2], real_T dist_data[], int32_T
  *dist_size)
{
  real_T pt_xy_data_0[113];
  real_T tmp_data[113];
  real_T tmp_data_0[113];
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
  real_T xy_ini_data[226];
  real_T xy_end_data[226];
  real_T seg_id_data[113];
  real_T ind_temp_data[113];
  real_T op_distance_n;
  real_T C;
  int32_T b_index_data[226];
  real_T SEG_GPS_HEAD_data[226];
  real_T dist_ini_data[113];
  real_T dist_end_data[113];
  boolean_T x_data[113];
  int32_T ii_data[113];
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
void MM_DPP_1ModelClass::power_j(const real_T a_data[], const int32_T *a_size,
  real_T y_data[], int32_T *y_size)
{
  int32_T loop_ub;
  int16_T a_idx_0;
  a_idx_0 = (int16_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&rtDW.z1_data_k[0], &y_data[0], a_idx_0 * sizeof(real_T));
  }

  for (loop_ub = 0; loop_ub < a_idx_0; loop_ub++) {
    rtDW.z1_data_k[loop_ub] = a_data[loop_ub] * a_data[loop_ub];
  }

  *y_size = (int16_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&y_data[0], &rtDW.z1_data_k[0], a_idx_0 * sizeof(real_T));
  }
}

// Function for MATLAB Function: '<S2>/EndPointDecision2'
void MM_DPP_1ModelClass::power_k(const real_T a_data[], const int32_T *a_size,
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

// Function for MATLAB Function: '<S2>/DangerousArea1'
void MM_DPP_1ModelClass::abs_i(const real_T x[4], real_T y[4])
{
  y[0] = std::abs(x[0]);
  y[1] = std::abs(x[1]);
  y[2] = std::abs(x[2]);
  y[3] = std::abs(x[3]);
}

// Function for MATLAB Function: '<S2>/DangerousArea1'
void MM_DPP_1ModelClass::power(const real_T a[4], real_T y[4])
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
void MM_DPP_1ModelClass::point2safetylevel(const real_T X_data[], const int32_T
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

// Function for MATLAB Function: '<S2>/DynamicPathPlanning'
void MM_DPP_1ModelClass::FreespaceDetectCollision(const real_T Freespace[37500],
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

// Function for MATLAB Function: '<S2>/DynamicPathPlanning1'
void MM_DPP_1ModelClass::point2safetylevel_b(const real_T X_data[], const
  int32_T X_size[2], const real_T Y_data[], const int32_T Y_size[2], const
  real_T Freespace[37500], real_T X_grid_data[], int32_T X_grid_size[2], real_T
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

// Function for MATLAB Function: '<S2>/DynamicPathPlanning1'
void MM_DPP_1ModelClass::FreespaceDetectCollision_b(const real_T Freespace[37500],
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

      point2safetylevel_b(Path_vehFLX_j_data, Path_vehFLX_j_size_0,
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

      point2safetylevel_b(Path_vehFLX_j_data, Path_vehRLX_j_size,
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

      point2safetylevel_b(Path_vehFLX_j_data, Path_vehFLX_j_size,
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

      point2safetylevel_b(Path_vehFLX_j_data, Path_vehFRX_j_size,
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
  static const real_T b[5] = { 0.00025, 0.00025, 0.00025, 1.0E-5, 0.0001 };

  real_T p0[25];
  real_T p_sqrt_data[25];
  real_T temp_dia[5];
  int32_T jj;
  real_T ajj;
  int32_T iy;
  int32_T b_ix;
  int8_T ii_data[5];
  int32_T c_ix;
  int32_T d_ix;
  real_T K1[4];
  real_T x[11];
  int8_T I[25];
  real_T b_j;
  real_T seg_direction;
  real_T head_err;
  real_T seg_heading;
  real_T oi_xy_data[226];
  real_T dist_op_data[113];
  real_T forward_length;
  real_T xy_ends_POS_data[452];
  real_T total_length;
  real_T Forward_Static_Path_id_0_data[113];
  int16_T cb_data[113];
  real_T Forward_Static_Path_data[454];
  real_T count;
  real_T count_1;
  real_T target_k;
  real_T Length_1;
  real_T ang_1;
  real_T Forward_Static_Path_data_0[454];
  real_T End_x;
  real_T y;
  real_T OBXY_m[8];
  real_T c;
  real_T delta_offset;
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
  int8_T p_data[9];
  int32_T t_data[113];
  int32_T u_data[113];
  int32_T v_data[113];
  real_T end_heading_0_data[113];
  int32_T t_data_0[113];
  int32_T u_data_0[113];
  int32_T v_data_0[113];
  static const real_T a[11] = { 0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0 };

  real_T table[226];
  real_T shortest_distance[113];
  int8_T settled[113];
  int8_T pidx_data[113];
  int8_T zz_data[113];
  real_T tmp_path_data[113];
  int8_T nidx_data[113];
  int8_T c_data[113];
  int32_T ii_data_0[113];
  int32_T idx;
  int32_T b_idx;
  int32_T n;
  boolean_T x_data[113];
  boolean_T b_x[113];
  int8_T f_ii_data[113];
  real_T rtb_Add;
  real_T rtb_Gain1;
  real_T rtb_Gain_p;
  real_T rtb_X[5];
  real_T rtb_TmpSignalConversionAtSFun_e[3];
  real_T rtb_UnitDelay34[5];
  real_T rtb_UnitDelay18;
  real_T rtb_X_state[5];
  real_T rtb_Oi_near_l[2];
  real_T rtb_num_lane_direction_f[4];
  real_T rtb_H_y_out[4];
  real_T rtb_J_out_k[13];
  real_T rtb_U_c_l[13];
  real_T rtb_safety_level_all_p[13];
  real_T rtb_forward_length_free_o[13];
  real_T rtb_forward_length_free[13];
  real_T rtb_forward_length_free_2[13];
  real_T rtb_Q_last_o[25];
  real_T rtb_R_last_o[25];
  real_T rtb_X_AUG[55];
  real_T rtb_K[25];
  real_T rtb_V_boundingbox[400];
  real_T rtb_Forward_Static_Path_id_l[113];
  real_T rtb_Forward_Static_Path_x_h[113];
  real_T rtb_Forward_Static_Path_y_p[113];
  real_T rtb_Forward_Static_Path_id_i[113];
  real_T rtb_XP_i[78];
  real_T rtb_YP_g[78];
  real_T rtb_XP[78];
  real_T rtb_YP[78];
  int32_T i;
  real_T tmp[4];
  real_T ex[4];
  real_T rtb_X_AUG_0[10];
  real_T LastPath_overlap_data_0[22];
  int32_T ix;
  real_T p_sqrt_data_0[25];
  real_T rtb_X_state_0[2];
  real_T rtb_num_lane_direction_b[4];
  int16_T tmp_0;
  real_T rtb_Oi_near_o[8];
  real_T rtb_Oi_near_o_0[8];
  real_T OBXY_EL_0[8];
  real_T b_Path_dis_data_0[121];
  int32_T count_0;
  int32_T count_2;
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
  real_T yy_idx_0;
  int32_T Path_RES_1_size_idx_0;
  int32_T xy_ends_POS_size_idx_0;
  real_T xy_end_point_idx_25;
  int32_T Path_RES_0_size_idx_1;
  real_T seg_direction_tmp_tmp;
  real_T target_k_tmp;
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

  rtb_Add = std::abs(1.0 - rtDW.Memory1_PreviousInput) + std::abs(113.0 -
    rtDW.Memory_PreviousInput);

  // Outputs for Enabled SubSystem: '<S2>/Enabled Subsystem' incorporates:
  //   EnablePort: '<S13>/Enable'

  if (rtb_Add > 0.0) {
    // MATLAB Function: '<S13>/Dijkstra' incorporates:
    //   Constant: '<S2>/Constant7'

    memset(&table[0], 0, 226U * sizeof(real_T));
    for (i = 0; i < 113; i++) {
      shortest_distance[i] = (rtInf);
      settled[i] = 0;
    }

    memset(&rtDW.path[0], 0, 12769U * sizeof(real_T));
    idx = 0;
    i = 1;
    exitg1 = false;
    while ((!exitg1) && (i < 114)) {
      if (rtConstP.Constant3_Value[i - 1] == 1.0) {
        idx++;
        ii_data_0[idx - 1] = i;
        if (idx >= 113) {
          exitg1 = true;
        } else {
          i++;
        }
      } else {
        i++;
      }
    }

    if (1 > idx) {
      idx = 0;
    }

    for (b_ix = 0; b_ix < idx; b_ix++) {
      pidx_data[b_ix] = (int8_T)ii_data_0[b_ix];
    }

    shortest_distance[ii_data_0[0] - 1] = 0.0;
    table[ii_data_0[0] + 112] = 0.0;
    settled[ii_data_0[0] - 1] = 1;
    rtDW.path[ii_data_0[0] - 1] = 1.0;
    b_idx = 0;
    idx = 1;
    exitg1 = false;
    while ((!exitg1) && (idx < 114)) {
      if (rtConstP.Constant3_Value[idx - 1] == 113.0) {
        b_idx++;
        ii_data_0[b_idx - 1] = idx;
        if (b_idx >= 113) {
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

    for (b_ix = 0; b_ix < idx; b_ix++) {
      zz_data[b_ix] = (int8_T)ii_data_0[b_ix];
    }

    do {
      exitg2 = 0;
      b_ix = zz_data[0] - 1;
      if (settled[b_ix] == 0) {
        for (iy = 0; iy < 113; iy++) {
          table[iy] = table[113 + iy];
        }

        n = pidx_data[0] + 112;
        table[n] = 0.0;
        b_idx = 0;
        for (ix = 0; ix < 113; ix++) {
          b_x_0 = (rtConstP.Constant3_Value[pidx_data[0] - 1] ==
                   rtConstP.Constant5_Value[113 + ix]);
          if (b_x_0) {
            b_idx++;
          }

          b_x[ix] = b_x_0;
        }

        ix = b_idx;
        b_idx = 0;
        for (idx = 0; idx < 113; idx++) {
          if (b_x[idx]) {
            c_data[b_idx] = (int8_T)(idx + 1);
            b_idx++;
          }
        }

        for (i = 0; i < ix; i++) {
          for (iy = 0; iy < 113; iy++) {
            b_x[iy] = (rtConstP.Constant5_Value[c_data[i] + 225] ==
                       rtConstP.Constant3_Value[iy]);
          }

          b_idx = -1;
          idx = 1;
          exitg1 = false;
          while ((!exitg1) && (idx < 114)) {
            if (b_x[idx - 1]) {
              b_idx++;
              ii_data_0[b_idx] = idx;
              if (b_idx + 1 >= 113) {
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
              rtConstP.Constant3_Value[ii_data_0[0] + 112];
            rtb_X_state_0[1] = rtConstP.Constant3_Value[pidx_data[0] + 225] -
              rtConstP.Constant3_Value[ii_data_0[0] + 225];
            power_n(rtb_X_state_0, rtb_Oi_near_l);
            ajj = std::sqrt(sum_e(rtb_Oi_near_l));
            if ((table[ii_data_0[0] - 1] == 0.0) || (table[ii_data_0[0] - 1] >
                 table[pidx_data[0] - 1] + ajj)) {
              table[ii_data_0[0] + 112] = table[pidx_data[0] - 1] + ajj;
              for (iy = 0; iy < 113; iy++) {
                b_x[iy] = (rtDW.path[(113 * iy + pidx_data[0]) - 1] != 0.0);
              }

              idx = 0;
              b_idx = 1;
              exitg1 = false;
              while ((!exitg1) && (b_idx < 114)) {
                if (b_x[b_idx - 1]) {
                  idx++;
                  f_ii_data[idx - 1] = (int8_T)b_idx;
                  if (idx >= 113) {
                    exitg1 = true;
                  } else {
                    b_idx++;
                  }
                } else {
                  b_idx++;
                }
              }

              if (1 > idx) {
                c_ix = 0;
              } else {
                c_ix = idx;
              }

              d_ix = c_ix - 1;
              if (0 <= d_ix) {
                memset(&tmp_path_data[0], 0, (d_ix + 1) * sizeof(real_T));
              }

              for (b_idx = 0; b_idx < c_ix; b_idx++) {
                tmp_path_data[b_idx] = rtDW.path[((f_ii_data[b_idx] - 1) * 113 +
                  pidx_data[0]) - 1];
              }

              idx = ii_data_0[0] - 1;
              for (iy = 0; iy < c_ix; iy++) {
                rtDW.path[idx + 113 * iy] = tmp_path_data[iy];
              }

              rtDW.path[idx + 113 * c_ix] = rtConstP.Constant5_Value[c_data[i] +
                225];
            } else {
              table[ii_data_0[0] + 112] = table[ii_data_0[0] - 1];
            }
          }
        }

        i = 0;
        idx = 1;
        exitg1 = false;
        while ((!exitg1) && (idx < 114)) {
          if (table[idx + 112] != 0.0) {
            i++;
            ii_data_0[i - 1] = idx;
            if (i >= 113) {
              exitg1 = true;
            } else {
              idx++;
            }
          } else {
            idx++;
          }
        }

        if (1 > i) {
          idx = 0;
        } else {
          idx = i;
        }

        for (iy = 0; iy < idx; iy++) {
          nidx_data[iy] = (int8_T)ii_data_0[iy];
        }

        if (idx <= 2) {
          if (idx == 1) {
            delta_offset = table[ii_data_0[0] + 112];
          } else if (table[ii_data_0[0] + 112] > table[ii_data_0[1] + 112]) {
            delta_offset = table[ii_data_0[1] + 112];
          } else if (rtIsNaN(table[ii_data_0[0] + 112])) {
            if (!rtIsNaN(table[ii_data_0[1] + 112])) {
              delta_offset = table[ii_data_0[1] + 112];
            } else {
              delta_offset = table[ii_data_0[0] + 112];
            }
          } else {
            delta_offset = table[ii_data_0[0] + 112];
          }
        } else {
          if (!rtIsNaN(table[ii_data_0[0] + 112])) {
            b_idx = 1;
          } else {
            b_idx = 0;
            i = 2;
            exitg1 = false;
            while ((!exitg1) && (i <= idx)) {
              if (!rtIsNaN(table[ii_data_0[i - 1] + 112])) {
                b_idx = i;
                exitg1 = true;
              } else {
                i++;
              }
            }
          }

          if (b_idx == 0) {
            delta_offset = table[ii_data_0[0] + 112];
          } else {
            delta_offset = table[ii_data_0[b_idx - 1] + 112];
            while (b_idx + 1 <= idx) {
              if (delta_offset > table[ii_data_0[b_idx] + 112]) {
                delta_offset = table[ii_data_0[b_idx] + 112];
              }

              b_idx++;
            }
          }
        }

        for (iy = 0; iy < idx; iy++) {
          x_data[iy] = (table[ii_data_0[iy] + 112] == delta_offset);
        }

        b_idx = 0;
        i = 1;
        exitg1 = false;
        while ((!exitg1) && (i <= idx)) {
          if (x_data[i - 1]) {
            b_idx++;
            ii_data_0[b_idx - 1] = i;
            if (b_idx >= idx) {
              exitg1 = true;
            } else {
              i++;
            }
          } else {
            i++;
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
          shortest_distance[i] = table[nidx_data[ii_data_0[0] - 1] + 112];
          settled[i] = 1;
        }
      } else {
        exitg2 = 1;
      }
    } while (exitg2 == 0);

    for (iy = 0; iy < 113; iy++) {
      b_x[iy] = (rtDW.path[(113 * iy + zz_data[0]) - 1] != 0.0);
    }

    i = 0;
    idx = 0;
    exitg1 = false;
    while ((!exitg1) && (idx + 1 < 114)) {
      if (b_x[idx]) {
        i++;
        if (i >= 113) {
          exitg1 = true;
        } else {
          idx++;
        }
      } else {
        idx++;
      }
    }

    if (1 > i) {
      c_ix = 0;
    } else {
      c_ix = i;
    }

    if (1 > c_ix) {
      c_ix = 0;
    }

    rtDW.dist = shortest_distance[b_ix];
    rtDW.SFunction_DIMS3_c = c_ix;
    for (b_ix = 0; b_ix < c_ix; b_ix++) {
      rtDW.path_2[b_ix] = rtDW.path[(113 * b_ix + zz_data[0]) - 1];
    }

    // End of MATLAB Function: '<S13>/Dijkstra'
  }

  // End of Outputs for SubSystem: '<S2>/Enabled Subsystem'

  // MATLAB Function: '<S2>/Final_Static_Path' incorporates:
  //   Constant: '<S2>/Constant6'

  if (!rtDW.path_out1_not_empty) {
    if (rtb_Add > 0.0) {
      rtDW.path_out1.size = rtDW.SFunction_DIMS3_c;
      for (b_ix = 0; b_ix < rtDW.SFunction_DIMS3_c; b_ix++) {
        rtDW.path_out1.data[b_ix] = rtDW.path_2[b_ix];
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
    for (b_ix = 0; b_ix < rtDW.SFunction_DIMS3_c; b_ix++) {
      rtDW.path_out1.data[b_ix] = rtDW.path_2[b_ix];
    }

    rtDW.path_out1_not_empty = !(rtDW.path_out1.size == 0);
  }

  rtDW.SFunction_DIMS2_m = rtDW.path_out1.size;
  rtDW.SFunction_DIMS3_l = rtDW.path_out1.size;
  rtDW.SFunction_DIMS4_h[0] = 113;
  rtDW.SFunction_DIMS4_h[1] = 23;
  memcpy(&rtDW.Static_Path_0[0], &rtConstP.pooled2[0], 2599U * sizeof(real_T));
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
    for (c_ix = 0; c_ix < 5; c_ix++) {
      rtb_Q_last_o[c_ix + 5 * c_ix] = b[c_ix];
    }

    rtb_X_state[0] = 0.0001;
    rtb_X_state[1] = 0.0001;
    if (std::abs(rtU.SLAM_heading - rtDW.UnitDelay1_DSTATE[2]) > 5.5) {
      rtb_X_state[2] = 100.0;
    } else {
      rtb_X_state[2] = 0.0001;
    }

    rtb_X_state[3] = 0.001;
    rtb_X_state[4] = 0.0001;
    memset(&rtb_R_last_o[0], 0, 25U * sizeof(real_T));
    for (c_ix = 0; c_ix < 5; c_ix++) {
      rtb_R_last_o[c_ix + 5 * c_ix] = rtb_X_state[c_ix];
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
  n = 0;
  c_ix = 1;
  exitg1 = false;
  while ((!exitg1) && (c_ix <= 5)) {
    jj = (n + c_ix) - 1;
    ajj = 0.0;
    if (!(c_ix - 1 < 1)) {
      ix = n;
      iy = n;
      for (i = 1; i < c_ix; i++) {
        ajj += p_sqrt_data[ix] * p_sqrt_data[iy];
        ix++;
        iy++;
      }
    }

    ajj = p_sqrt_data[jj] - ajj;
    if (ajj > 0.0) {
      ajj = std::sqrt(ajj);
      p_sqrt_data[jj] = ajj;
      if (c_ix < 5) {
        if (c_ix - 1 != 0) {
          i = jj + 5;
          ix = ((4 - c_ix) * 5 + n) + 6;
          for (iy = n + 6; iy <= ix; iy += 5) {
            b_ix = n;
            y_endpoint2 = 0.0;
            d_ix = (iy + c_ix) - 2;
            for (xy_ends_POS_size_idx_0 = iy; xy_ends_POS_size_idx_0 <= d_ix;
                 xy_ends_POS_size_idx_0++) {
              y_endpoint2 += p_sqrt_data[xy_ends_POS_size_idx_0 - 1] *
                p_sqrt_data[b_ix];
              b_ix++;
            }

            p_sqrt_data[i] += -y_endpoint2;
            i += 5;
          }
        }

        ajj = 1.0 / ajj;
        ix = ((4 - c_ix) * 5 + jj) + 6;
        for (i = jj + 5; i + 1 <= ix; i += 5) {
          p_sqrt_data[i] *= ajj;
        }

        n += 5;
      }

      c_ix++;
    } else {
      p_sqrt_data[jj] = ajj;
      idx = c_ix;
      exitg1 = true;
    }
  }

  if (idx == 0) {
    i = 5;
  } else {
    i = idx - 1;
  }

  for (c_ix = 1; c_ix <= i; c_ix++) {
    for (ix = c_ix; ix < i; ix++) {
      p_sqrt_data[ix + 5 * (c_ix - 1)] = 0.0;
    }
  }

  if (1 > i) {
    i = 0;
    d_ix = 0;
  } else {
    d_ix = i;
  }

  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    for (iy = 0; iy < i; iy++) {
      p_sqrt_data_0[iy + i * b_ix] = p_sqrt_data[5 * b_ix + iy];
    }
  }

  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    for (iy = 0; iy < i; iy++) {
      c_ix = i * b_ix;
      p_sqrt_data[iy + c_ix] = p_sqrt_data_0[c_ix + iy];
    }
  }

  memset(&rtb_X_AUG[0], 0, 55U * sizeof(real_T));
  if (idx != 0) {
    for (i = 0; i < 5; i++) {
      temp_dia[i] = std::abs(rtDW.UnitDelay33_DSTATE[5 * i + i]);
    }

    idx = 0;
    i = 1;
    exitg1 = false;
    while ((!exitg1) && (i < 6)) {
      if (temp_dia[i - 1] < 1.0E-10) {
        idx++;
        ii_data[idx - 1] = (int8_T)i;
        if (idx >= 5) {
          exitg1 = true;
        } else {
          i++;
        }
      } else {
        i++;
      }
    }

    if (!(1 > idx)) {
      for (b_ix = 0; b_ix < idx; b_ix++) {
        temp_dia[ii_data[b_ix] - 1] = 1.0E-10;
      }
    }

    memset(&p0[0], 0, 25U * sizeof(real_T));
    for (idx = 0; idx < 5; idx++) {
      p0[idx + 5 * idx] = temp_dia[idx];
    }

    i = 0;
    ix = 0;
    idx = 1;
    exitg1 = false;
    while ((!exitg1) && (idx < 6)) {
      jj = (ix + idx) - 1;
      ajj = 0.0;
      if (!(idx - 1 < 1)) {
        c_ix = ix;
        n = ix;
        for (iy = 1; iy < idx; iy++) {
          ajj += p0[c_ix] * p0[n];
          c_ix++;
          n++;
        }
      }

      ajj = p0[jj] - ajj;
      if (ajj > 0.0) {
        ajj = std::sqrt(ajj);
        p0[jj] = ajj;
        if (idx < 5) {
          if (idx - 1 != 0) {
            iy = jj + 5;
            c_ix = ((4 - idx) * 5 + ix) + 6;
            for (b_ix = ix + 6; b_ix <= c_ix; b_ix += 5) {
              d_ix = ix;
              y_endpoint2 = 0.0;
              n = (b_ix + idx) - 2;
              for (xy_ends_POS_size_idx_0 = b_ix; xy_ends_POS_size_idx_0 <= n;
                   xy_ends_POS_size_idx_0++) {
                y_endpoint2 += p0[xy_ends_POS_size_idx_0 - 1] * p0[d_ix];
                d_ix++;
              }

              p0[iy] += -y_endpoint2;
              iy += 5;
            }
          }

          ajj = 1.0 / ajj;
          n = ((4 - idx) * 5 + jj) + 6;
          for (iy = jj + 5; iy + 1 <= n; iy += 5) {
            p0[iy] *= ajj;
          }

          ix += 5;
        }

        idx++;
      } else {
        p0[jj] = ajj;
        i = idx;
        exitg1 = true;
      }
    }

    if (i == 0) {
      i = 5;
    } else {
      i--;
    }

    for (idx = 0; idx < i; idx++) {
      for (c_ix = idx + 1; c_ix < i; c_ix++) {
        p0[c_ix + 5 * idx] = 0.0;
      }
    }

    i = 5;
    d_ix = 5;
    memcpy(&p_sqrt_data[0], &p0[0], 25U * sizeof(real_T));
  }

  for (b_ix = 0; b_ix < i; b_ix++) {
    for (iy = 0; iy < d_ix; iy++) {
      p_sqrt_data_0[iy + d_ix * b_ix] = p_sqrt_data[i * iy + b_ix] *
        2.23606797749979;
    }
  }

  for (b_ix = 0; b_ix < i; b_ix++) {
    for (iy = 0; iy < d_ix; iy++) {
      c_ix = d_ix * b_ix;
      p_sqrt_data[iy + c_ix] = p_sqrt_data_0[c_ix + iy];
    }
  }

  for (b_ix = 0; b_ix < 5; b_ix++) {
    rtb_X_AUG[b_ix] = rtb_UnitDelay34[b_ix];
  }

  for (ix = 0; ix < 5; ix++) {
    jj = d_ix - 1;
    for (b_ix = 0; b_ix <= jj; b_ix++) {
      temp_dia[b_ix] = p_sqrt_data[d_ix * ix + b_ix];
    }

    b_ix = ix + 2;
    for (iy = 0; iy < 5; iy++) {
      rtb_X_AUG[iy + 5 * (b_ix - 1)] = rtb_UnitDelay34[iy] + temp_dia[iy];
    }
  }

  for (idx = 0; idx < 5; idx++) {
    jj = d_ix - 1;
    for (b_ix = 0; b_ix <= jj; b_ix++) {
      temp_dia[b_ix] = p_sqrt_data[d_ix * idx + b_ix];
    }

    b_ix = idx + 7;
    for (iy = 0; iy < 5; iy++) {
      rtb_X_AUG[iy + 5 * (b_ix - 1)] = rtb_UnitDelay34[iy] - temp_dia[iy];
    }
  }

  // End of MATLAB Function: '<S3>/SLAM_Generate_sigma_pt_UKF'

  // MATLAB Function: '<S3>/SLAM_UKF' incorporates:
  //   Constant: '<Root>/[Para] D_GC'
  //   Constant: '<S1>/Constant25'
  //   MATLAB Function: '<S3>/SLAM_Check'
  //   SignalConversion: '<S6>/TmpSignal ConversionAt SFunction Inport5'

  rtb_UnitDelay18 = 0.01 * rtb_Gain_p * 3.8;
  for (i = 0; i < 11; i++) {
    rtb_X_AUG[5 * i] = (rtb_X_AUG[5 * i + 4] * 0.01 * std::cos(rtb_X_AUG[5 * i +
      2]) + rtb_X_AUG[5 * i]) + std::cos(rtb_X_AUG[5 * i + 2] +
      1.5707963267948966) * rtb_UnitDelay18;
    rtb_X_AUG[1 + 5 * i] = (rtb_X_AUG[5 * i + 4] * 0.01 * std::sin(rtb_X_AUG[5 *
      i + 2]) + rtb_X_AUG[5 * i + 1]) + std::sin(rtb_X_AUG[5 * i + 2] +
      1.5707963267948966) * rtb_UnitDelay18;
    rtb_X_AUG[2 + 5 * i] += rtb_X_AUG[5 * i + 3] * 0.01;
  }

  for (b_ix = 0; b_ix < 10; b_ix++) {
    rtb_X_AUG_0[b_ix] = rtb_X_AUG[(1 + b_ix) * 5];
  }

  rtb_X[0] = rtb_X_AUG[0] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (b_ix = 0; b_ix < 10; b_ix++) {
    rtb_X_AUG_0[b_ix] = rtb_X_AUG[(1 + b_ix) * 5 + 1];
  }

  rtb_X[1] = rtb_X_AUG[1] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (b_ix = 0; b_ix < 10; b_ix++) {
    rtb_X_AUG_0[b_ix] = rtb_X_AUG[(1 + b_ix) * 5 + 2];
  }

  rtb_X[2] = rtb_X_AUG[2] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (b_ix = 0; b_ix < 10; b_ix++) {
    rtb_X_AUG_0[b_ix] = rtb_X_AUG[(1 + b_ix) * 5 + 3];
  }

  rtb_X[3] = rtb_X_AUG[3] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (b_ix = 0; b_ix < 10; b_ix++) {
    rtb_X_AUG_0[b_ix] = rtb_X_AUG[(1 + b_ix) * 5 + 4];
  }

  rtb_X[4] = rtb_X_AUG[4] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (b_ix = 0; b_ix < 5; b_ix++) {
    rtb_Gain1 = rtb_X_AUG[b_ix] - rtb_X[b_ix];
    rtb_UnitDelay34[b_ix] = rtb_Gain1;
    temp_dia[b_ix] = rtb_Gain1;
  }

  for (b_ix = 0; b_ix < 5; b_ix++) {
    for (iy = 0; iy < 5; iy++) {
      p_sqrt_data[b_ix + 5 * iy] = rtb_UnitDelay34[b_ix] * temp_dia[iy];
    }
  }

  for (b_ix = 0; b_ix < 5; b_ix++) {
    for (iy = 0; iy < 5; iy++) {
      p0[iy + 5 * b_ix] = p_sqrt_data[5 * b_ix + iy] * 2.0;
    }
  }

  for (ix = 0; ix < 10; ix++) {
    for (b_ix = 0; b_ix < 5; b_ix++) {
      rtb_Gain1 = rtb_X_AUG[(ix + 1) * 5 + b_ix] - rtb_X[b_ix];
      rtb_UnitDelay34[b_ix] = rtb_Gain1;
      temp_dia[b_ix] = rtb_Gain1;
    }

    for (b_ix = 0; b_ix < 5; b_ix++) {
      for (iy = 0; iy < 5; iy++) {
        p_sqrt_data[b_ix + 5 * iy] = rtb_UnitDelay34[b_ix] * temp_dia[iy];
      }
    }

    for (b_ix = 0; b_ix < 5; b_ix++) {
      for (iy = 0; iy < 5; iy++) {
        i = 5 * b_ix + iy;
        p0[iy + 5 * b_ix] = p_sqrt_data[i] * 0.1 + p0[i];
      }
    }
  }

  for (b_ix = 0; b_ix < 25; b_ix++) {
    p0[b_ix] += rtb_Q_last_o[b_ix];
  }

  if (rtb_X[2] < 0.0) {
    rtb_X[2] += 6.2831853071795862;
  } else {
    if (rtb_X[2] >= 6.2831853071795862) {
      rtb_X[2] -= 6.2831853071795862;
    }
  }

  if (b_idx > 0) {
    for (b_ix = 0; b_ix < 25; b_ix++) {
      p_sqrt_data[b_ix] = p0[b_ix] + rtb_R_last_o[b_ix];
    }

    invNxN(p_sqrt_data, p_sqrt_data_0);
    for (b_ix = 0; b_ix < 5; b_ix++) {
      for (iy = 0; iy < 5; iy++) {
        idx = b_ix + 5 * iy;
        rtb_K[idx] = 0.0;
        for (i = 0; i < 5; i++) {
          rtb_K[idx] = p0[5 * i + b_ix] * p_sqrt_data_0[5 * iy + i] + rtb_K[5 *
            iy + b_ix];
        }
      }

      rtb_UnitDelay34[b_ix] = rtb_X_state[b_ix] - rtb_X[b_ix];
    }

    for (b_ix = 0; b_ix < 5; b_ix++) {
      delta_offset = 0.0;
      for (iy = 0; iy < 5; iy++) {
        delta_offset += rtb_K[5 * iy + b_ix] * rtb_UnitDelay34[iy];
      }

      rtb_X[b_ix] += delta_offset;
    }

    for (b_ix = 0; b_ix < 25; b_ix++) {
      I[b_ix] = 0;
    }

    for (i = 0; i < 5; i++) {
      I[i + 5 * i] = 1;
    }

    for (b_ix = 0; b_ix < 5; b_ix++) {
      for (iy = 0; iy < 5; iy++) {
        idx = 5 * b_ix + iy;
        p_sqrt_data[iy + 5 * b_ix] = (real_T)I[idx] - rtb_K[idx];
      }
    }

    for (b_ix = 0; b_ix < 5; b_ix++) {
      for (iy = 0; iy < 5; iy++) {
        idx = iy + 5 * b_ix;
        rtb_K[idx] = 0.0;
        for (i = 0; i < 5; i++) {
          rtb_K[idx] = p_sqrt_data[5 * i + iy] * p0[5 * b_ix + i] + rtb_K[5 *
            b_ix + iy];
        }
      }
    }

    for (b_ix = 0; b_ix < 5; b_ix++) {
      for (iy = 0; iy < 5; iy++) {
        p0[iy + 5 * b_ix] = rtb_K[5 * b_ix + iy];
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

    for (b_ix = 0; b_ix < 2; b_ix++) {
      K1[b_ix] = 0.0;
      K1[b_ix] += p0[b_ix + 18] * rtb_H_y_out[0];
      K1[b_ix] += p0[b_ix + 23] * rtb_H_y_out[1];
      K1[b_ix + 2] = 0.0;
      K1[b_ix + 2] += p0[b_ix + 18] * rtb_H_y_out[2];
      K1[b_ix + 2] += p0[b_ix + 23] * rtb_H_y_out[3];
      rtb_X_state_0[b_ix] = rtb_X_state[3 + b_ix] - rtb_X[3 + b_ix];
    }

    rtb_X[3] += K1[0] * rtb_X_state_0[0] + K1[2] * rtb_X_state_0[1];
    rtb_X[4] += K1[1] * rtb_X_state_0[0] + K1[3] * rtb_X_state_0[1];
    rtb_num_lane_direction_f[0] = 1.0 - K1[0];
    rtb_num_lane_direction_f[1] = 0.0 - K1[1];
    rtb_num_lane_direction_f[2] = 0.0 - K1[2];
    rtb_num_lane_direction_f[3] = 1.0 - K1[3];
    for (b_ix = 0; b_ix < 2; b_ix++) {
      rtb_num_lane_direction_b[b_ix] = 0.0;
      rtb_num_lane_direction_b[b_ix] += rtb_num_lane_direction_f[b_ix] * p0[18];
      rtb_num_lane_direction_b[b_ix] += rtb_num_lane_direction_f[b_ix + 2] * p0
        [19];
      rtb_num_lane_direction_b[b_ix + 2] = 0.0;
      rtb_num_lane_direction_b[b_ix + 2] += rtb_num_lane_direction_f[b_ix] * p0
        [23];
      rtb_num_lane_direction_b[b_ix + 2] += rtb_num_lane_direction_f[b_ix + 2] *
        p0[24];
    }

    p0[18] = rtb_num_lane_direction_b[0];
    p0[19] = rtb_num_lane_direction_b[1];
    p0[23] = rtb_num_lane_direction_b[2];
    p0[24] = rtb_num_lane_direction_b[3];
  }

  // End of MATLAB Function: '<S3>/SLAM_UKF'

  // MATLAB Function: '<S2>/Boundingbox_trans' incorporates:
  //   Inport: '<Root>/BB_all_XY'
  //   Inport: '<Root>/BB_num'
  //   Inport: '<Root>/VirBB_mode'

  rtb_Gain1 = rtb_X[2];
  memcpy(&rtb_V_boundingbox[0], &rtU.BB_all_XY[0], 400U * sizeof(real_T));
  if (rtU.VirBB_mode == 0.0) {
    for (n = 0; n < (int32_T)rtU.BB_num; n++) {
      delta_offset = (1.0 + (real_T)n) * 2.0;
      for (ix = 0; ix < 4; ix++) {
        b_j = std::sin(rtb_Gain1);
        forward_length = std::cos(rtb_Gain1);
        rtb_V_boundingbox[((int32_T)(delta_offset + -1.0) + 100 * ix) - 1] =
          (rtU.BB_all_XY[((int32_T)(delta_offset + -1.0) + 100 * ix) - 1] *
           forward_length + rtU.BB_all_XY[(100 * ix + (int32_T)delta_offset) - 1]
           * -b_j) + rtb_X[0];
        rtb_V_boundingbox[((int32_T)delta_offset + 100 * ix) - 1] =
          (rtU.BB_all_XY[((int32_T)(delta_offset + -1.0) + 100 * ix) - 1] * b_j
           + rtU.BB_all_XY[(100 * ix + (int32_T)delta_offset) - 1] *
           forward_length) + rtb_X[1];
      }
    }
  }

  // End of MATLAB Function: '<S2>/Boundingbox_trans'

  // MATLAB Function: '<S3>/SLAM_UKF_MM' incorporates:
  //   Constant: '<S3>/Constant4'

  for (ix = 0; ix < 113; ix++) {
    if (rtConstP.pooled2[565 + ix] == (rtInf)) {
      table[ix] = rtConstP.pooled2[113 + ix];
      table[113 + ix] = rtb_X[1];
    } else if (rtConstP.pooled2[565 + ix] == 0.0) {
      table[ix] = rtb_X[0];
      table[113 + ix] = rtConstP.pooled2[226 + ix];
    } else {
      rtb_Gain1 = -1.0 / rtConstP.pooled2[565 + ix];
      b_j = rtb_X[1] - rtb_Gain1 * rtb_X[0];
      forward_length = rtConstP.pooled2[565 + ix] - rtb_Gain1;
      table[ix] = (b_j - rtConstP.pooled2[678 + ix]) / forward_length;
      table[113 + ix] = (rtConstP.pooled2[565 + ix] * b_j - rtConstP.pooled2[678
                         + ix] * rtb_Gain1) / forward_length;
    }

    delta_offset = table[ix] - rtb_X[0];
    offset_6 = table[113 + ix] - rtb_X[1];
    shortest_distance[ix] = std::sqrt(delta_offset * delta_offset + offset_6 *
      offset_6);
  }

  rtb_X_state_0[0] = rtb_X[0];
  rtb_X_state_0[1] = rtb_X[1];
  MM(rtb_X[2] * 180.0 / 3.1415926535897931, rtb_X_state_0, table,
     shortest_distance, rtConstP.pooled2, &rtb_Gain1, &b_j, rtb_Oi_near_l,
     &forward_length, &seg_direction, &head_err, rtb_num_lane_direction_f,
     &seg_heading);

  // End of MATLAB Function: '<S3>/SLAM_UKF_MM'

  // Gain: '<S2>/Gain3'
  ajj = 57.295779513082323 * rtb_X[2];

  // MATLAB Function: '<S2>/MM' incorporates:
  //   Gain: '<S2>/Gain'

  idx = rtDW.SFunction_DIMS4_h[0];
  oi_xy_size[0] = rtDW.SFunction_DIMS4_h[0];
  oi_xy_size[1] = 2;
  d_ix = (rtDW.SFunction_DIMS4_h[0] << 1) - 1;
  if (0 <= d_ix) {
    memset(&oi_xy_data[0], 0, (d_ix + 1) * sizeof(real_T));
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
      rtb_Gain1 = -1.0 / rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 5 + ix];
      b_j = rtb_X[1] - rtb_Gain1 * rtb_X[0];
      forward_length = rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 5 + ix] -
        rtb_Gain1;
      oi_xy_data[ix] = (b_j - rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 6 +
                        ix]) / forward_length;
      oi_xy_data[ix + idx] = (rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 5 +
        ix] * b_j - rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 6 + ix] *
        rtb_Gain1) / forward_length;
    }
  }

  for (idx = 0; idx < oi_xy_size[0]; idx++) {
    delta_offset = oi_xy_data[idx] - rtb_X[0];
    offset_6 = oi_xy_data[idx + oi_xy_size[0]] - rtb_X[1];
    dist_op_data[idx] = std::sqrt(delta_offset * delta_offset + offset_6 *
      offset_6);
  }

  rtb_X_state_0[0] = rtb_X[0];
  rtb_X_state_0[1] = rtb_X[1];
  MM_f(0.017453292519943295 * ajj * 180.0 / 3.1415926535897931, rtb_X_state_0,
       oi_xy_data, oi_xy_size, dist_op_data, &n, rtDW.Static_Path_0,
       rtDW.SFunction_DIMS4_h, &rtb_Gain1, &b_j, rtb_Oi_near_l, &forward_length,
       &seg_direction, &head_err, rtb_num_lane_direction_f, &seg_heading);
  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    x_data[b_ix] = (rtDW.Static_Path_0[b_ix] == rtb_Gain1);
  }

  idx = 0;
  b_x_0 = x_data[0];
  for (i = 1; i < rtDW.SFunction_DIMS4_h[0]; i++) {
    if ((int32_T)b_x_0 < (int32_T)x_data[i]) {
      b_x_0 = x_data[i];
      idx = i;
    }
  }

  // MATLAB Function: '<S2>/MATLAB Function2' incorporates:
  //   Inport: '<Root>/ID_turn'
  //   Inport: '<Root>/Look_ahead_time_straight'
  //   Inport: '<Root>/Look_ahead_time_turn'
  //   MATLAB Function: '<S2>/MM'

  if ((rtb_Gain1 >= rtU.ID_turn[0]) && (rtb_Gain1 <= rtU.ID_turn[1])) {
    b_j = rtU.Look_ahead_time_turn;
  } else if ((rtb_Gain1 >= rtU.ID_turn[2]) && (rtb_Gain1 <= rtU.ID_turn[3])) {
    b_j = rtU.Look_ahead_time_turn;
  } else if ((rtb_Gain1 >= rtU.ID_turn[4]) && (rtb_Gain1 <= rtU.ID_turn[5])) {
    b_j = rtU.Look_ahead_time_turn;
  } else if (rtb_Gain1 >= rtU.ID_turn[6]) {
    if (rtb_Gain1 <= rtU.ID_turn[7]) {
      b_j = rtU.Look_ahead_time_turn;
    } else {
      b_j = rtU.Look_ahead_time_straight;
    }
  } else {
    b_j = rtU.Look_ahead_time_straight;
  }

  // End of MATLAB Function: '<S2>/MATLAB Function2'

  // MATLAB Function: '<S2>/target_seg_id_search' incorporates:
  //   Inport: '<Root>/Speed_mps'
  //   MATLAB Function: '<S2>/Forward_Length_Decision1'
  //   MATLAB Function: '<S2>/MM'
  //   MATLAB Function: '<S2>/Target_Point_Decision'

  rtb_Gain_p = rtU.Speed_mps * b_j + 3.0;
  xy_ends_POS_size_idx_0 = rtDW.SFunction_DIMS4_h[0];
  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    xy_ends_POS_data[b_ix] = rtDW.Static_Path_0[b_ix + rtDW.SFunction_DIMS4_h[0]];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    xy_ends_POS_data[b_ix + xy_ends_POS_size_idx_0] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS4_h[0] << 1) + b_ix];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    xy_ends_POS_data[b_ix + (xy_ends_POS_size_idx_0 << 1)] =
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 3 + b_ix];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    xy_ends_POS_data[b_ix + xy_ends_POS_size_idx_0 * 3] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS4_h[0] << 2) + b_ix];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  if (0 <= d_ix - 1) {
    memcpy(&dist_op_data[0], &rtDW.Static_Path_0[0], d_ix * sizeof(real_T));
  }

  if (rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 3 +
                          rtDW.SFunction_DIMS4_h[0]) - 1] ==
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0]]) {
    c_ix = (rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] << 2) +
             rtDW.SFunction_DIMS4_h[0]) - 1] ==
            rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] << 1]);
  } else {
    c_ix = 0;
  }

  ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < ix; b_ix++) {
    x_data[b_ix] = (dist_op_data[b_ix] == rtb_Gain1);
  }

  b_idx = 1;
  b_x_0 = x_data[0];
  for (i = 2; i <= ix; i++) {
    if ((int32_T)b_x_0 < (int32_T)x_data[i - 1]) {
      b_x_0 = x_data[i - 1];
      b_idx = i;
    }
  }

  delta_offset = rtb_Oi_near_l[0] - rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0]
    * 3 + b_idx) - 1];
  offset_6 = rtb_Oi_near_l[1] - rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] <<
    2) + b_idx) - 1];
  total_length = std::sqrt(delta_offset * delta_offset + offset_6 * offset_6);
  jj = b_idx;
  n = 0;
  i = 0;
  ix = 0;
  exitg1 = false;
  while ((!exitg1) && (ix <= rtDW.SFunction_DIMS4_h[0] - 1)) {
    if (total_length > rtb_Gain_p) {
      i = jj;
      exitg1 = true;
    } else {
      b_ix = b_idx + ix;
      iy = b_ix + 1;
      if (iy <= rtDW.SFunction_DIMS4_h[0]) {
        total_length += rtDW.Static_Path_0[b_ix + (rtDW.SFunction_DIMS4_h[0] <<
          3)];
        jj = iy;
        n = 1;
        ix++;
      } else if (c_ix == 1) {
        b_ix -= rtDW.SFunction_DIMS4_h[0];
        total_length += rtDW.Static_Path_0[b_ix + (rtDW.SFunction_DIMS4_h[0] <<
          3)];
        jj = b_ix + 1;
        n = 2;
        ix++;
      } else {
        i = jj;
        n = 3;
        exitg1 = true;
      }
    }
  }

  ix = rtDW.SFunction_DIMS4_h[0];
  if (0 <= ix - 1) {
    memset(&Forward_Static_Path_id_0_data[0], 0, ix * sizeof(real_T));
  }

  if ((n == 1) || (n == 0)) {
    if (b_idx > i) {
      jj = 0;
      c_ix = 0;
    } else {
      jj = b_idx - 1;
      c_ix = i;
    }

    iy = c_ix - jj;
    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix] = xy_ends_POS_data[jj + b_ix];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy] = xy_ends_POS_data[(jj + b_ix) +
        xy_ends_POS_size_idx_0];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + (iy << 1)] = xy_ends_POS_data[(jj +
        b_ix) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy * 3] = xy_ends_POS_data[(jj +
        b_ix) + xy_ends_POS_size_idx_0 * 3];
    }

    if (b_idx > i) {
      ix = 1;
      n = 0;
    } else {
      ix = b_idx;
      n = i;
    }

    d_ix = n - ix;
    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      Forward_Static_Path_id_0_data[b_ix] = dist_op_data[(ix + b_ix) - 1];
    }

    if (b_idx > i) {
      b_idx = 1;
      i = 0;
    }

    i = (i - b_idx) + 1;
  } else if (n == 2) {
    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      c_ix = 0;
      ix = 0;
    } else {
      c_ix = b_idx - 1;
      ix = rtDW.SFunction_DIMS4_h[0];
    }

    if (1 > i) {
      d_ix = 0;
    } else {
      d_ix = i;
    }

    n = ix - c_ix;
    iy = n + d_ix;
    for (b_ix = 0; b_ix < n; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix] = xy_ends_POS_data[c_ix + b_ix];
    }

    for (b_ix = 0; b_ix < n; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy] = xy_ends_POS_data[(c_ix + b_ix)
        + xy_ends_POS_size_idx_0];
    }

    for (b_ix = 0; b_ix < n; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + (iy << 1)] = xy_ends_POS_data[(c_ix
        + b_ix) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (b_ix = 0; b_ix < n; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy * 3] = xy_ends_POS_data[(c_ix +
        b_ix) + xy_ends_POS_size_idx_0 * 3];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Static_Path_ends_POS_data[(b_ix + ix) - c_ix] = xy_ends_POS_data[b_ix];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Static_Path_ends_POS_data[((b_ix + ix) - c_ix) + iy] =
        xy_ends_POS_data[b_ix + xy_ends_POS_size_idx_0];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Static_Path_ends_POS_data[((b_ix + ix) - c_ix) + (iy << 1)] =
        xy_ends_POS_data[(xy_ends_POS_size_idx_0 << 1) + b_ix];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Static_Path_ends_POS_data[((b_ix + ix) - c_ix) + iy * 3] =
        xy_ends_POS_data[xy_ends_POS_size_idx_0 * 3 + b_ix];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      n = 0;
      c_ix = 0;
    } else {
      n = b_idx - 1;
      c_ix = rtDW.SFunction_DIMS4_h[0];
    }

    ix = ((rtDW.SFunction_DIMS4_h[0] - b_idx) + i) + 1;
    if (1 > ix) {
      tmp_0 = 0;
    } else {
      tmp_0 = (int16_T)ix;
    }

    ix = tmp_0;
    d_ix = tmp_0 - 1;
    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      cb_data[b_ix] = (int16_T)b_ix;
    }

    if (1 > i) {
      b_ix = 0;
    } else {
      b_ix = i;
    }

    d_ix = b_ix - 1;
    jj = c_ix - n;
    for (b_ix = 0; b_ix < jj; b_ix++) {
      table[b_ix] = dist_op_data[n + b_ix];
    }

    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      table[(b_ix + c_ix) - n] = dist_op_data[b_ix];
    }

    for (b_ix = 0; b_ix < ix; b_ix++) {
      Forward_Static_Path_id_0_data[cb_data[b_ix]] = table[b_ix];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_idx = 1;
      ix = 1;
    } else {
      ix = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    if (1 > i) {
      i = 0;
    }

    i += ix - b_idx;
  } else {
    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      c_ix = 0;
      n = 0;
    } else {
      c_ix = b_idx - 1;
      n = rtDW.SFunction_DIMS4_h[0];
    }

    iy = n - c_ix;
    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix] = xy_ends_POS_data[c_ix + b_ix];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy] = xy_ends_POS_data[(c_ix + b_ix)
        + xy_ends_POS_size_idx_0];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + (iy << 1)] = xy_ends_POS_data[(c_ix
        + b_ix) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy * 3] = xy_ends_POS_data[(c_ix +
        b_ix) + xy_ends_POS_size_idx_0 * 3];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      c_ix = 1;
      ix = 0;
    } else {
      c_ix = b_idx;
      ix = rtDW.SFunction_DIMS4_h[0];
    }

    d_ix = ix - c_ix;
    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      Forward_Static_Path_id_0_data[b_ix] = dist_op_data[(c_ix + b_ix) - 1];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_idx = 1;
      i = 1;
    } else {
      i = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    i -= b_idx;
  }

  if (1 > i) {
    i = 0;
  }

  rtDW.SFunction_DIMS4 = i;
  if (0 <= i - 1) {
    memcpy(&shortest_distance[0], &Forward_Static_Path_id_0_data[0], i * sizeof
           (real_T));
  }

  b_idx = iy + 1;
  rtDW.SFunction_DIMS2 = b_idx;
  rtDW.SFunction_DIMS3 = b_idx;
  rtDW.SFunction_DIMS6[0] = rtDW.SFunction_DIMS4_h[0];
  rtDW.SFunction_DIMS6[1] = 1;

  // MATLAB Function: '<S2>/Forward_Length_Decision1' incorporates:
  //   UnitDelay: '<S2>/Unit Delay14'
  //   UnitDelay: '<S2>/Unit Delay16'

  if (rtDW.UnitDelay14_DSTATE == 1.0) {
    forward_length = rtDW.UnitDelay16_DSTATE;
  } else {
    forward_length = rtb_Gain_p;
  }

  // MATLAB Function: '<S2>/Forward_Seg' incorporates:
  //   MATLAB Function: '<S2>/MM'

  xy_ends_POS_size_idx_0 = rtDW.SFunction_DIMS4_h[0];
  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    xy_ends_POS_data[b_ix] = rtDW.Static_Path_0[b_ix + rtDW.SFunction_DIMS4_h[0]];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    xy_ends_POS_data[b_ix + xy_ends_POS_size_idx_0] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS4_h[0] << 1) + b_ix];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    xy_ends_POS_data[b_ix + (xy_ends_POS_size_idx_0 << 1)] =
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 3 + b_ix];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    xy_ends_POS_data[b_ix + xy_ends_POS_size_idx_0 * 3] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS4_h[0] << 2) + b_ix];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  if (0 <= d_ix - 1) {
    memcpy(&dist_op_data[0], &rtDW.Static_Path_0[0], d_ix * sizeof(real_T));
  }

  if (rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 3 +
                          rtDW.SFunction_DIMS4_h[0]) - 1] ==
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0]]) {
    c_ix = (rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] << 2) +
             rtDW.SFunction_DIMS4_h[0]) - 1] ==
            rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] << 1]);
  } else {
    c_ix = 0;
  }

  ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < ix; b_ix++) {
    x_data[b_ix] = (dist_op_data[b_ix] == rtb_Gain1);
  }

  b_idx = 1;
  b_x_0 = x_data[0];
  for (i = 2; i <= ix; i++) {
    if ((int32_T)b_x_0 < (int32_T)x_data[i - 1]) {
      b_x_0 = x_data[i - 1];
      b_idx = i;
    }
  }

  delta_offset = rtb_Oi_near_l[0] - rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0]
    * 3 + b_idx) - 1];
  offset_6 = rtb_Oi_near_l[1] - rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] <<
    2) + b_idx) - 1];
  total_length = std::sqrt(delta_offset * delta_offset + offset_6 * offset_6);
  jj = b_idx;
  n = 0;
  i = 0;
  ix = 0;
  exitg1 = false;
  while ((!exitg1) && (ix <= rtDW.SFunction_DIMS4_h[0] - 1)) {
    if (total_length > forward_length) {
      i = jj;
      exitg1 = true;
    } else {
      b_ix = b_idx + ix;
      iy = b_ix + 1;
      if (iy <= rtDW.SFunction_DIMS4_h[0]) {
        total_length += rtDW.Static_Path_0[b_ix + (rtDW.SFunction_DIMS4_h[0] <<
          3)];
        jj = iy;
        n = 1;
        ix++;
      } else if (c_ix == 1) {
        b_ix -= rtDW.SFunction_DIMS4_h[0];
        total_length += rtDW.Static_Path_0[b_ix + (rtDW.SFunction_DIMS4_h[0] <<
          3)];
        jj = b_ix + 1;
        n = 2;
        ix++;
      } else {
        i = jj;
        n = 3;
        exitg1 = true;
      }
    }
  }

  ix = rtDW.SFunction_DIMS4_h[0];
  if (0 <= ix - 1) {
    memset(&Forward_Static_Path_id_0_data[0], 0, ix * sizeof(real_T));
  }

  if ((n == 1) || (n == 0)) {
    if (b_idx > i) {
      jj = 0;
      c_ix = 0;
    } else {
      jj = b_idx - 1;
      c_ix = i;
    }

    iy = c_ix - jj;
    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix] = xy_ends_POS_data[jj + b_ix];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy] = xy_ends_POS_data[(jj + b_ix) +
        xy_ends_POS_size_idx_0];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + (iy << 1)] = xy_ends_POS_data[(jj +
        b_ix) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy * 3] = xy_ends_POS_data[(jj +
        b_ix) + xy_ends_POS_size_idx_0 * 3];
    }

    if (b_idx > i) {
      ix = 1;
      n = 0;
    } else {
      ix = b_idx;
      n = i;
    }

    d_ix = n - ix;
    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      Forward_Static_Path_id_0_data[b_ix] = dist_op_data[(ix + b_ix) - 1];
    }

    if (b_idx > i) {
      b_idx = 1;
      i = 0;
    }

    i = (i - b_idx) + 1;
  } else if (n == 2) {
    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      c_ix = 0;
      ix = 0;
    } else {
      c_ix = b_idx - 1;
      ix = rtDW.SFunction_DIMS4_h[0];
    }

    if (1 > i) {
      d_ix = 0;
    } else {
      d_ix = i;
    }

    n = ix - c_ix;
    iy = n + d_ix;
    for (b_ix = 0; b_ix < n; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix] = xy_ends_POS_data[c_ix + b_ix];
    }

    for (b_ix = 0; b_ix < n; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy] = xy_ends_POS_data[(c_ix + b_ix)
        + xy_ends_POS_size_idx_0];
    }

    for (b_ix = 0; b_ix < n; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + (iy << 1)] = xy_ends_POS_data[(c_ix
        + b_ix) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (b_ix = 0; b_ix < n; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy * 3] = xy_ends_POS_data[(c_ix +
        b_ix) + xy_ends_POS_size_idx_0 * 3];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Static_Path_ends_POS_data[(b_ix + ix) - c_ix] = xy_ends_POS_data[b_ix];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Static_Path_ends_POS_data[((b_ix + ix) - c_ix) + iy] =
        xy_ends_POS_data[b_ix + xy_ends_POS_size_idx_0];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Static_Path_ends_POS_data[((b_ix + ix) - c_ix) + (iy << 1)] =
        xy_ends_POS_data[(xy_ends_POS_size_idx_0 << 1) + b_ix];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Static_Path_ends_POS_data[((b_ix + ix) - c_ix) + iy * 3] =
        xy_ends_POS_data[xy_ends_POS_size_idx_0 * 3 + b_ix];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      n = 0;
      c_ix = 0;
    } else {
      n = b_idx - 1;
      c_ix = rtDW.SFunction_DIMS4_h[0];
    }

    ix = ((rtDW.SFunction_DIMS4_h[0] - b_idx) + i) + 1;
    if (1 > ix) {
      tmp_0 = 0;
    } else {
      tmp_0 = (int16_T)ix;
    }

    ix = tmp_0;
    d_ix = tmp_0 - 1;
    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      cb_data[b_ix] = (int16_T)b_ix;
    }

    if (1 > i) {
      b_ix = 0;
    } else {
      b_ix = i;
    }

    d_ix = b_ix - 1;
    jj = c_ix - n;
    for (b_ix = 0; b_ix < jj; b_ix++) {
      table[b_ix] = dist_op_data[n + b_ix];
    }

    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      table[(b_ix + c_ix) - n] = dist_op_data[b_ix];
    }

    for (b_ix = 0; b_ix < ix; b_ix++) {
      Forward_Static_Path_id_0_data[cb_data[b_ix]] = table[b_ix];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_idx = 1;
      ix = 1;
    } else {
      ix = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    if (1 > i) {
      i = 0;
    }

    i += ix - b_idx;
  } else {
    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      c_ix = 0;
      n = 0;
    } else {
      c_ix = b_idx - 1;
      n = rtDW.SFunction_DIMS4_h[0];
    }

    iy = n - c_ix;
    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix] = xy_ends_POS_data[c_ix + b_ix];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy] = xy_ends_POS_data[(c_ix + b_ix)
        + xy_ends_POS_size_idx_0];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + (iy << 1)] = xy_ends_POS_data[(c_ix
        + b_ix) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy * 3] = xy_ends_POS_data[(c_ix +
        b_ix) + xy_ends_POS_size_idx_0 * 3];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      c_ix = 1;
      ix = 0;
    } else {
      c_ix = b_idx;
      ix = rtDW.SFunction_DIMS4_h[0];
    }

    d_ix = ix - c_ix;
    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      Forward_Static_Path_id_0_data[b_ix] = dist_op_data[(c_ix + b_ix) - 1];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_idx = 1;
      i = 1;
    } else {
      i = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    i -= b_idx;
  }

  if (1 > i) {
    i = 0;
  }

  rtDW.SFunction_DIMS4_f = i;
  if (0 <= i - 1) {
    memcpy(&rtb_Forward_Static_Path_id_l[0], &Forward_Static_Path_id_0_data[0],
           i * sizeof(real_T));
  }

  b_idx = iy + 1;
  d_ix = (b_idx << 1) - 1;
  if (0 <= d_ix) {
    memset(&Forward_Static_Path_data[0], 0, (d_ix + 1) * sizeof(real_T));
  }

  d_ix = iy - 1;
  if (0 <= d_ix) {
    memcpy(&Forward_Static_Path_data[0], &rtDW.Static_Path_ends_POS_data[0],
           (d_ix + 1) * sizeof(real_T));
  }

  for (b_ix = 0; b_ix <= d_ix; b_ix++) {
    Forward_Static_Path_data[b_ix + b_idx] = rtDW.Static_Path_ends_POS_data[b_ix
      + iy];
  }

  b_ix = iy - 1;
  Forward_Static_Path_data[iy] = rtDW.Static_Path_ends_POS_data[(iy << 1) + b_ix];
  Forward_Static_Path_data[iy + b_idx] = rtDW.Static_Path_ends_POS_data[iy * 3 +
    b_ix];
  rtDW.SFunction_DIMS2_h = b_idx;
  d_ix = b_idx - 1;
  if (0 <= d_ix) {
    memcpy(&rtb_Forward_Static_Path_x_h[0], &Forward_Static_Path_data[0], (d_ix
            + 1) * sizeof(real_T));
  }

  rtDW.SFunction_DIMS3_k = b_idx;
  d_ix = b_idx - 1;
  for (b_ix = 0; b_ix <= d_ix; b_ix++) {
    rtb_Forward_Static_Path_y_p[b_ix] = Forward_Static_Path_data[b_ix + b_idx];
  }

  rtDW.SFunction_DIMS6_a[0] = rtDW.SFunction_DIMS4_h[0];
  rtDW.SFunction_DIMS6_a[1] = 1;

  // MATLAB Function: '<S2>/EndPointDecision'
  xy_ends_POS_size_idx_0 = 20000;
  Path_RES_0_size_idx_1 = 2;
  memset(&rtDW.Path_RES_0_data[0], 0, 40000U * sizeof(real_T));
  memset(&rtDW.Path_RES_0_1[0], 0, 40000U * sizeof(real_T));
  count = 0.0;
  count_1 = 0.0;
  iy = 0;
  target_k = std::floor(forward_length / 0.1);
  delta_offset = rtb_Forward_Static_Path_x_h[1] - rtb_Forward_Static_Path_x_h[0];
  offset_6 = rtb_Forward_Static_Path_y_p[1] - rtb_Forward_Static_Path_y_p[0];
  Length_1 = std::sqrt(delta_offset * delta_offset + offset_6 * offset_6);
  ang_1 = rt_atan2d_snf(rtb_Forward_Static_Path_y_p[1] -
                        rtb_Forward_Static_Path_y_p[0],
                        rtb_Forward_Static_Path_x_h[1] -
                        rtb_Forward_Static_Path_x_h[0]);
  if (Length_1 > 0.1) {
    Length_1 = rt_roundd_snf(Length_1 / 0.1);
    for (c_ix = 0; c_ix < (int32_T)Length_1; c_ix++) {
      count_1 = ((1.0 + (real_T)c_ix) - 1.0) * 0.1;
      rtDW.Path_RES_0_1[c_ix] = count_1 * std::cos(ang_1) +
        rtb_Forward_Static_Path_x_h[0];
      rtDW.Path_RES_0_1[20000 + c_ix] = count_1 * std::sin(ang_1) +
        rtb_Forward_Static_Path_y_p[0];
      count_1 = 1.0 + (real_T)c_ix;
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

  for (b_ix = 0; b_ix < jj; b_ix++) {
    rtDW.Path_RES_1_data[b_ix + jj] = rtDW.Path_RES_0_1[b_ix + 20000];
  }

  i = jj;
  for (b_ix = 0; b_ix < jj; b_ix++) {
    rtDW.rtb_X_data[b_ix] = rtb_X[0] - rtDW.Path_RES_1_data[b_ix];
  }

  power_ec(rtDW.rtb_X_data, &jj, rtDW.tmp_data, &n);
  for (b_ix = 0; b_ix < jj; b_ix++) {
    rtDW.rtb_X_data[b_ix] = rtb_X[1] - rtDW.Path_RES_1_data[b_ix + jj];
  }

  power_ec(rtDW.rtb_X_data, &jj, rtDW.tmp_data_m, &i);
  for (b_ix = 0; b_ix < n; b_ix++) {
    rtDW.ob_distance_data[b_ix] = rtDW.tmp_data[b_ix] + rtDW.tmp_data_m[b_ix];
  }

  if (n <= 2) {
    if (n == 1) {
      b_idx = 0;
    } else if (rtDW.ob_distance_data[0] > rtDW.ob_distance_data[1]) {
      b_idx = 1;
    } else if (rtIsNaN(rtDW.ob_distance_data[0])) {
      if (!rtIsNaN(rtDW.ob_distance_data[1])) {
        b_ix = 2;
      } else {
        b_ix = 1;
      }

      b_idx = b_ix - 1;
    } else {
      b_idx = 0;
    }
  } else {
    if (!rtIsNaN(rtDW.ob_distance_data[0])) {
      b_idx = 0;
    } else {
      b_idx = -1;
      i = 2;
      exitg1 = false;
      while ((!exitg1) && (i <= n)) {
        if (!rtIsNaN(rtDW.ob_distance_data[i - 1])) {
          b_idx = i - 1;
          exitg1 = true;
        } else {
          i++;
        }
      }
    }

    if (b_idx + 1 == 0) {
      b_idx = 0;
    } else {
      delta_offset = rtDW.ob_distance_data[b_idx];
      for (i = b_idx + 1; i < n; i++) {
        if (delta_offset > rtDW.ob_distance_data[i]) {
          delta_offset = rtDW.ob_distance_data[i];
          b_idx = i;
        }
      }
    }
  }

  Length_1 = count_1 - (real_T)(b_idx + 1);
  if (rtDW.SFunction_DIMS2_h - 2 >= 1) {
    for (ix = 1; ix - 1 <= rtDW.SFunction_DIMS2_h - 3; ix++) {
      if (iy == 0) {
        offset_6 = rtb_Forward_Static_Path_x_h[ix + 1] -
          rtb_Forward_Static_Path_x_h[ix];
        delta_offset = rtb_Forward_Static_Path_y_p[ix + 1] -
          rtb_Forward_Static_Path_y_p[ix];
        ang_1 = std::sqrt(offset_6 * offset_6 + delta_offset * delta_offset);
        count_1 = rt_atan2d_snf(rtb_Forward_Static_Path_y_p[ix + 1] -
          rtb_Forward_Static_Path_y_p[ix], rtb_Forward_Static_Path_x_h[ix + 1] -
          rtb_Forward_Static_Path_x_h[ix]);
        if (ang_1 >= 0.1) {
          ang_1 = rt_roundd_snf(ang_1 / 0.1);
          for (c_ix = 0; c_ix < (int32_T)ang_1; c_ix++) {
            c = ((1.0 + (real_T)c_ix) - 1.0) * 0.1;
            i = (int32_T)((1.0 + (real_T)c_ix) + count);
            rtDW.Path_RES_0_data[i - 1] = c * std::cos(count_1) +
              rtb_Forward_Static_Path_x_h[ix];
            rtDW.Path_RES_0_data[i + 19999] = c * std::sin(count_1) +
              rtb_Forward_Static_Path_y_p[ix];
          }

          count += ang_1;
        } else {
          rtDW.Path_RES_0_data[(int32_T)(1.0 + count) - 1] =
            rtb_Forward_Static_Path_x_h[ix];
          rtDW.Path_RES_0_data[(int32_T)(1.0 + count) + 19999] =
            rtb_Forward_Static_Path_y_p[ix];
          count++;
        }

        if (count > target_k - Length_1) {
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

      b_ix = b_idx + (int32_T)target_k;
      seg_direction = rtDW.Path_RES_1_data[b_ix - 1];
      head_err = rtDW.Path_RES_1_data[(b_ix + jj) - 1];
      seg_heading = target_k * 0.1;
    } else {
      if (b_idx + 1 > jj) {
        b_idx = 0;
        ix = 0;
      } else {
        ix = jj;
      }

      n = ix - b_idx;
      b_ix = n + b_idx;
      seg_direction = rtDW.Path_RES_1_data[b_ix - 1];
      head_err = rtDW.Path_RES_1_data[(b_ix + jj) - 1];
      if (n == 0) {
        n = 0;
      } else {
        if (!(n > 2)) {
          n = 2;
        }
      }

      seg_heading = (real_T)n * 0.1;
    }
  } else {
    if (b_idx + 1 > jj) {
      b_idx = 0;
      i = 0;
    } else {
      i = jj;
    }

    if (1.0 > count) {
      n = 0;
    } else {
      n = (int32_T)count;
    }

    d_ix = i - b_idx;
    if (!(d_ix == 0)) {
      c_ix = 2;
      ix = d_ix;
    } else {
      if (!(n == 0)) {
        c_ix = Path_RES_0_size_idx_1;
      } else {
        c_ix = 2;
      }

      ix = 0;
    }

    if (!(n == 0)) {
      jj = n;
    } else {
      jj = 0;
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Path_RES_0_1[b_ix] = rtDW.Path_RES_1_data[b_idx + b_ix];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Path_RES_0_1[b_ix + d_ix] = rtDW.Path_RES_1_data[(b_idx + b_ix) +
        Path_RES_1_size_idx_0];
    }

    d_ix = Path_RES_0_size_idx_1 - 1;
    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      for (iy = 0; iy < n; iy++) {
        rtDW.Path_RES_0_data_c[iy + n * b_ix] =
          rtDW.Path_RES_0_data[xy_ends_POS_size_idx_0 * b_ix + iy];
      }
    }

    i = ix + jj;
    for (b_ix = 0; b_ix < c_ix; b_ix++) {
      for (iy = 0; iy < ix; iy++) {
        rtDW.Path_RES_data[iy + i * b_ix] = rtDW.Path_RES_0_1[ix * b_ix + iy];
      }
    }

    for (b_ix = 0; b_ix < c_ix; b_ix++) {
      for (iy = 0; iy < jj; iy++) {
        rtDW.Path_RES_data[(iy + ix) + i * b_ix] = rtDW.Path_RES_0_data_c[jj *
          b_ix + iy];
      }
    }

    if (target_k - Length_1 <= count) {
      seg_direction = rtDW.Path_RES_data[(int32_T)target_k - 1];
      head_err = rtDW.Path_RES_data[((int32_T)target_k + i) - 1];
      seg_heading = target_k * 0.1;
    } else {
      seg_direction_tmp_tmp = count + Length_1;
      b_ix = (int32_T)seg_direction_tmp_tmp;
      seg_direction = rtDW.Path_RES_data[b_ix - 1];
      head_err = rtDW.Path_RES_data[(b_ix + i) - 1];
      seg_heading = seg_direction_tmp_tmp * 0.1;
    }
  }

  // MATLAB Function: '<S2>/Forward_Seg1' incorporates:
  //   MATLAB Function: '<S2>/EndPointDecision'
  //   MATLAB Function: '<S2>/Forward_Seg'

  xy_ends_POS_size_idx_0 = rtDW.SFunction_DIMS4_h[0];
  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    xy_ends_POS_data[b_ix] = rtDW.Static_Path_0[b_ix + rtDW.SFunction_DIMS4_h[0]];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    xy_ends_POS_data[b_ix + xy_ends_POS_size_idx_0] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS4_h[0] << 1) + b_ix];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    xy_ends_POS_data[b_ix + (xy_ends_POS_size_idx_0 << 1)] =
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * 3 + b_ix];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    xy_ends_POS_data[b_ix + xy_ends_POS_size_idx_0 * 3] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS4_h[0] << 2) + b_ix];
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  if (0 <= d_ix - 1) {
    memcpy(&dist_op_data[0], &rtDW.Static_Path_0[0], d_ix * sizeof(real_T));
  }

  if (rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 3 +
                          rtDW.SFunction_DIMS4_h[0]) - 1] ==
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0]]) {
    c_ix = (rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] << 2) +
             rtDW.SFunction_DIMS4_h[0]) - 1] ==
            rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] << 1]);
  } else {
    c_ix = 0;
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    x_data[b_ix] = (rtb_Forward_Static_Path_id_l[rtDW.SFunction_DIMS4_f - 1] ==
                    dist_op_data[b_ix]);
  }

  b_idx = 1;
  b_x_0 = x_data[0];
  for (i = 2; i <= rtDW.SFunction_DIMS4_h[0]; i++) {
    if ((int32_T)b_x_0 < (int32_T)x_data[i - 1]) {
      b_x_0 = x_data[i - 1];
      b_idx = i;
    }
  }

  delta_offset = seg_direction - rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] *
    3 + b_idx) - 1];
  offset_6 = head_err - rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_h[0] << 2) +
    b_idx) - 1];
  total_length = std::sqrt(delta_offset * delta_offset + offset_6 * offset_6);
  jj = b_idx;
  n = 0;
  i = 0;
  ix = 0;
  exitg1 = false;
  while ((!exitg1) && (ix <= rtDW.SFunction_DIMS4_h[0] - 1)) {
    if (total_length > rtU.forward_length_2) {
      i = jj;
      exitg1 = true;
    } else {
      b_ix = b_idx + ix;
      iy = b_ix + 1;
      if (iy <= rtDW.SFunction_DIMS4_h[0]) {
        total_length += rtDW.Static_Path_0[b_ix + (rtDW.SFunction_DIMS4_h[0] <<
          3)];
        jj = iy;
        n = 1;
        ix++;
      } else if (c_ix == 1) {
        b_ix -= rtDW.SFunction_DIMS4_h[0];
        total_length += rtDW.Static_Path_0[b_ix + (rtDW.SFunction_DIMS4_h[0] <<
          3)];
        jj = b_ix + 1;
        n = 2;
        ix++;
      } else {
        i = jj;
        n = 3;
        exitg1 = true;
      }
    }
  }

  c_ix = rtDW.SFunction_DIMS4_h[0] - 1;
  if (0 <= c_ix) {
    memset(&Forward_Static_Path_id_0_data[0], 0, (c_ix + 1) * sizeof(real_T));
  }

  if ((n == 1) || (n == 0)) {
    if (b_idx > i) {
      jj = 0;
      c_ix = 0;
    } else {
      jj = b_idx - 1;
      c_ix = i;
    }

    iy = c_ix - jj;
    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix] = xy_ends_POS_data[jj + b_ix];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy] = xy_ends_POS_data[(jj + b_ix) +
        xy_ends_POS_size_idx_0];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + (iy << 1)] = xy_ends_POS_data[(jj +
        b_ix) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy * 3] = xy_ends_POS_data[(jj +
        b_ix) + xy_ends_POS_size_idx_0 * 3];
    }

    if (b_idx > i) {
      ix = 1;
      n = 0;
    } else {
      ix = b_idx;
      n = i;
    }

    d_ix = n - ix;
    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      Forward_Static_Path_id_0_data[b_ix] = dist_op_data[(ix + b_ix) - 1];
    }

    if (b_idx > i) {
      b_idx = 1;
      i = 0;
    }

    i = (i - b_idx) + 1;
  } else if (n == 2) {
    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      c_ix = 0;
      ix = 0;
    } else {
      c_ix = b_idx - 1;
      ix = rtDW.SFunction_DIMS4_h[0];
    }

    if (1 > i) {
      d_ix = 0;
    } else {
      d_ix = i;
    }

    n = ix - c_ix;
    iy = n + d_ix;
    for (b_ix = 0; b_ix < n; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix] = xy_ends_POS_data[c_ix + b_ix];
    }

    for (b_ix = 0; b_ix < n; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy] = xy_ends_POS_data[(c_ix + b_ix)
        + xy_ends_POS_size_idx_0];
    }

    for (b_ix = 0; b_ix < n; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + (iy << 1)] = xy_ends_POS_data[(c_ix
        + b_ix) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (b_ix = 0; b_ix < n; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy * 3] = xy_ends_POS_data[(c_ix +
        b_ix) + xy_ends_POS_size_idx_0 * 3];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Static_Path_ends_POS_data[(b_ix + ix) - c_ix] = xy_ends_POS_data[b_ix];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Static_Path_ends_POS_data[((b_ix + ix) - c_ix) + iy] =
        xy_ends_POS_data[b_ix + xy_ends_POS_size_idx_0];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Static_Path_ends_POS_data[((b_ix + ix) - c_ix) + (iy << 1)] =
        xy_ends_POS_data[(xy_ends_POS_size_idx_0 << 1) + b_ix];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Static_Path_ends_POS_data[((b_ix + ix) - c_ix) + iy * 3] =
        xy_ends_POS_data[xy_ends_POS_size_idx_0 * 3 + b_ix];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      n = 0;
      c_ix = 0;
    } else {
      n = b_idx - 1;
      c_ix = rtDW.SFunction_DIMS4_h[0];
    }

    ix = ((rtDW.SFunction_DIMS4_h[0] - b_idx) + i) + 1;
    if (1 > ix) {
      tmp_0 = 0;
    } else {
      tmp_0 = (int16_T)ix;
    }

    ix = tmp_0;
    d_ix = tmp_0 - 1;
    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      cb_data[b_ix] = (int16_T)b_ix;
    }

    if (1 > i) {
      b_ix = 0;
    } else {
      b_ix = i;
    }

    d_ix = b_ix - 1;
    jj = c_ix - n;
    for (b_ix = 0; b_ix < jj; b_ix++) {
      table[b_ix] = dist_op_data[n + b_ix];
    }

    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      table[(b_ix + c_ix) - n] = dist_op_data[b_ix];
    }

    for (b_ix = 0; b_ix < ix; b_ix++) {
      Forward_Static_Path_id_0_data[cb_data[b_ix]] = table[b_ix];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_idx = 1;
      ix = 1;
    } else {
      ix = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    if (1 > i) {
      i = 0;
    }

    i += ix - b_idx;
  } else {
    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      c_ix = 0;
      n = 0;
    } else {
      c_ix = b_idx - 1;
      n = rtDW.SFunction_DIMS4_h[0];
    }

    iy = n - c_ix;
    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix] = xy_ends_POS_data[c_ix + b_ix];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy] = xy_ends_POS_data[(c_ix + b_ix)
        + xy_ends_POS_size_idx_0];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + (iy << 1)] = xy_ends_POS_data[(c_ix
        + b_ix) + (xy_ends_POS_size_idx_0 << 1)];
    }

    for (b_ix = 0; b_ix < iy; b_ix++) {
      rtDW.Static_Path_ends_POS_data[b_ix + iy * 3] = xy_ends_POS_data[(c_ix +
        b_ix) + xy_ends_POS_size_idx_0 * 3];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      c_ix = 1;
      ix = 0;
    } else {
      c_ix = b_idx;
      ix = rtDW.SFunction_DIMS4_h[0];
    }

    d_ix = ix - c_ix;
    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      Forward_Static_Path_id_0_data[b_ix] = dist_op_data[(c_ix + b_ix) - 1];
    }

    if (b_idx > rtDW.SFunction_DIMS4_h[0]) {
      b_idx = 1;
      i = 1;
    } else {
      i = rtDW.SFunction_DIMS4_h[0] + 1;
    }

    i -= b_idx;
  }

  if (1 > i) {
    i = 0;
  }

  b_idx = iy + 1;
  d_ix = (b_idx << 1) - 1;
  if (0 <= d_ix) {
    memset(&Forward_Static_Path_data_0[0], 0, (d_ix + 1) * sizeof(real_T));
  }

  d_ix = iy - 1;
  if (0 <= d_ix) {
    memcpy(&Forward_Static_Path_data_0[0], &rtDW.Static_Path_ends_POS_data[0],
           (d_ix + 1) * sizeof(real_T));
  }

  for (b_ix = 0; b_ix <= d_ix; b_ix++) {
    Forward_Static_Path_data_0[b_ix + b_idx] =
      rtDW.Static_Path_ends_POS_data[b_ix + iy];
  }

  b_ix = iy - 1;
  Forward_Static_Path_data_0[iy] = rtDW.Static_Path_ends_POS_data[(iy << 1) +
    b_ix];
  Forward_Static_Path_data_0[iy + b_idx] = rtDW.Static_Path_ends_POS_data[iy * 3
    + b_ix];
  rtDW.SFunction_DIMS2_a = b_idx;
  d_ix = b_idx - 1;
  if (0 <= d_ix) {
    memcpy(&rtb_Forward_Static_Path_x_h[0], &Forward_Static_Path_data_0[0],
           (d_ix + 1) * sizeof(real_T));
  }

  rtDW.SFunction_DIMS3_a = b_idx;
  d_ix = b_idx - 1;
  for (b_ix = 0; b_ix <= d_ix; b_ix++) {
    rtb_Forward_Static_Path_y_p[b_ix] = Forward_Static_Path_data_0[b_ix + b_idx];
  }

  rtDW.SFunction_DIMS4_l = i;
  if (0 <= i - 1) {
    memcpy(&rtb_Forward_Static_Path_id_i[0], &Forward_Static_Path_id_0_data[0],
           i * sizeof(real_T));
  }

  // End of MATLAB Function: '<S2>/Forward_Seg1'

  // MATLAB Function: '<S2>/EndPointDecision1' incorporates:
  //   Inport: '<Root>/forward_length_2'
  //   MATLAB Function: '<S2>/EndPointDecision'
  //   MATLAB Function: '<S2>/EndPointDecision2'

  xy_ends_POS_size_idx_0 = 20000;
  Path_RES_0_size_idx_1 = 2;
  memset(&rtDW.Path_RES_0_data[0], 0, 40000U * sizeof(real_T));
  memset(&rtDW.Path_RES_0_1[0], 0, 40000U * sizeof(real_T));
  count = 0.0;
  count_1 = 0.0;
  iy = 0;
  target_k = std::floor(rtU.forward_length_2 / 0.1);
  rtb_UnitDelay18 = rtb_Forward_Static_Path_x_h[1] -
    rtb_Forward_Static_Path_x_h[0];
  yy_idx_0 = rtb_Forward_Static_Path_y_p[1] - rtb_Forward_Static_Path_y_p[0];
  seg_direction_tmp_tmp = std::sqrt(rtb_UnitDelay18 * rtb_UnitDelay18 + yy_idx_0
    * yy_idx_0);
  ang_1 = rt_atan2d_snf(rtb_Forward_Static_Path_y_p[1] -
                        rtb_Forward_Static_Path_y_p[0],
                        rtb_Forward_Static_Path_x_h[1] -
                        rtb_Forward_Static_Path_x_h[0]);
  if (seg_direction_tmp_tmp > 0.1) {
    Length_1 = rt_roundd_snf(seg_direction_tmp_tmp / 0.1);
    for (c_ix = 0; c_ix < (int32_T)Length_1; c_ix++) {
      count_1 = ((1.0 + (real_T)c_ix) - 1.0) * 0.1;
      rtDW.Path_RES_0_1[c_ix] = count_1 * std::cos(ang_1) +
        rtb_Forward_Static_Path_x_h[0];
      rtDW.Path_RES_0_1[20000 + c_ix] = count_1 * std::sin(ang_1) +
        rtb_Forward_Static_Path_y_p[0];
      count_1 = 1.0 + (real_T)c_ix;
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

  for (b_ix = 0; b_ix < jj; b_ix++) {
    rtDW.Path_RES_1_data[b_ix + jj] = rtDW.Path_RES_0_1[b_ix + 20000];
  }

  for (b_ix = 0; b_ix < jj; b_ix++) {
    rtDW.tmp_data_m[b_ix] = seg_direction - rtDW.Path_RES_1_data[b_ix];
  }

  power_j(rtDW.tmp_data_m, &jj, rtDW.tmp_data, &n);
  for (b_ix = 0; b_ix < jj; b_ix++) {
    rtDW.rtb_X_data[b_ix] = head_err - rtDW.Path_RES_1_data[b_ix + jj];
  }

  power_j(rtDW.rtb_X_data, &jj, rtDW.tmp_data_m, &i);
  for (b_ix = 0; b_ix < n; b_ix++) {
    rtDW.ob_distance_data[b_ix] = rtDW.tmp_data[b_ix] + rtDW.tmp_data_m[b_ix];
  }

  if (n <= 2) {
    if (n == 1) {
      b_idx = 0;
    } else if (rtDW.ob_distance_data[0] > rtDW.ob_distance_data[1]) {
      b_idx = 1;
    } else if (rtIsNaN(rtDW.ob_distance_data[0])) {
      if (!rtIsNaN(rtDW.ob_distance_data[1])) {
        b_ix = 2;
      } else {
        b_ix = 1;
      }

      b_idx = b_ix - 1;
    } else {
      b_idx = 0;
    }
  } else {
    if (!rtIsNaN(rtDW.ob_distance_data[0])) {
      b_idx = 0;
    } else {
      b_idx = -1;
      i = 2;
      exitg1 = false;
      while ((!exitg1) && (i <= n)) {
        if (!rtIsNaN(rtDW.ob_distance_data[i - 1])) {
          b_idx = i - 1;
          exitg1 = true;
        } else {
          i++;
        }
      }
    }

    if (b_idx + 1 == 0) {
      b_idx = 0;
    } else {
      delta_offset = rtDW.ob_distance_data[b_idx];
      for (i = b_idx + 1; i < n; i++) {
        if (delta_offset > rtDW.ob_distance_data[i]) {
          delta_offset = rtDW.ob_distance_data[i];
          b_idx = i;
        }
      }
    }
  }

  Length_1 = count_1 - (real_T)(b_idx + 1);
  if (rtDW.SFunction_DIMS2_a - 2 >= 1) {
    for (ix = 1; ix - 1 <= rtDW.SFunction_DIMS2_a - 3; ix++) {
      if (iy == 0) {
        offset_6 = rtb_Forward_Static_Path_x_h[ix + 1] -
          rtb_Forward_Static_Path_x_h[ix];
        delta_offset = rtb_Forward_Static_Path_y_p[ix + 1] -
          rtb_Forward_Static_Path_y_p[ix];
        ang_1 = std::sqrt(offset_6 * offset_6 + delta_offset * delta_offset);
        count_1 = rt_atan2d_snf(rtb_Forward_Static_Path_y_p[ix + 1] -
          rtb_Forward_Static_Path_y_p[ix], rtb_Forward_Static_Path_x_h[ix + 1] -
          rtb_Forward_Static_Path_x_h[ix]);
        if (ang_1 >= 0.1) {
          ang_1 = rt_roundd_snf(ang_1 / 0.1);
          for (c_ix = 0; c_ix < (int32_T)ang_1; c_ix++) {
            c = ((1.0 + (real_T)c_ix) - 1.0) * 0.1;
            i = (int32_T)((1.0 + (real_T)c_ix) + count);
            rtDW.Path_RES_0_data[i - 1] = c * std::cos(count_1) +
              rtb_Forward_Static_Path_x_h[ix];
            rtDW.Path_RES_0_data[i + 19999] = c * std::sin(count_1) +
              rtb_Forward_Static_Path_y_p[ix];
          }

          count += ang_1;
        } else {
          rtDW.Path_RES_0_data[(int32_T)(1.0 + count) - 1] =
            rtb_Forward_Static_Path_x_h[ix];
          rtDW.Path_RES_0_data[(int32_T)(1.0 + count) + 19999] =
            rtb_Forward_Static_Path_y_p[ix];
          count++;
        }

        if (count > target_k - Length_1) {
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

      b_ix = b_idx + (int32_T)target_k;
      End_x = rtDW.Path_RES_1_data[b_ix - 1];
      seg_direction_tmp_tmp = rtDW.Path_RES_1_data[(b_ix + jj) - 1];
      total_length = target_k * 0.1;
    } else {
      if (b_idx + 1 > jj) {
        b_idx = 0;
        ix = 0;
      } else {
        ix = jj;
      }

      n = ix - b_idx;
      b_ix = n + b_idx;
      End_x = rtDW.Path_RES_1_data[b_ix - 1];
      seg_direction_tmp_tmp = rtDW.Path_RES_1_data[(b_ix + jj) - 1];
      if (n == 0) {
        n = 0;
      } else {
        if (!(n > 2)) {
          n = 2;
        }
      }

      total_length = (real_T)n * 0.1;
    }
  } else {
    if (b_idx + 1 > jj) {
      b_idx = 0;
      i = 0;
    } else {
      i = jj;
    }

    if (1.0 > count) {
      n = 0;
    } else {
      n = (int32_T)count;
    }

    d_ix = i - b_idx;
    if (!(d_ix == 0)) {
      c_ix = 2;
      ix = d_ix;
    } else {
      if (!(n == 0)) {
        c_ix = Path_RES_0_size_idx_1;
      } else {
        c_ix = 2;
      }

      ix = 0;
    }

    if (!(n == 0)) {
      jj = n;
    } else {
      jj = 0;
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Path_RES_0_1[b_ix] = rtDW.Path_RES_1_data[b_idx + b_ix];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Path_RES_0_1[b_ix + d_ix] = rtDW.Path_RES_1_data[(b_idx + b_ix) +
        Path_RES_1_size_idx_0];
    }

    d_ix = Path_RES_0_size_idx_1 - 1;
    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      for (iy = 0; iy < n; iy++) {
        rtDW.Path_RES_0_data_c[iy + n * b_ix] =
          rtDW.Path_RES_0_data[xy_ends_POS_size_idx_0 * b_ix + iy];
      }
    }

    i = ix + jj;
    for (b_ix = 0; b_ix < c_ix; b_ix++) {
      for (iy = 0; iy < ix; iy++) {
        rtDW.Path_RES_data[iy + i * b_ix] = rtDW.Path_RES_0_1[ix * b_ix + iy];
      }
    }

    for (b_ix = 0; b_ix < c_ix; b_ix++) {
      for (iy = 0; iy < jj; iy++) {
        rtDW.Path_RES_data[(iy + ix) + i * b_ix] = rtDW.Path_RES_0_data_c[jj *
          b_ix + iy];
      }
    }

    if (target_k - Length_1 <= count) {
      End_x = rtDW.Path_RES_data[(int32_T)target_k - 1];
      seg_direction_tmp_tmp = rtDW.Path_RES_data[((int32_T)target_k + i) - 1];
      total_length = target_k * 0.1;
    } else {
      count_1 = count + Length_1;
      b_ix = (int32_T)count_1;
      End_x = rtDW.Path_RES_data[b_ix - 1];
      seg_direction_tmp_tmp = rtDW.Path_RES_data[(b_ix + i) - 1];
      total_length = count_1 * 0.1;
    }
  }

  // MATLAB Function: '<S2>/EndPointDecision2' incorporates:
  //   Inport: '<Root>/takeover_mag'
  //   MATLAB Function: '<S2>/DynamicPathPlanning1'
  //   MATLAB Function: '<S2>/EndPointDecision'

  xy_ends_POS_size_idx_0 = 20000;
  Path_RES_0_size_idx_1 = 2;
  memset(&rtDW.Path_RES_0_data[0], 0, 40000U * sizeof(real_T));
  memset(&rtDW.Path_RES_0_1[0], 0, 40000U * sizeof(real_T));
  count = 0.0;
  count_1 = 0.0;
  iy = 0;
  target_k_tmp = forward_length * rtU.takeover_mag;
  target_k = std::floor((target_k_tmp + 2.0) / 0.1);
  delta_offset = std::sqrt(rtb_UnitDelay18 * rtb_UnitDelay18 + yy_idx_0 *
    yy_idx_0);
  ang_1 = rt_atan2d_snf(rtb_Forward_Static_Path_y_p[1] -
                        rtb_Forward_Static_Path_y_p[0],
                        rtb_Forward_Static_Path_x_h[1] -
                        rtb_Forward_Static_Path_x_h[0]);
  if (delta_offset > 0.1) {
    Length_1 = rt_roundd_snf(delta_offset / 0.1);
    for (c_ix = 0; c_ix < (int32_T)Length_1; c_ix++) {
      count_1 = ((1.0 + (real_T)c_ix) - 1.0) * 0.1;
      rtDW.Path_RES_0_1[c_ix] = count_1 * std::cos(ang_1) +
        rtb_Forward_Static_Path_x_h[0];
      rtDW.Path_RES_0_1[20000 + c_ix] = count_1 * std::sin(ang_1) +
        rtb_Forward_Static_Path_y_p[0];
      count_1 = 1.0 + (real_T)c_ix;
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

  for (b_ix = 0; b_ix < jj; b_ix++) {
    rtDW.Path_RES_1_data[b_ix + jj] = rtDW.Path_RES_0_1[b_ix + 20000];
  }

  for (b_ix = 0; b_ix < jj; b_ix++) {
    rtDW.tmp_data_m[b_ix] = seg_direction - rtDW.Path_RES_1_data[b_ix];
  }

  power_k(rtDW.tmp_data_m, &jj, rtDW.tmp_data, &n);
  for (b_ix = 0; b_ix < jj; b_ix++) {
    rtDW.rtb_X_data[b_ix] = head_err - rtDW.Path_RES_1_data[b_ix + jj];
  }

  power_k(rtDW.rtb_X_data, &jj, rtDW.tmp_data_m, &i);
  for (b_ix = 0; b_ix < n; b_ix++) {
    rtDW.ob_distance_data[b_ix] = rtDW.tmp_data[b_ix] + rtDW.tmp_data_m[b_ix];
  }

  if (n <= 2) {
    if (n == 1) {
      b_idx = 0;
    } else if (rtDW.ob_distance_data[0] > rtDW.ob_distance_data[1]) {
      b_idx = 1;
    } else if (rtIsNaN(rtDW.ob_distance_data[0])) {
      if (!rtIsNaN(rtDW.ob_distance_data[1])) {
        b_ix = 2;
      } else {
        b_ix = 1;
      }

      b_idx = b_ix - 1;
    } else {
      b_idx = 0;
    }
  } else {
    if (!rtIsNaN(rtDW.ob_distance_data[0])) {
      b_idx = 0;
    } else {
      b_idx = -1;
      i = 2;
      exitg1 = false;
      while ((!exitg1) && (i <= n)) {
        if (!rtIsNaN(rtDW.ob_distance_data[i - 1])) {
          b_idx = i - 1;
          exitg1 = true;
        } else {
          i++;
        }
      }
    }

    if (b_idx + 1 == 0) {
      b_idx = 0;
    } else {
      delta_offset = rtDW.ob_distance_data[b_idx];
      for (i = b_idx + 1; i < n; i++) {
        if (delta_offset > rtDW.ob_distance_data[i]) {
          delta_offset = rtDW.ob_distance_data[i];
          b_idx = i;
        }
      }
    }
  }

  Length_1 = count_1 - (real_T)(b_idx + 1);
  if (rtDW.SFunction_DIMS2_a - 2 >= 1) {
    for (ix = 1; ix - 1 <= rtDW.SFunction_DIMS2_a - 3; ix++) {
      if (iy == 0) {
        offset_6 = rtb_Forward_Static_Path_x_h[ix + 1] -
          rtb_Forward_Static_Path_x_h[ix];
        delta_offset = rtb_Forward_Static_Path_y_p[ix + 1] -
          rtb_Forward_Static_Path_y_p[ix];
        ang_1 = std::sqrt(offset_6 * offset_6 + delta_offset * delta_offset);
        count_1 = rt_atan2d_snf(rtb_Forward_Static_Path_y_p[ix + 1] -
          rtb_Forward_Static_Path_y_p[ix], rtb_Forward_Static_Path_x_h[ix + 1] -
          rtb_Forward_Static_Path_x_h[ix]);
        if (ang_1 >= 0.1) {
          ang_1 = rt_roundd_snf(ang_1 / 0.1);
          for (c_ix = 0; c_ix < (int32_T)ang_1; c_ix++) {
            c = ((1.0 + (real_T)c_ix) - 1.0) * 0.1;
            i = (int32_T)((1.0 + (real_T)c_ix) + count);
            rtDW.Path_RES_0_data[i - 1] = c * std::cos(count_1) +
              rtb_Forward_Static_Path_x_h[ix];
            rtDW.Path_RES_0_data[i + 19999] = c * std::sin(count_1) +
              rtb_Forward_Static_Path_y_p[ix];
          }

          count += ang_1;
        } else {
          rtDW.Path_RES_0_data[(int32_T)(1.0 + count) - 1] =
            rtb_Forward_Static_Path_x_h[ix];
          rtDW.Path_RES_0_data[(int32_T)(1.0 + count) + 19999] =
            rtb_Forward_Static_Path_y_p[ix];
          count++;
        }

        if (count > target_k - Length_1) {
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

      b_ix = b_idx + (int32_T)target_k;
      count_1 = rtDW.Path_RES_1_data[b_ix - 1];
      ang_1 = rtDW.Path_RES_1_data[(b_ix + jj) - 1];
      count = target_k * 0.1;
    } else {
      if (b_idx + 1 > jj) {
        b_idx = 0;
        ix = 0;
      } else {
        ix = jj;
      }

      n = ix - b_idx;
      b_ix = n + b_idx;
      count_1 = rtDW.Path_RES_1_data[b_ix - 1];
      ang_1 = rtDW.Path_RES_1_data[(b_ix + jj) - 1];
      if (n == 0) {
        n = 0;
      } else {
        if (!(n > 2)) {
          n = 2;
        }
      }

      count = (real_T)n * 0.1;
    }
  } else {
    if (b_idx + 1 > jj) {
      b_idx = 0;
      i = 0;
    } else {
      i = jj;
    }

    if (1.0 > count) {
      n = 0;
    } else {
      n = (int32_T)count;
    }

    d_ix = i - b_idx;
    if (!(d_ix == 0)) {
      c_ix = 2;
      ix = d_ix;
    } else {
      if (!(n == 0)) {
        c_ix = Path_RES_0_size_idx_1;
      } else {
        c_ix = 2;
      }

      ix = 0;
    }

    if (!(n == 0)) {
      jj = n;
    } else {
      jj = 0;
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Path_RES_0_1[b_ix] = rtDW.Path_RES_1_data[b_idx + b_ix];
    }

    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Path_RES_0_1[b_ix + d_ix] = rtDW.Path_RES_1_data[(b_idx + b_ix) +
        Path_RES_1_size_idx_0];
    }

    d_ix = Path_RES_0_size_idx_1 - 1;
    for (b_ix = 0; b_ix <= d_ix; b_ix++) {
      for (iy = 0; iy < n; iy++) {
        rtDW.Path_RES_0_data_c[iy + n * b_ix] =
          rtDW.Path_RES_0_data[xy_ends_POS_size_idx_0 * b_ix + iy];
      }
    }

    i = ix + jj;
    for (b_ix = 0; b_ix < c_ix; b_ix++) {
      for (iy = 0; iy < ix; iy++) {
        rtDW.Path_RES_data[iy + i * b_ix] = rtDW.Path_RES_0_1[ix * b_ix + iy];
      }
    }

    for (b_ix = 0; b_ix < c_ix; b_ix++) {
      for (iy = 0; iy < jj; iy++) {
        rtDW.Path_RES_data[(iy + ix) + i * b_ix] = rtDW.Path_RES_0_data_c[jj *
          b_ix + iy];
      }
    }

    if (target_k - Length_1 <= count) {
      count_1 = rtDW.Path_RES_data[(int32_T)target_k - 1];
      ang_1 = rtDW.Path_RES_data[((int32_T)target_k + i) - 1];
      count = target_k * 0.1;
    } else {
      count += Length_1;
      b_ix = (int32_T)count;
      count_1 = rtDW.Path_RES_data[b_ix - 1];
      ang_1 = rtDW.Path_RES_data[(b_ix + i) - 1];
      count *= 0.1;
    }
  }

  // UnitDelay: '<S2>/Unit Delay18'
  rtb_UnitDelay18 = rtDW.UnitDelay18_DSTATE;

  // MATLAB Function: '<S2>/DangerousArea1' incorporates:
  //   Inport: '<Root>/w_off_'
  //   Inport: '<Root>/w_off_avoid'
  //   MATLAB Function: '<S2>/EndPointDecision2'
  //   UnitDelay: '<S2>/Unit Delay15'
  //   UnitDelay: '<S2>/Unit Delay17'
  //   UnitDelay: '<S2>/Unit Delay19'

  Length_1 = rtb_UnitDelay18;
  rtb_num_lane_direction_f[0] = rtDW.UnitDelay19_DSTATE[0];
  rtb_H_y_out[0] = rtDW.UnitDelay15_DSTATE[0];
  rtb_num_lane_direction_f[1] = rtDW.UnitDelay19_DSTATE[1];
  rtb_H_y_out[1] = rtDW.UnitDelay15_DSTATE[1];
  rtb_num_lane_direction_f[2] = rtDW.UnitDelay19_DSTATE[2];
  rtb_H_y_out[2] = rtDW.UnitDelay15_DSTATE[2];
  rtb_num_lane_direction_f[3] = rtDW.UnitDelay19_DSTATE[3];
  rtb_H_y_out[3] = rtDW.UnitDelay15_DSTATE[3];
  jj = 0;
  target_k = rtb_X[0];
  y = rtb_X[1];
  d_ix = rtDW.SFunction_DIMS4_f * rtDW.SFunction_DIMS4_h[1] - 1;
  if (0 <= d_ix) {
    memset(&rtDW.Forward_Static_Path_0_data[0], 0, (d_ix + 1) * sizeof(real_T));
  }

  for (ix = 0; ix < rtDW.SFunction_DIMS4_f; ix++) {
    d_ix = rtDW.SFunction_DIMS4_h[0];
    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      x_data[b_ix] = (rtb_Forward_Static_Path_id_l[ix] ==
                      rtDW.Static_Path_0[b_ix]);
    }

    b_idx = 0;
    b_x_0 = x_data[0];
    for (i = 1; i < rtDW.SFunction_DIMS4_h[0]; i++) {
      if ((int32_T)b_x_0 < (int32_T)x_data[i]) {
        b_x_0 = x_data[i];
        b_idx = i;
      }
    }

    d_ix = rtDW.SFunction_DIMS4_h[1];
    for (b_ix = 0; b_ix < d_ix; b_ix++) {
      rtDW.Forward_Static_Path_0_data[ix + rtDW.SFunction_DIMS4_f * b_ix] =
        rtDW.Static_Path_0[rtDW.SFunction_DIMS4_h[0] * b_ix + b_idx];
    }
  }

  n = 0;
  exitg1 = false;
  while ((!exitg1) && (n <= (int32_T)rtU.BB_num - 1)) {
    delta_offset = (1.0 + (real_T)n) * 2.0;
    for (b_ix = 0; b_ix < 4; b_ix++) {
      OBXY_m[b_ix << 1] = rtb_V_boundingbox[((int32_T)(delta_offset + -1.0) +
        100 * b_ix) - 1];
      OBXY_m[1 + (b_ix << 1)] = rtb_V_boundingbox[(100 * b_ix + (int32_T)
        delta_offset) - 1];
    }

    c_ix = 0;
    exitg3 = false;
    while ((!exitg3) && (c_ix <= rtDW.SFunction_DIMS4_f - 1)) {
      delta_offset = rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f <<
        2) + c_ix] - rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f <<
        1) + c_ix];
      Length_1 = rtDW.Forward_Static_Path_0_data[c_ix + rtDW.SFunction_DIMS4_f]
        - rtDW.Forward_Static_Path_0_data[rtDW.SFunction_DIMS4_f * 3 + c_ix];
      c = (rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f << 2) + c_ix]
           - rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f << 1) +
           c_ix]) * -rtDW.Forward_Static_Path_0_data[c_ix +
        rtDW.SFunction_DIMS4_f] +
        (rtDW.Forward_Static_Path_0_data[rtDW.SFunction_DIMS4_f * 3 + c_ix] -
         rtDW.Forward_Static_Path_0_data[c_ix + rtDW.SFunction_DIMS4_f]) *
        rtDW.Forward_Static_Path_0_data[(rtDW.SFunction_DIMS4_f << 1) + c_ix];
      yy_idx_0 = Length_1 * Length_1;
      offset_6 = std::sqrt(delta_offset * delta_offset + yy_idx_0);
      ex[0] = (delta_offset * OBXY_m[0] + Length_1 * OBXY_m[1]) + c;
      ex[1] = (delta_offset * OBXY_m[2] + Length_1 * OBXY_m[3]) + c;
      ex[2] = (delta_offset * OBXY_m[4] + Length_1 * OBXY_m[5]) + c;
      ex[3] = (delta_offset * OBXY_m[6] + Length_1 * OBXY_m[7]) + c;
      abs_i(ex, rtb_num_lane_direction_b);
      K1[0] = rtb_num_lane_direction_b[0] / offset_6;
      K1[1] = rtb_num_lane_direction_b[1] / offset_6;
      K1[2] = rtb_num_lane_direction_b[2] / offset_6;
      K1[3] = rtb_num_lane_direction_b[3] / offset_6;
      offset_6 = delta_offset * Length_1;
      x_endpoint2 = delta_offset * delta_offset + yy_idx_0;
      y_endpoint2 = delta_offset * c;
      rtb_num_lane_direction_f[0] = ((yy_idx_0 * OBXY_m[0] - offset_6 * OBXY_m[1])
        - y_endpoint2) / x_endpoint2;
      rtb_num_lane_direction_f[1] = ((yy_idx_0 * OBXY_m[2] - offset_6 * OBXY_m[3])
        - y_endpoint2) / x_endpoint2;
      rtb_num_lane_direction_f[2] = ((yy_idx_0 * OBXY_m[4] - offset_6 * OBXY_m[5])
        - y_endpoint2) / x_endpoint2;
      rtb_num_lane_direction_f[3] = ((yy_idx_0 * OBXY_m[6] - offset_6 * OBXY_m[7])
        - y_endpoint2) / x_endpoint2;
      offset_6 = -delta_offset * Length_1;
      y_endpoint2 = delta_offset * delta_offset;
      x_endpoint2 = delta_offset * delta_offset + yy_idx_0;
      x_endpoint3 = Length_1 * c;
      rtb_H_y_out[0] = ((offset_6 * OBXY_m[0] + y_endpoint2 * OBXY_m[1]) -
                        x_endpoint3) / x_endpoint2;
      rtb_H_y_out[1] = ((offset_6 * OBXY_m[2] + y_endpoint2 * OBXY_m[3]) -
                        x_endpoint3) / x_endpoint2;
      rtb_H_y_out[2] = ((offset_6 * OBXY_m[4] + y_endpoint2 * OBXY_m[5]) -
                        x_endpoint3) / x_endpoint2;
      rtb_H_y_out[3] = ((offset_6 * OBXY_m[6] + y_endpoint2 * OBXY_m[7]) -
                        x_endpoint3) / x_endpoint2;
      rtb_Oi_near_l[0] = ((yy_idx_0 * target_k - delta_offset * Length_1 * y) -
                          delta_offset * c) / (delta_offset * delta_offset +
        yy_idx_0);
      yy_idx_0 = ((-delta_offset * Length_1 * target_k + delta_offset *
                   delta_offset * y) - Length_1 * c) / (delta_offset *
        delta_offset + yy_idx_0);
      b_x_0 = rtIsNaN(rtb_num_lane_direction_f[0]);
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
        delta_offset = rtb_num_lane_direction_f[0];
      } else {
        delta_offset = rtb_num_lane_direction_f[b_idx - 1];
        while (b_idx + 1 < 5) {
          if (delta_offset > rtb_num_lane_direction_f[b_idx]) {
            delta_offset = rtb_num_lane_direction_f[b_idx];
          }

          b_idx++;
        }
      }

      if (rtb_Oi_near_l[0] < count_1) {
        Length_1 = count_1;
      } else if (rtIsNaN(rtb_Oi_near_l[0])) {
        if (!rtIsNaN(count_1)) {
          Length_1 = count_1;
        } else {
          Length_1 = rtb_Oi_near_l[0];
        }
      } else {
        Length_1 = rtb_Oi_near_l[0];
      }

      guard1 = false;
      if (delta_offset <= Length_1) {
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
          delta_offset = rtb_num_lane_direction_f[0];
        } else {
          delta_offset = rtb_num_lane_direction_f[b_idx - 1];
          while (b_idx + 1 < 5) {
            if (delta_offset < rtb_num_lane_direction_f[b_idx]) {
              delta_offset = rtb_num_lane_direction_f[b_idx];
            }

            b_idx++;
          }
        }

        if (rtb_Oi_near_l[0] > count_1) {
          Length_1 = count_1;
        } else if (rtIsNaN(rtb_Oi_near_l[0])) {
          if (!rtIsNaN(count_1)) {
            Length_1 = count_1;
          } else {
            Length_1 = rtb_Oi_near_l[0];
          }
        } else {
          Length_1 = rtb_Oi_near_l[0];
        }

        if (delta_offset >= Length_1) {
          b_x_0 = rtIsNaN(rtb_H_y_out[0]);
          if (!b_x_0) {
            i = 1;
          } else {
            i = 0;
            iy = 2;
            exitg4 = false;
            while ((!exitg4) && (iy < 5)) {
              if (!rtIsNaN(rtb_H_y_out[iy - 1])) {
                i = iy;
                exitg4 = true;
              } else {
                iy++;
              }
            }
          }

          if (i == 0) {
            delta_offset = rtb_H_y_out[0];
          } else {
            delta_offset = rtb_H_y_out[i - 1];
            while (i + 1 < 5) {
              if (delta_offset > rtb_H_y_out[i]) {
                delta_offset = rtb_H_y_out[i];
              }

              i++;
            }
          }

          if (yy_idx_0 < ang_1) {
            Length_1 = ang_1;
          } else if (rtIsNaN(yy_idx_0)) {
            if (!rtIsNaN(ang_1)) {
              Length_1 = ang_1;
            } else {
              Length_1 = yy_idx_0;
            }
          } else {
            Length_1 = yy_idx_0;
          }

          if (delta_offset <= Length_1) {
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
              delta_offset = rtb_H_y_out[0];
            } else {
              delta_offset = rtb_H_y_out[b_idx - 1];
              while (b_idx + 1 < 5) {
                if (delta_offset < rtb_H_y_out[b_idx]) {
                  delta_offset = rtb_H_y_out[b_idx];
                }

                b_idx++;
              }
            }

            if (yy_idx_0 > ang_1) {
              yy_idx_0 = ang_1;
            } else {
              if (rtIsNaN(yy_idx_0) && (!rtIsNaN(ang_1))) {
                yy_idx_0 = ang_1;
              }
            }

            if (delta_offset >= yy_idx_0) {
              if (!rtIsNaN(K1[0])) {
                b_idx = 1;
              } else {
                b_idx = 0;
                i = 2;
                exitg4 = false;
                while ((!exitg4) && (i < 5)) {
                  if (!rtIsNaN(K1[i - 1])) {
                    b_idx = i;
                    exitg4 = true;
                  } else {
                    i++;
                  }
                }
              }

              if (b_idx == 0) {
                delta_offset = K1[0];
              } else {
                delta_offset = K1[b_idx - 1];
                while (b_idx + 1 < 5) {
                  if (delta_offset > K1[b_idx]) {
                    delta_offset = K1[b_idx];
                  }

                  b_idx++;
                }
              }

              if (delta_offset <=
                  rtDW.Forward_Static_Path_0_data[rtDW.SFunction_DIMS4_f * 10 +
                  c_ix] / 2.0) {
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
        Length_1 = rtb_UnitDelay18;
        rtb_num_lane_direction_f[0] = rtDW.UnitDelay19_DSTATE[0];
        rtb_H_y_out[0] = rtDW.UnitDelay15_DSTATE[0];
        rtb_num_lane_direction_f[1] = rtDW.UnitDelay19_DSTATE[1];
        rtb_H_y_out[1] = rtDW.UnitDelay15_DSTATE[1];
        rtb_num_lane_direction_f[2] = rtDW.UnitDelay19_DSTATE[2];
        rtb_H_y_out[2] = rtDW.UnitDelay15_DSTATE[2];
        rtb_num_lane_direction_f[3] = rtDW.UnitDelay19_DSTATE[3];
        rtb_H_y_out[3] = rtDW.UnitDelay15_DSTATE[3];
        c_ix++;
      }
    }

    if (jj == 1) {
      exitg1 = true;
    } else {
      n++;
    }
  }

  if (Length_1 == 1.0) {
    ex[0] = count_1 - rtb_num_lane_direction_f[0];
    ex[1] = count_1 - rtb_num_lane_direction_f[1];
    ex[2] = count_1 - rtb_num_lane_direction_f[2];
    ex[3] = count_1 - rtb_num_lane_direction_f[3];
    power(ex, rtb_num_lane_direction_b);
    ex[0] = ang_1 - rtb_H_y_out[0];
    ex[1] = ang_1 - rtb_H_y_out[1];
    ex[2] = ang_1 - rtb_H_y_out[2];
    ex[3] = ang_1 - rtb_H_y_out[3];
    power(ex, tmp);
    K1[0] = rtb_num_lane_direction_b[0] + tmp[0];
    K1[1] = rtb_num_lane_direction_b[1] + tmp[1];
    K1[2] = rtb_num_lane_direction_b[2] + tmp[2];
    K1[3] = rtb_num_lane_direction_b[3] + tmp[3];
    if (!rtIsNaN(K1[0])) {
      i = 1;
    } else {
      i = 0;
      b_idx = 2;
      exitg1 = false;
      while ((!exitg1) && (b_idx < 5)) {
        if (!rtIsNaN(K1[b_idx - 1])) {
          i = b_idx;
          exitg1 = true;
        } else {
          b_idx++;
        }
      }
    }

    if (i == 0) {
      target_k = K1[0];
    } else {
      target_k = K1[i - 1];
      while (i + 1 < 5) {
        if (target_k > K1[i]) {
          target_k = K1[i];
        }

        i++;
      }
    }

    if (std::sqrt(target_k) > (forward_length + count) + 4.666666666666667) {
      Length_1 = 0.0;
    }
  }

  if (Length_1 == 1.0) {
    count_1 = 100.0;
    count = rtU.w_off_avoid;
  } else {
    count_1 = rtDW.UnitDelay17_DSTATE - 1.0;
    if (rtDW.UnitDelay17_DSTATE - 1.0 < 0.0) {
      count_1 = 0.0;
    }

    count = rtU.w_off_;
  }

  // SignalConversion: '<S11>/TmpSignal ConversionAt SFunction Inport7' incorporates:
  //   Gain: '<S2>/Gain1'
  //   MATLAB Function: '<S2>/DynamicPathPlanning'

  rtb_TmpSignalConversionAtSFun_e[0] = rtb_X[0];
  rtb_TmpSignalConversionAtSFun_e[1] = rtb_X[1];
  rtb_TmpSignalConversionAtSFun_e[2] = 0.017453292519943295 * ajj;

  // MATLAB Function: '<S2>/DynamicPathPlanning' incorporates:
  //   Constant: '<S2>/Constant12'
  //   Constant: '<S2>/Constant13'
  //   Constant: '<S2>/Constant16'
  //   Inport: '<Root>/BB_num'
  //   Inport: '<Root>/Freespace'
  //   Inport: '<Root>/Freespace_mode'
  //   Inport: '<Root>/W_1'
  //   Inport: '<Root>/safe_range'
  //   MATLAB Function: '<S2>/DangerousArea1'
  //   MATLAB Function: '<S2>/EndPointDecision'
  //   MATLAB Function: '<S2>/MM'
  //   SignalConversion: '<S11>/TmpSignal ConversionAt SFunction Inport7'
  //   UnitDelay: '<S2>/Unit Delay5'

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    x_data[b_ix] = (rtb_Forward_Static_Path_id_l[rtDW.SFunction_DIMS4_f - 1] ==
                    rtDW.Static_Path_0[b_ix]);
  }

  i = rtDW.SFunction_DIMS4_h[0] - 1;
  b_idx = 0;
  for (c_ix = 0; c_ix <= i; c_ix++) {
    if (x_data[c_ix]) {
      b_idx++;
    }
  }

  ix = b_idx;
  b_idx = 0;
  for (c_ix = 0; c_ix <= i; c_ix++) {
    if (x_data[c_ix]) {
      t_data[b_idx] = c_ix + 1;
      b_idx++;
    }
  }

  for (b_ix = 0; b_ix < ix; b_ix++) {
    dist_op_data[b_ix] = rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 7 +
      t_data[b_ix]) - 1] * 3.1415926535897931 / 180.0;
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    x_data[b_ix] = (rtb_Forward_Static_Path_id_l[rtDW.SFunction_DIMS4_f - 1] ==
                    rtDW.Static_Path_0[b_ix]);
  }

  c_ix = 0;
  for (ix = 0; ix < rtDW.SFunction_DIMS4_h[0]; ix++) {
    if (x_data[ix]) {
      u_data[c_ix] = ix + 1;
      c_ix++;
    }
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    x_data[b_ix] = (rtb_Forward_Static_Path_id_l[rtDW.SFunction_DIMS4_f - 1] ==
                    rtDW.Static_Path_0[b_ix]);
  }

  b_idx = 0;
  for (ix = 0; ix < rtDW.SFunction_DIMS4_h[0]; ix++) {
    if (x_data[ix]) {
      v_data[b_idx] = ix + 1;
      b_idx++;
    }
  }

  delta_offset = rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 10 + v_data[0])
    - 1] / 4.0;
  target_k = delta_offset * 2.0;
  yy_idx_0 = delta_offset * 3.0;
  rtb_UnitDelay18 = delta_offset * 4.0;
  y = delta_offset * 5.0;
  offset_6 = delta_offset * 6.0;
  offset[0] = offset_6;
  offset[1] = y;
  offset[2] = rtb_UnitDelay18;
  offset[3] = yy_idx_0;
  offset[4] = target_k;
  offset[5] = delta_offset;
  offset[6] = 0.0;
  offset[7] = delta_offset;
  offset[8] = target_k;
  offset[9] = yy_idx_0;
  offset[10] = rtb_UnitDelay18;
  offset[11] = y;
  offset[12] = offset_6;
  x_endpoint6 = std::cos(dist_op_data[0] + 1.5707963267948966);
  ang_1 = x_endpoint6 * offset_6 + seg_direction;
  y_endpoint6 = std::sin(dist_op_data[0] + 1.5707963267948966);
  c = y_endpoint6 * offset_6 + head_err;
  x_endpoint2 = x_endpoint6 * y + seg_direction;
  y_endpoint2 = y_endpoint6 * y + head_err;
  x_endpoint3 = x_endpoint6 * rtb_UnitDelay18 + seg_direction;
  y_endpoint3 = y_endpoint6 * rtb_UnitDelay18 + head_err;
  x_endpoint4 = x_endpoint6 * yy_idx_0 + seg_direction;
  y_endpoint4 = y_endpoint6 * yy_idx_0 + head_err;
  x_endpoint5 = x_endpoint6 * target_k + seg_direction;
  y_endpoint5 = y_endpoint6 * target_k + head_err;
  x_endpoint6 = x_endpoint6 * delta_offset + seg_direction;
  y_endpoint6 = y_endpoint6 * delta_offset + head_err;
  x_endpoint13 = std::cos(dist_op_data[0] - 1.5707963267948966);
  x_endpoint8 = x_endpoint13 * delta_offset + seg_direction;
  xy_end_point_idx_25 = std::sin(dist_op_data[0] - 1.5707963267948966);
  y_endpoint8 = xy_end_point_idx_25 * delta_offset + head_err;
  x_endpoint9 = x_endpoint13 * target_k + seg_direction;
  y_endpoint9 = xy_end_point_idx_25 * target_k + head_err;
  x_endpoint10 = x_endpoint13 * yy_idx_0 + seg_direction;
  y_endpoint10 = xy_end_point_idx_25 * yy_idx_0 + head_err;
  x_endpoint11 = x_endpoint13 * rtb_UnitDelay18 + seg_direction;
  y_endpoint11 = xy_end_point_idx_25 * rtb_UnitDelay18 + head_err;
  x_endpoint12 = x_endpoint13 * y + seg_direction;
  y_endpoint12 = xy_end_point_idx_25 * y + head_err;
  x_endpoint13 = x_endpoint13 * offset_6 + seg_direction;
  offset_6 = xy_end_point_idx_25 * offset_6 + head_err;
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_e[2],
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13], ang_1, c,
            dist_op_data[0], rtDW.Static_Path_0[(u_data[0] +
             rtDW.SFunction_DIMS4_h[0] * 13) - 1], seg_heading, x,
            b_Path_dis_data, XP1, YP1, K1_0, K_11, &rtb_J_out_k[0]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_e[2],
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint2, y_endpoint2, dist_op_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], seg_heading, X2,
            Y2, XP2, YP2, K2, K_12, &rtb_J_out_k[1]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_e[2],
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint3, y_endpoint3, dist_op_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], seg_heading, X3,
            Y3, XP3, YP3, K3, K_13, &rtb_J_out_k[2]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_e[2],
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint4, y_endpoint4, dist_op_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], seg_heading, X4,
            Y4, XP4, YP4, K4, K_14, &rtb_J_out_k[3]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_e[2],
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint5, y_endpoint5, dist_op_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], seg_heading, X5,
            Y5, XP5, YP5, K5, K_15, &rtb_J_out_k[4]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_e[2],
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint6, y_endpoint6, dist_op_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], seg_heading, X6,
            Y6, XP6, YP6, K6, K_16, &rtb_J_out_k[5]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_e[2],
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            seg_direction, head_err, dist_op_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], seg_heading, X7,
            Y7, XP7, YP7, K7, K_17, &rtb_J_out_k[6]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_e[2],
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint8, y_endpoint8, dist_op_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], seg_heading, X8,
            Y8, XP8, YP8, K8, K_18, &rtb_J_out_k[7]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_e[2],
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint9, y_endpoint9, dist_op_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], seg_heading, X9,
            Y9, XP9, YP9, K9, K_19, &rtb_J_out_k[8]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_e[2],
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint10, y_endpoint10, dist_op_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], seg_heading, X10,
            Y10, XP10, YP10, K10, K_110, &rtb_J_out_k[9]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_e[2],
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint11, y_endpoint11, dist_op_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], seg_heading, X11,
            Y11, XP11, YP11, K11, K_111, &rtb_J_out_k[10]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_e[2],
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint12, y_endpoint12, dist_op_data[0], rtDW.Static_Path_0
            [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], seg_heading, X12,
            Y12, XP12, YP12, K12, K_112, &rtb_J_out_k[11]);
  G2splines(rtb_X[0], rtb_X[1], rtb_TmpSignalConversionAtSFun_e[2],
            rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4_h[0] * 13],
            x_endpoint13, offset_6, dist_op_data[0], rtDW.Static_Path_0[(u_data
             [0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], seg_heading, X13, Y13,
            XP13, YP13, K13, K_113, &rtb_J_out_k[12]);
  for (b_ix = 0; b_ix < 11; b_ix++) {
    X_2[b_ix] = x[b_ix];
    X_2[b_ix + 11] = X2[b_ix];
    X_2[b_ix + 22] = X3[b_ix];
    X_2[b_ix + 33] = X4[b_ix];
    X_2[b_ix + 44] = X5[b_ix];
    X_2[b_ix + 55] = X6[b_ix];
    X_2[b_ix + 66] = X7[b_ix];
    X_2[b_ix + 77] = X8[b_ix];
    X_2[b_ix + 88] = X9[b_ix];
    X_2[b_ix + 99] = X10[b_ix];
    X_2[b_ix + 110] = X11[b_ix];
    X_2[b_ix + 121] = X12[b_ix];
    X_2[b_ix + 132] = X13[b_ix];
    Y[b_ix] = b_Path_dis_data[b_ix];
    Y[b_ix + 11] = Y2[b_ix];
    Y[b_ix + 22] = Y3[b_ix];
    Y[b_ix + 33] = Y4[b_ix];
    Y[b_ix + 44] = Y5[b_ix];
    Y[b_ix + 55] = Y6[b_ix];
    Y[b_ix + 66] = Y7[b_ix];
    Y[b_ix + 77] = Y8[b_ix];
    Y[b_ix + 88] = Y9[b_ix];
    Y[b_ix + 99] = Y10[b_ix];
    Y[b_ix + 110] = Y11[b_ix];
    Y[b_ix + 121] = Y12[b_ix];
    Y[b_ix + 132] = Y13[b_ix];
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

  for (b_ix = 0; b_ix < 11; b_ix++) {
    K[b_ix] = K1_0[b_ix];
    K[b_ix + 11] = K2[b_ix];
    K[b_ix + 22] = K3[b_ix];
    K[b_ix + 33] = K4[b_ix];
    K[b_ix + 44] = K5[b_ix];
    K[b_ix + 55] = K6[b_ix];
    K[b_ix + 66] = K7[b_ix];
    K[b_ix + 77] = K8[b_ix];
    K[b_ix + 88] = K9[b_ix];
    K[b_ix + 99] = K10[b_ix];
    K[b_ix + 110] = K11[b_ix];
    K[b_ix + 121] = K12[b_ix];
    K[b_ix + 132] = K13[b_ix];
    K_1[b_ix] = K_11[b_ix];
    K_1[b_ix + 11] = K_12[b_ix];
    K_1[b_ix + 22] = K_13[b_ix];
    K_1[b_ix + 33] = K_14[b_ix];
    K_1[b_ix + 44] = K_15[b_ix];
    K_1[b_ix + 55] = K_16[b_ix];
    K_1[b_ix + 66] = K_17[b_ix];
    K_1[b_ix + 77] = K_18[b_ix];
    K_1[b_ix + 88] = K_19[b_ix];
    K_1[b_ix + 99] = K_110[b_ix];
    K_1[b_ix + 110] = K_111[b_ix];
    K_1[b_ix + 121] = K_112[b_ix];
    K_1[b_ix + 132] = K_113[b_ix];
  }

  xy_end_point_idx_25 = offset_6;
  memset(&Path_col[0], 0, 52U * sizeof(real_T));
  for (b_ix = 0; b_ix < 5; b_ix++) {
    Path_col[3 + ((8 + b_ix) << 2)] = 1.0;
  }

  Path_col[3] = 1.0;
  Path_col[51] = 1.0;
  if ((rtU.Freespace_mode == 0.0) || (rtU.Freespace_mode == 2.0)) {
    memcpy(&OBXY_EL[0], &rtb_V_boundingbox[0], 400U * sizeof(real_T));
    for (n = 0; n < (int32_T)rtU.BB_num; n++) {
      delta_offset = (1.0 + (real_T)n) * 2.0;
      i = (int32_T)(delta_offset + -1.0);
      idx = i - 1;
      OBXY_EL[idx] = ((rtb_V_boundingbox[idx] - rtb_V_boundingbox[i + 99]) *
                      0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0
        + -1.0) - 1]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 +
        -1.0) - 1] - rtb_V_boundingbox[i + 299]) * 0.3;
      idx = (int32_T)delta_offset;
      b_ix = idx - 1;
      OBXY_EL[b_ix] = ((rtb_V_boundingbox[b_ix] - rtb_V_boundingbox[idx + 99]) *
                       0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
        2.0) - 1]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) - 1]
                      - rtb_V_boundingbox[idx + 299]) * 0.3;
      OBXY_EL[(int32_T)(delta_offset + -1.0) + 99] = ((rtb_V_boundingbox
        [(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 99] - rtb_V_boundingbox
        [(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) - 1]) * 0.15 +
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 99]) +
        (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 99] -
         rtb_V_boundingbox[i + 199]) * 0.3;
      OBXY_EL[(int32_T)delta_offset + 99] = ((rtb_V_boundingbox[(int32_T)((1.0 +
        (real_T)n) * 2.0) + 99] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
        2.0) - 1]) * 0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0)
        + 99]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) + 99] -
                  rtb_V_boundingbox[idx + 199]) * 0.3;
      OBXY_EL[(int32_T)(delta_offset + -1.0) + 199] = ((rtb_V_boundingbox
        [(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 199] - rtb_V_boundingbox
        [(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 299]) * 0.15 +
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 199]) +
        (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 199] -
         rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 99]) *
        0.3;
      OBXY_EL[(int32_T)delta_offset + 199] = ((rtb_V_boundingbox[(int32_T)((1.0
        + (real_T)n) * 2.0) + 199] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
        n) * 2.0) + 299]) * 0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n)
        * 2.0) + 199]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) +
                          199] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
        2.0) + 99]) * 0.3;
      OBXY_EL[(int32_T)(delta_offset + -1.0) + 299] = ((rtb_V_boundingbox
        [(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 299] - rtb_V_boundingbox
        [(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 199]) * 0.15 +
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 299]) +
        (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 299] -
         rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) - 1]) * 0.3;
      OBXY_EL[(int32_T)delta_offset + 299] = ((rtb_V_boundingbox[(int32_T)((1.0
        + (real_T)n) * 2.0) + 299] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
        n) * 2.0) + 199]) * 0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n)
        * 2.0) + 299]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) +
                          299] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
        2.0) - 1]) * 0.3;
    }

    for (b_ix = 0; b_ix < 13; b_ix++) {
      for (iy = 0; iy < 10; iy++) {
        i = 11 * b_ix + iy;
        delta_offset = X_2[i + 1] - X_2[i];
        X_diff[iy + 11 * b_ix] = delta_offset;
        X_diff_0[iy + 10 * b_ix] = delta_offset;
      }

      idx = 10 + 11 * b_ix;
      X_diff[idx] = X_diff_0[10 * b_ix + 9];
      for (iy = 0; iy < 10; iy++) {
        i = 11 * b_ix + iy;
        delta_offset = Y[i + 1] - Y[i];
        Y_diff[iy + 11 * b_ix] = delta_offset;
        X_diff_0[iy + 10 * b_ix] = delta_offset;
      }

      Y_diff[idx] = X_diff_0[10 * b_ix + 9];
    }

    power_dw(X_diff, XY_difflen);
    power_dw(Y_diff, Path_vehFLY);
    for (b_ix = 0; b_ix < 143; b_ix++) {
      Path_vehFLX[b_ix] = XY_difflen[b_ix] + Path_vehFLY[b_ix];
    }

    power_dw3(Path_vehFLX, XY_difflen);
    for (b_ix = 0; b_ix < 143; b_ix++) {
      delta_offset = X_diff[b_ix] / XY_difflen[b_ix];
      target_k = Y_diff[b_ix] / XY_difflen[b_ix];
      yy_idx_0 = 1.1 * -target_k + X_2[b_ix];
      Path_vehFLX[b_ix] = yy_idx_0 + 1.4000000000000001 * delta_offset;
      rtb_UnitDelay18 = 1.1 * delta_offset + Y[b_ix];
      Path_vehFLY[b_ix] = rtb_UnitDelay18 + 1.4000000000000001 * target_k;
      y = X_2[b_ix] - 1.1 * -target_k;
      Path_vehFRX[b_ix] = y + 1.4000000000000001 * delta_offset;
      offset_6 = Y[b_ix] - 1.1 * delta_offset;
      Path_vehFRY[b_ix] = offset_6 + 1.4000000000000001 * target_k;
      Path_vehRLX[b_ix] = yy_idx_0 - 5.6000000000000005 * delta_offset;
      Path_vehRLY[b_ix] = rtb_UnitDelay18 - 5.6000000000000005 * target_k;
      Path_vehRRX[b_ix] = y - 5.6000000000000005 * delta_offset;
      Path_vehRRY[b_ix] = offset_6 - 5.6000000000000005 * target_k;
      X_diff[b_ix] = delta_offset;
      XY_difflen[b_ix] = -target_k;
      Y_diff[b_ix] = target_k;
    }

    for (ix = 0; ix < 13; ix++) {
      Path_col[ix << 2] = 0.0;
      if (!(Path_col[(ix << 2) + 3] == 1.0)) {
        c_ix = 0;
        exitg1 = false;
        while ((!exitg1) && (c_ix < 11)) {
          d_ix = 11 * ix + c_ix;
          OBXY_m[0] = Path_vehFLX[d_ix];
          OBXY_m[2] = Path_vehFRX[d_ix];
          OBXY_m[4] = Path_vehRLX[d_ix];
          OBXY_m[6] = Path_vehRRX[d_ix];
          OBXY_m[1] = Path_vehFLY[d_ix];
          OBXY_m[3] = Path_vehFRY[d_ix];
          OBXY_m[5] = Path_vehRLY[d_ix];
          OBXY_m[7] = Path_vehRRY[d_ix];
          i = 0;
          exitg3 = false;
          while ((!exitg3) && (i <= (int32_T)rtU.BB_num - 1)) {
            delta_offset = (1.0 + (real_T)i) * 2.0;
            idx = (int32_T)(delta_offset + -1.0);
            rtb_UnitDelay18 = OBXY_EL[idx + 99] - OBXY_EL[idx - 1];
            target_k = std::sqrt(rtb_UnitDelay18 * rtb_UnitDelay18 +
                                 rtb_UnitDelay18 * rtb_UnitDelay18);
            b_ix = (int32_T)delta_offset;
            rtb_Oi_near_l[0] = -(OBXY_EL[b_ix + 99] - OBXY_EL[b_ix - 1]) /
              target_k;
            rtb_Oi_near_l[1] = rtb_UnitDelay18 / target_k;
            yy_idx_0 = OBXY_EL[b_ix + 199] - OBXY_EL[(int32_T)((1.0 + (real_T)i)
              * 2.0) + 99];
            rtb_UnitDelay18 = OBXY_EL[idx + 199] - OBXY_EL[(int32_T)((1.0 +
              (real_T)i) * 2.0 + -1.0) + 99];
            offset_6 = std::sqrt(yy_idx_0 * yy_idx_0 + rtb_UnitDelay18 *
                                 rtb_UnitDelay18);
            yy_idx_0 = -yy_idx_0 / offset_6;
            target_k = rtb_UnitDelay18 / offset_6;
            rtb_Oi_near_o[0] = rtb_Oi_near_l[0];
            rtb_Oi_near_o[1] = yy_idx_0;
            rtb_Oi_near_o[4] = rtb_Oi_near_l[1];
            rtb_Oi_near_o[5] = target_k;
            rtb_Oi_near_o[2] = X_diff[d_ix];
            rtb_Oi_near_o[6] = Y_diff[d_ix];
            rtb_Oi_near_o[3] = XY_difflen[d_ix];
            rtb_Oi_near_o[7] = X_diff[11 * ix + c_ix];
            rtb_Oi_near_o_0[0] = rtb_Oi_near_l[0];
            rtb_Oi_near_o_0[1] = yy_idx_0;
            rtb_Oi_near_o_0[4] = rtb_Oi_near_l[1];
            rtb_Oi_near_o_0[5] = target_k;
            rtb_Oi_near_o_0[2] = X_diff[11 * ix + c_ix];
            rtb_Oi_near_o_0[6] = Y_diff[11 * ix + c_ix];
            rtb_Oi_near_o_0[3] = XY_difflen[11 * ix + c_ix];
            rtb_Oi_near_o_0[7] = X_diff[11 * ix + c_ix];
            for (b_ix = 0; b_ix < 4; b_ix++) {
              for (iy = 0; iy < 4; iy++) {
                proj_veh[b_ix + (iy << 2)] = 0.0;
                proj_veh[b_ix + (iy << 2)] += OBXY_m[iy << 1] *
                  rtb_Oi_near_o[b_ix];
                proj_veh[b_ix + (iy << 2)] += OBXY_m[(iy << 1) + 1] *
                  rtb_Oi_near_o[b_ix + 4];
              }

              OBXY_EL_0[b_ix << 1] = OBXY_EL[((int32_T)(delta_offset + -1.0) +
                100 * b_ix) - 1];
              OBXY_EL_0[1 + (b_ix << 1)] = OBXY_EL[(100 * b_ix + (int32_T)
                delta_offset) - 1];
            }

            for (b_idx = 0; b_idx < 4; b_idx++) {
              for (b_ix = 0; b_ix < 4; b_ix++) {
                proj_ob[b_idx + (b_ix << 2)] = 0.0;
                proj_ob[b_idx + (b_ix << 2)] += OBXY_EL_0[b_ix << 1] *
                  rtb_Oi_near_o_0[b_idx];
                proj_ob[b_idx + (b_ix << 2)] += OBXY_EL_0[(b_ix << 1) + 1] *
                  rtb_Oi_near_o_0[b_idx + 4];
              }

              K1[b_idx] = proj_veh[b_idx];
            }

            delta_offset = proj_veh[0];
            target_k = proj_veh[1];
            yy_idx_0 = proj_veh[2];
            rtb_UnitDelay18 = proj_veh[3];
            for (idx = 0; idx < 3; idx++) {
              if ((!rtIsNaN(proj_veh[(idx + 1) << 2])) && (rtIsNaN(K1[0]) ||
                   (K1[0] > proj_veh[(idx + 1) << 2]))) {
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

              y = delta_offset;
              if ((!rtIsNaN(proj_veh[(idx + 1) << 2])) && (rtIsNaN(delta_offset)
                   || (delta_offset < proj_veh[(idx + 1) << 2]))) {
                y = proj_veh[(idx + 1) << 2];
              }

              delta_offset = y;
              y = target_k;
              if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 1])) && (rtIsNaN
                   (target_k) || (target_k < proj_veh[((idx + 1) << 2) + 1]))) {
                y = proj_veh[((idx + 1) << 2) + 1];
              }

              target_k = y;
              y = yy_idx_0;
              if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 2])) && (rtIsNaN
                   (yy_idx_0) || (yy_idx_0 < proj_veh[((idx + 1) << 2) + 2]))) {
                y = proj_veh[((idx + 1) << 2) + 2];
              }

              yy_idx_0 = y;
              y = rtb_UnitDelay18;
              if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 3])) && (rtIsNaN
                   (rtb_UnitDelay18) || (rtb_UnitDelay18 < proj_veh[((idx + 1) <<
                     2) + 3]))) {
                y = proj_veh[((idx + 1) << 2) + 3];
              }

              rtb_UnitDelay18 = y;
            }

            minmax_veh[0] = K1[0];
            minmax_veh[4] = delta_offset;
            minmax_veh[1] = K1[1];
            minmax_veh[5] = target_k;
            minmax_veh[2] = K1[2];
            minmax_veh[6] = yy_idx_0;
            minmax_veh[3] = K1[3];
            minmax_veh[7] = rtb_UnitDelay18;
            K1[0] = proj_ob[0];
            K1[1] = proj_ob[1];
            K1[2] = proj_ob[2];
            K1[3] = proj_ob[3];
            delta_offset = proj_ob[0];
            target_k = proj_ob[1];
            yy_idx_0 = proj_ob[2];
            rtb_UnitDelay18 = proj_ob[3];
            for (idx = 0; idx < 3; idx++) {
              if ((!rtIsNaN(proj_ob[(idx + 1) << 2])) && (rtIsNaN(K1[0]) || (K1
                    [0] > proj_ob[(idx + 1) << 2]))) {
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

              y = delta_offset;
              if ((!rtIsNaN(proj_ob[(idx + 1) << 2])) && (rtIsNaN(delta_offset) ||
                   (delta_offset < proj_ob[(idx + 1) << 2]))) {
                y = proj_ob[(idx + 1) << 2];
              }

              delta_offset = y;
              y = target_k;
              if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 1])) && (rtIsNaN(target_k)
                   || (target_k < proj_ob[((idx + 1) << 2) + 1]))) {
                y = proj_ob[((idx + 1) << 2) + 1];
              }

              target_k = y;
              y = yy_idx_0;
              if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 2])) && (rtIsNaN(yy_idx_0)
                   || (yy_idx_0 < proj_ob[((idx + 1) << 2) + 2]))) {
                y = proj_ob[((idx + 1) << 2) + 2];
              }

              yy_idx_0 = y;
              y = rtb_UnitDelay18;
              if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 3])) && (rtIsNaN
                   (rtb_UnitDelay18) || (rtb_UnitDelay18 < proj_ob[((idx + 1) <<
                     2) + 3]))) {
                y = proj_ob[((idx + 1) << 2) + 3];
              }

              rtb_UnitDelay18 = y;
            }

            minmax_obj[0] = K1[0];
            minmax_obj[4] = delta_offset;
            minmax_obj[1] = K1[1];
            minmax_obj[5] = target_k;
            minmax_obj[2] = K1[2];
            minmax_obj[6] = yy_idx_0;
            minmax_obj[3] = K1[3];
            minmax_obj[7] = rtb_UnitDelay18;
            n = 0;
            exitg4 = false;
            while ((!exitg4) && (n < 4)) {
              if (minmax_veh[n] > minmax_obj[4 + n]) {
                Path_col[ix << 2] = 0.0;
                exitg4 = true;
              } else if (minmax_veh[4 + n] < minmax_obj[n]) {
                Path_col[ix << 2] = 0.0;
                exitg4 = true;
              } else {
                Path_col[ix << 2] = 1.0;
                n++;
              }
            }

            if (Path_col[ix << 2] == 1.0) {
              Path_col[2 + (ix << 2)] = 1.0 + (real_T)i;
              exitg3 = true;
            } else {
              i++;
            }
          }

          if (Path_col[ix << 2] == 1.0) {
            Path_col[1 + (ix << 2)] = 1.0 + (real_T)c_ix;
            exitg1 = true;
          } else {
            c_ix++;
          }
        }
      }
    }
  }

  for (b_ix = 0; b_ix < 13; b_ix++) {
    Cobs[b_ix] = Path_col[b_ix << 2];
    Cobs_0[b_ix] = Path_col[b_ix << 2];
  }

  target_k = std(Cobs_0);
  if (target_k != 0.0) {
    delta_offset = target_k * target_k * 2.0;
    rtb_UnitDelay18 = 2.5066282746310002 * target_k;
    for (idx = 0; idx < 13; idx++) {
      b_ix = 1 + idx;
      for (iy = 0; iy < 13; iy++) {
        Cc_0[iy] = (b_ix - iy) - 1;
      }

      power_dw3x(Cc_0, rtb_forward_length_free);
      for (b_ix = 0; b_ix < 13; b_ix++) {
        Cc_0[b_ix] = -rtb_forward_length_free[b_ix] / delta_offset;
      }

      exp_n(Cc_0);
      for (b_ix = 0; b_ix < 13; b_ix++) {
        Cobs_0[b_ix] = Path_col[b_ix << 2] * (Cc_0[b_ix] / rtb_UnitDelay18);
      }

      Cobs[idx] = sum_a(Cobs_0);
      if ((1 + idx == 1) && (Path_col[0] == 1.0)) {
        Cobs[0] += std::exp(-1.0 / (target_k * target_k * 2.0)) /
          (2.5066282746310002 * target_k);
      } else {
        if ((1 + idx == 13) && (Path_col[48] == 1.0)) {
          Cobs[12] += std::exp(-1.0 / (target_k * target_k * 2.0)) /
            (2.5066282746310002 * target_k);
        }
      }
    }

    b_x_0 = rtIsNaN(Cobs[0]);
    if (!b_x_0) {
      idx = 1;
    } else {
      idx = 0;
      i = 2;
      exitg1 = false;
      while ((!exitg1) && (i < 14)) {
        if (!rtIsNaN(Cobs[i - 1])) {
          idx = i;
          exitg1 = true;
        } else {
          i++;
        }
      }
    }

    if (idx == 0) {
      delta_offset = Cobs[0];
    } else {
      delta_offset = Cobs[idx - 1];
      while (idx + 1 < 14) {
        if (delta_offset < Cobs[idx]) {
          delta_offset = Cobs[idx];
        }

        idx++;
      }
    }

    if (delta_offset != 1.0) {
      if (!b_x_0) {
        b_idx = 1;
      } else {
        b_idx = 0;
        i = 2;
        exitg1 = false;
        while ((!exitg1) && (i < 14)) {
          if (!rtIsNaN(Cobs[i - 1])) {
            b_idx = i;
            exitg1 = true;
          } else {
            i++;
          }
        }
      }

      if (b_idx == 0) {
        delta_offset = Cobs[0];
      } else {
        delta_offset = Cobs[b_idx - 1];
        while (b_idx + 1 < 14) {
          if (delta_offset < Cobs[b_idx]) {
            delta_offset = Cobs[b_idx];
          }

          b_idx++;
        }
      }

      for (b_ix = 0; b_ix < 13; b_ix++) {
        Cobs[b_ix] /= delta_offset;
      }
    }
  }

  for (b_ix = 0; b_ix < 13; b_ix++) {
    Clane[b_ix] = Path_col[(b_ix << 2) + 3];
    Cobs_0[b_ix] = Path_col[(b_ix << 2) + 3];
  }

  target_k = std(Cobs_0);
  if (target_k != 0.0) {
    delta_offset = target_k * target_k * 2.0;
    rtb_UnitDelay18 = 2.5066282746310002 * target_k;
    for (ix = 0; ix < 13; ix++) {
      b_ix = 1 + ix;
      for (iy = 0; iy < 13; iy++) {
        Cc_0[iy] = (b_ix - iy) - 1;
      }

      power_dw3x(Cc_0, rtb_forward_length_free);
      for (b_ix = 0; b_ix < 13; b_ix++) {
        Cc_0[b_ix] = -rtb_forward_length_free[b_ix] / delta_offset;
      }

      exp_n(Cc_0);
      for (b_ix = 0; b_ix < 13; b_ix++) {
        Cobs_0[b_ix] = Path_col[(b_ix << 2) + 3] * (Cc_0[b_ix] / rtb_UnitDelay18);
      }

      Clane[ix] = sum_a(Cobs_0);
      if ((1 + ix == 1) && (Path_col[3] == 1.0)) {
        Clane[0] += std::exp(-1.0 / (target_k * target_k * 2.0)) /
          (2.5066282746310002 * target_k);
      } else {
        if ((1 + ix == 13) && (Path_col[51] == 1.0)) {
          Clane[12] += std::exp(-1.0 / (target_k * target_k * 2.0)) /
            (2.5066282746310002 * target_k);
        }
      }
    }

    b_x_0 = rtIsNaN(Clane[0]);
    if (!b_x_0) {
      i = 1;
    } else {
      i = 0;
      b_idx = 2;
      exitg1 = false;
      while ((!exitg1) && (b_idx < 14)) {
        if (!rtIsNaN(Clane[b_idx - 1])) {
          i = b_idx;
          exitg1 = true;
        } else {
          b_idx++;
        }
      }
    }

    if (i == 0) {
      delta_offset = Clane[0];
    } else {
      delta_offset = Clane[i - 1];
      while (i + 1 < 14) {
        if (delta_offset < Clane[i]) {
          delta_offset = Clane[i];
        }

        i++;
      }
    }

    if (delta_offset != 1.0) {
      if (!b_x_0) {
        b_idx = 1;
      } else {
        b_idx = 0;
        i = 2;
        exitg1 = false;
        while ((!exitg1) && (i < 14)) {
          if (!rtIsNaN(Clane[i - 1])) {
            b_idx = i;
            exitg1 = true;
          } else {
            i++;
          }
        }
      }

      if (b_idx == 0) {
        delta_offset = Clane[0];
      } else {
        delta_offset = Clane[b_idx - 1];
        while (b_idx + 1 < 14) {
          if (delta_offset < Clane[b_idx]) {
            delta_offset = Clane[b_idx];
          }

          b_idx++;
        }
      }

      for (b_ix = 0; b_ix < 13; b_ix++) {
        Clane[b_ix] /= delta_offset;
      }
    }
  }

  for (b_ix = 0; b_ix < 11; b_ix++) {
    x[b_ix] = rtDW.UnitDelay5_DSTATE[b_ix] - rtb_TmpSignalConversionAtSFun_e[0];
  }

  power_d(x, K1_0);
  for (b_ix = 0; b_ix < 11; b_ix++) {
    x[b_ix] = rtDW.UnitDelay5_DSTATE[11 + b_ix] -
      rtb_TmpSignalConversionAtSFun_e[1];
  }

  power_d(x, X2);
  for (b_ix = 0; b_ix < 11; b_ix++) {
    b_Path_dis_data[b_ix] = K1_0[b_ix] + X2[b_ix];
  }

  sqrt_l(b_Path_dis_data);
  if (!rtIsNaN(b_Path_dis_data[0])) {
    b_idx = 1;
  } else {
    b_idx = 0;
    i = 2;
    exitg1 = false;
    while ((!exitg1) && (i < 12)) {
      if (!rtIsNaN(b_Path_dis_data[i - 1])) {
        b_idx = i;
        exitg1 = true;
      } else {
        i++;
      }
    }
  }

  if (b_idx == 0) {
    b_idx = 1;
  } else {
    delta_offset = b_Path_dis_data[b_idx - 1];
    for (i = b_idx; i + 1 < 12; i++) {
      if (delta_offset > b_Path_dis_data[i]) {
        delta_offset = b_Path_dis_data[i];
        b_idx = i + 1;
      }
    }
  }

  jj = 12 - b_idx;
  d_ix = -b_idx;
  for (b_ix = 0; b_ix <= d_ix + 11; b_ix++) {
    LastPath_overlap_data[b_ix] = rtDW.UnitDelay5_DSTATE[(b_idx + b_ix) - 1];
  }

  d_ix = -b_idx;
  for (b_ix = 0; b_ix <= d_ix + 11; b_ix++) {
    LastPath_overlap_data[b_ix + jj] = rtDW.UnitDelay5_DSTATE[(b_idx + b_ix) +
      10];
  }

  for (i = 0; i < 13; i++) {
    for (b_ix = 0; b_ix < 11; b_ix++) {
      b_Path_dis_data[b_ix] = X_2[11 * i + b_ix] - rtDW.UnitDelay5_DSTATE[10];
    }

    power_d(b_Path_dis_data, x);
    for (b_ix = 0; b_ix < 11; b_ix++) {
      b_Path_dis_data[b_ix] = Y[11 * i + b_ix] - rtDW.UnitDelay5_DSTATE[21];
    }

    power_d(b_Path_dis_data, K1_0);
    for (b_ix = 0; b_ix < 11; b_ix++) {
      b_Path_dis_data[b_ix] = x[b_ix] + K1_0[b_ix];
    }

    sqrt_l(b_Path_dis_data);
    if (!rtIsNaN(b_Path_dis_data[0])) {
      idx = 0;
    } else {
      idx = -1;
      c_ix = 2;
      exitg1 = false;
      while ((!exitg1) && (c_ix < 12)) {
        if (!rtIsNaN(b_Path_dis_data[c_ix - 1])) {
          idx = c_ix - 1;
          exitg1 = true;
        } else {
          c_ix++;
        }
      }
    }

    if (idx + 1 == 0) {
      idx = 0;
    } else {
      target_k = b_Path_dis_data[idx];
      for (c_ix = idx + 1; c_ix + 1 < 12; c_ix++) {
        if (target_k > b_Path_dis_data[c_ix]) {
          target_k = b_Path_dis_data[c_ix];
          idx = c_ix;
        }
      }
    }

    Path_overlap_size[0] = idx + 1;
    if (0 <= idx) {
      memcpy(&Path_overlap_data[0], &X_2[i * 11], (idx + 1) * sizeof(real_T));
    }

    for (b_ix = 0; b_ix <= idx; b_ix++) {
      Path_overlap_data[b_ix + Path_overlap_size[0]] = Y[11 * i + b_ix];
    }

    if (12 - b_idx >= Path_overlap_size[0]) {
      idx = 13 - (b_idx + Path_overlap_size[0]);
      if (idx > 12 - b_idx) {
        idx = 1;
        ix = 0;
      } else {
        ix = 12 - b_idx;
      }

      b_ix = idx - 1;
      idx = ix - b_ix;
      LastPath_overlap_size_0[0] = idx;
      LastPath_overlap_size_0[1] = 2;
      for (iy = 0; iy < idx; iy++) {
        LastPath_overlap_data_0[iy] = LastPath_overlap_data[b_ix + iy] -
          Path_overlap_data[iy];
      }

      for (iy = 0; iy < idx; iy++) {
        LastPath_overlap_data_0[iy + idx] = LastPath_overlap_data[(b_ix + iy) +
          jj] - Path_overlap_data[iy + Path_overlap_size[0]];
      }

      power_dw3xd(LastPath_overlap_data_0, LastPath_overlap_size_0,
                  Path_overlap_data, Path_overlap_size);
      Path_overlap_size_1[0] = 2;
      Path_overlap_size_1[1] = Path_overlap_size[0];
      d_ix = Path_overlap_size[0];
      for (b_ix = 0; b_ix < d_ix; b_ix++) {
        LastPath_overlap_data_0[b_ix << 1] = Path_overlap_data[b_ix];
        LastPath_overlap_data_0[1 + (b_ix << 1)] = Path_overlap_data[b_ix +
          Path_overlap_size[0]];
      }

      sum_ae(LastPath_overlap_data_0, Path_overlap_size_1, b_Path_dis_data,
             oi_xy_size);
      sqrt_l5(b_Path_dis_data, oi_xy_size);
      d_ix = oi_xy_size[1];
      for (b_ix = 0; b_ix < d_ix; b_ix++) {
        K_11[b_ix] = b_Path_dis_data[oi_xy_size[0] * b_ix];
      }

      idx = oi_xy_size[1];
      iy = oi_xy_size[1];
      if (0 <= idx - 1) {
        memcpy(&Path_dis_data[0], &K_11[0], idx * sizeof(real_T));
      }
    } else {
      ix = 12 - b_idx;
      LastPath_overlap_size[0] = ix;
      LastPath_overlap_size[1] = 2;
      for (b_ix = 0; b_ix < ix; b_ix++) {
        LastPath_overlap_data_0[b_ix] = LastPath_overlap_data[b_ix] -
          Path_overlap_data[b_ix];
      }

      for (b_ix = 0; b_ix < ix; b_ix++) {
        LastPath_overlap_data_0[b_ix + ix] = LastPath_overlap_data[b_ix + jj] -
          Path_overlap_data[b_ix + Path_overlap_size[0]];
      }

      power_dw3xd(LastPath_overlap_data_0, LastPath_overlap_size,
                  Path_overlap_data, Path_overlap_size);
      Path_overlap_size_0[0] = 2;
      Path_overlap_size_0[1] = Path_overlap_size[0];
      d_ix = Path_overlap_size[0];
      for (b_ix = 0; b_ix < d_ix; b_ix++) {
        LastPath_overlap_data_0[b_ix << 1] = Path_overlap_data[b_ix];
        LastPath_overlap_data_0[1 + (b_ix << 1)] = Path_overlap_data[b_ix +
          Path_overlap_size[0]];
      }

      sum_ae(LastPath_overlap_data_0, Path_overlap_size_0, b_Path_dis_data,
             oi_xy_size);
      sqrt_l5(b_Path_dis_data, oi_xy_size);
      d_ix = oi_xy_size[1];
      for (b_ix = 0; b_ix < d_ix; b_ix++) {
        b_Path_dis_data_0[b_ix] = b_Path_dis_data[oi_xy_size[0] * b_ix];
      }

      idx = oi_xy_size[1];
      iy = oi_xy_size[1];
      if (0 <= idx - 1) {
        memcpy(&Path_dis_data[0], &b_Path_dis_data_0[0], idx * sizeof(real_T));
      }
    }

    if (iy > 1) {
      idx = iy;
    } else {
      idx = 1;
    }

    if (mod((real_T)idx) == 0.0) {
      if (iy > 1) {
        idx = iy - 1;
      } else {
        idx = 0;
      }

      oi_xy_size[1] = idx;
      d_ix = idx - 1;
      for (b_ix = 0; b_ix <= d_ix; b_ix++) {
        b_Path_dis_data[b_ix] = 4.0;
      }
    } else {
      if (iy > 1) {
        idx = iy;
      } else {
        idx = 1;
      }

      oi_xy_size[1] = idx;
      d_ix = idx - 1;
      for (b_ix = 0; b_ix <= d_ix; b_ix++) {
        b_Path_dis_data[b_ix] = 4.0;
      }
    }

    b_Path_dis_data[0] = 1.0;
    b_Path_dis_data[oi_xy_size[1] - 1] = 1.0;
    if (3 > oi_xy_size[1] - 2) {
      ix = 1;
      n = 1;
      c_ix = 0;
    } else {
      ix = 3;
      n = 2;
      c_ix = oi_xy_size[1] - 2;
    }

    idx = div_nde_s32_floor((int8_T)c_ix - ix, n);
    for (b_ix = 0; b_ix <= idx; b_ix++) {
      p_data[b_ix] = (int8_T)((int8_T)((int8_T)(n * (int8_T)b_ix) + ix) - 1);
    }

    for (b_ix = 0; b_ix <= idx; b_ix++) {
      b_Path_dis_data[p_data[b_ix]] = 2.0;
    }

    delta_offset = 0.0;
    for (b_ix = 0; b_ix < oi_xy_size[1]; b_ix++) {
      delta_offset += b_Path_dis_data[b_ix] * Path_dis_data[b_ix];
    }

    if (!(iy > 1)) {
      iy = 1;
    }

    Cc_0[i] = rtb_J_out_k[i] / 11.0 * delta_offset / 3.0 / (rtb_J_out_k[i] *
      (real_T)iy / 11.0);
  }

  for (i = 0; i < 13; i++) {
    rtb_U_c_l[i] = 1.0;
    rtb_safety_level_all_p[i] = 0.0;
    rtb_forward_length_free_o[i] = seg_heading;
  }

  if ((rtU.Freespace_mode == 1.0) || (rtU.Freespace_mode == 2.0)) {
    for (idx = 0; idx < 13; idx++) {
      FreespaceDetectCollision(rtU.Freespace, &rtb_XP_i[6 * idx], &rtb_YP_g[6 *
        idx], rtb_TmpSignalConversionAtSFun_e, seg_heading, rtU.safe_range,
        rtConstP.pooled8, rtConstP.pooled7, &rtb_U_c_l[idx],
        &rtb_safety_level_all_p[idx], &rtb_forward_length_free_o[idx]);
    }
  }

  abs_a(K, XY_difflen);
  for (idx = 0; idx < 13; idx++) {
    rtb_forward_length_free[idx] = XY_difflen[11 * idx];
    for (c_ix = 0; c_ix < 10; c_ix++) {
      delta_offset = rtb_forward_length_free[idx];
      b_ix = (11 * idx + c_ix) + 1;
      if ((!rtIsNaN(XY_difflen[b_ix])) && (rtIsNaN(rtb_forward_length_free[idx])
           || (rtb_forward_length_free[idx] < XY_difflen[b_ix]))) {
        delta_offset = XY_difflen[b_ix];
      }

      rtb_forward_length_free[idx] = delta_offset;
    }
  }

  abs_a(K, XY_difflen);
  for (idx = 0; idx < 13; idx++) {
    rtb_forward_length_free_2[idx] = XY_difflen[11 * idx];
    for (b_idx = 0; b_idx < 10; b_idx++) {
      delta_offset = rtb_forward_length_free_2[idx];
      b_ix = (11 * idx + b_idx) + 1;
      if ((!rtIsNaN(XY_difflen[b_ix])) && (rtIsNaN(rtb_forward_length_free_2[idx])
           || (rtb_forward_length_free_2[idx] < XY_difflen[b_ix]))) {
        delta_offset = XY_difflen[b_ix];
      }

      rtb_forward_length_free_2[idx] = delta_offset;
    }

    rtb_forward_length_free_2[idx] *= 10.0;
  }

  if (!rtIsNaN(rtb_forward_length_free_2[0])) {
    b_idx = 1;
  } else {
    b_idx = 0;
    i = 2;
    exitg1 = false;
    while ((!exitg1) && (i < 14)) {
      if (!rtIsNaN(rtb_forward_length_free_2[i - 1])) {
        b_idx = i;
        exitg1 = true;
      } else {
        i++;
      }
    }
  }

  if (b_idx == 0) {
    target_k = rtb_forward_length_free_2[0];
  } else {
    target_k = rtb_forward_length_free_2[b_idx - 1];
    while (b_idx + 1 < 14) {
      if (target_k < rtb_forward_length_free_2[b_idx]) {
        target_k = rtb_forward_length_free_2[b_idx];
      }

      b_idx++;
    }
  }

  abs_a(K_1, XY_difflen);
  for (idx = 0; idx < 13; idx++) {
    rtb_forward_length_free_2[idx] = XY_difflen[11 * idx];
    for (b_idx = 0; b_idx < 10; b_idx++) {
      delta_offset = rtb_forward_length_free_2[idx];
      b_ix = (11 * idx + b_idx) + 1;
      if ((!rtIsNaN(XY_difflen[b_ix])) && (rtIsNaN(rtb_forward_length_free_2[idx])
           || (rtb_forward_length_free_2[idx] < XY_difflen[b_ix]))) {
        delta_offset = XY_difflen[b_ix];
      }

      rtb_forward_length_free_2[idx] = delta_offset;
    }
  }

  abs_a(K_1, XY_difflen);
  for (idx = 0; idx < 13; idx++) {
    Cobs_0[idx] = XY_difflen[11 * idx];
    for (b_idx = 0; b_idx < 10; b_idx++) {
      delta_offset = Cobs_0[idx];
      if ((!rtIsNaN(XY_difflen[(11 * idx + b_idx) + 1])) && (rtIsNaN(Cobs_0[idx])
           || (Cobs_0[idx] < XY_difflen[(11 * idx + b_idx) + 1]))) {
        delta_offset = XY_difflen[(11 * idx + b_idx) + 1];
      }

      Cobs_0[idx] = delta_offset;
    }

    Cobs_0[idx] *= 10.0;
  }

  if (!rtIsNaN(Cobs_0[0])) {
    idx = 1;
  } else {
    idx = 0;
    i = 2;
    exitg1 = false;
    while ((!exitg1) && (i < 14)) {
      if (!rtIsNaN(Cobs_0[i - 1])) {
        idx = i;
        exitg1 = true;
      } else {
        i++;
      }
    }
  }

  if (idx == 0) {
    delta_offset = Cobs_0[0];
  } else {
    delta_offset = Cobs_0[idx - 1];
    while (idx + 1 < 14) {
      if (delta_offset < Cobs_0[idx]) {
        delta_offset = Cobs_0[idx];
      }

      idx++;
    }
  }

  if (!rtIsNaN(offset[0])) {
    idx = 1;
  } else {
    idx = 0;
    c_ix = 2;
    exitg1 = false;
    while ((!exitg1) && (c_ix < 14)) {
      if (!rtIsNaN(offset[c_ix - 1])) {
        idx = c_ix;
        exitg1 = true;
      } else {
        c_ix++;
      }
    }
  }

  if (idx == 0) {
    rtb_UnitDelay18 = offset[0];
  } else {
    rtb_UnitDelay18 = offset[idx - 1];
    while (idx + 1 < 14) {
      if (rtb_UnitDelay18 < offset[idx]) {
        rtb_UnitDelay18 = offset[idx];
      }

      idx++;
    }
  }

  b_x_0 = rtIsNaN(Cc_0[0]);
  if (!b_x_0) {
    idx = 1;
  } else {
    idx = 0;
    i = 2;
    exitg1 = false;
    while ((!exitg1) && (i < 14)) {
      if (!rtIsNaN(Cc_0[i - 1])) {
        idx = i;
        exitg1 = true;
      } else {
        i++;
      }
    }
  }

  if (idx == 0) {
    y = Cc_0[0];
  } else {
    y = Cc_0[idx - 1];
    while (idx + 1 < 14) {
      if (y < Cc_0[idx]) {
        y = Cc_0[idx];
      }

      idx++;
    }
  }

  if (!(y == 0.0)) {
    if (!b_x_0) {
      idx = 1;
    } else {
      idx = 0;
      i = 2;
      exitg1 = false;
      while ((!exitg1) && (i < 14)) {
        if (!rtIsNaN(Cc_0[i - 1])) {
          idx = i;
          exitg1 = true;
        } else {
          i++;
        }
      }
    }

    if (idx == 0) {
      y = Cc_0[0];
    } else {
      y = Cc_0[idx - 1];
      while (idx + 1 < 14) {
        if (y < Cc_0[idx]) {
          y = Cc_0[idx];
        }

        idx++;
      }
    }

    for (b_ix = 0; b_ix < 13; b_ix++) {
      Cc_0[b_ix] /= y;
    }
  }

  for (b_ix = 0; b_ix < 13; b_ix++) {
    rtb_J_out_k[b_ix] = (((((rtb_forward_length_free[b_ix] * 10.0 / target_k *
      rtU.W_1[1] + rtb_J_out_k[b_ix] / seg_heading * rtU.W_1[0]) +
      rtb_forward_length_free_2[b_ix] * 10.0 / delta_offset * rtU.W_1[2]) +
      offset[b_ix] / rtb_UnitDelay18 * count) + rtU.W_1[3] * Cobs[b_ix]) +
                         rtU.W_1[4] * Cc_0[b_ix]) + rtU.W_1[5] * Clane[b_ix];
  }

  // SignalConversion: '<S12>/TmpSignal ConversionAt SFunction Inport5' incorporates:
  //   Gain: '<S2>/Gain4'
  //   MATLAB Function: '<S2>/DynamicPathPlanning1'

  rtb_TmpSignalConversionAtSFun_e[0] = rtb_X[0];
  rtb_TmpSignalConversionAtSFun_e[1] = rtb_X[1];
  rtb_TmpSignalConversionAtSFun_e[2] = 0.017453292519943295 * ajj;

  // MATLAB Function: '<S2>/DynamicPathPlanning1' incorporates:
  //   Constant: '<S2>/Constant14'
  //   Constant: '<S2>/Constant4'
  //   Inport: '<Root>/BB_num'
  //   Inport: '<Root>/Freespace'
  //   Inport: '<Root>/Freespace_mode'
  //   Inport: '<Root>/W_2'
  //   Inport: '<Root>/safe_range'
  //   MATLAB Function: '<S2>/DynamicPathPlanning'
  //   MATLAB Function: '<S2>/EndPointDecision'
  //   MATLAB Function: '<S2>/EndPointDecision1'
  //   UnitDelay: '<S2>/Unit Delay6'

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    x_data[b_ix] = (rtb_Forward_Static_Path_id_i[rtDW.SFunction_DIMS4_l - 1] ==
                    rtDW.Static_Path_0[b_ix]);
  }

  i = rtDW.SFunction_DIMS4_h[0] - 1;
  b_idx = 0;
  for (c_ix = 0; c_ix <= i; c_ix++) {
    if (x_data[c_ix]) {
      b_idx++;
    }
  }

  ix = b_idx;
  b_idx = 0;
  for (c_ix = 0; c_ix <= i; c_ix++) {
    if (x_data[c_ix]) {
      t_data_0[b_idx] = c_ix + 1;
      b_idx++;
    }
  }

  for (b_ix = 0; b_ix < ix; b_ix++) {
    Forward_Static_Path_id_0_data[b_ix] = rtDW.Static_Path_0
      [(rtDW.SFunction_DIMS4_h[0] * 7 + t_data_0[b_ix]) - 1] *
      3.1415926535897931;
  }

  for (b_ix = 0; b_ix < ix; b_ix++) {
    end_heading_0_data[b_ix] = Forward_Static_Path_id_0_data[b_ix] / 180.0;
  }

  count = Forward_Static_Path_id_0_data[0] / 180.0;
  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    x_data[b_ix] = (rtb_Forward_Static_Path_id_i[rtDW.SFunction_DIMS4_l - 1] ==
                    rtDW.Static_Path_0[b_ix]);
  }

  c_ix = 0;
  for (b_idx = 0; b_idx < rtDW.SFunction_DIMS4_h[0]; b_idx++) {
    if (x_data[b_idx]) {
      u_data_0[c_ix] = b_idx + 1;
      c_ix++;
    }
  }

  d_ix = rtDW.SFunction_DIMS4_h[0];
  for (b_ix = 0; b_ix < d_ix; b_ix++) {
    x_data[b_ix] = (rtb_Forward_Static_Path_id_i[rtDW.SFunction_DIMS4_l - 1] ==
                    rtDW.Static_Path_0[b_ix]);
  }

  b_idx = 0;
  for (c_ix = 0; c_ix < rtDW.SFunction_DIMS4_h[0]; c_ix++) {
    if (x_data[c_ix]) {
      v_data_0[b_idx] = c_ix + 1;
      b_idx++;
    }
  }

  delta_offset = rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_h[0] * 10 + v_data_0[0])
    - 1] / 4.0;
  target_k = delta_offset * 2.0;
  yy_idx_0 = delta_offset * 3.0;
  rtb_UnitDelay18 = delta_offset * 4.0;
  y = delta_offset * 5.0;
  offset_6 = delta_offset * 6.0;
  G2splines_e(ang_1, c, dist_op_data[0], rtDW.Static_Path_0[(u_data[0] +
    rtDW.SFunction_DIMS4_h[0] * 13) - 1], End_x + offset_6 * std::cos(count +
    1.5707963267948966), seg_direction_tmp_tmp + offset_6 * std::sin(count +
    1.5707963267948966), end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0]
    + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length, x, b_Path_dis_data,
              XP1, YP1, K1_0, K_11, &Cobs[0]);
  G2splines_e(x_endpoint2, y_endpoint2, dist_op_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], End_x + y *
              std::cos(count + 1.5707963267948966), seg_direction_tmp_tmp + y *
              std::sin(count + 1.5707963267948966), end_heading_0_data[0],
              rtDW.Static_Path_0[(u_data_0[0] + rtDW.SFunction_DIMS4_h[0] * 13)
              - 1], total_length, X2, Y2, XP2, YP2, K1_0, K_11, &Cobs[1]);
  G2splines_e(x_endpoint3, y_endpoint3, dist_op_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], End_x +
              rtb_UnitDelay18 * std::cos(count + 1.5707963267948966),
              seg_direction_tmp_tmp + rtb_UnitDelay18 * std::sin(count +
    1.5707963267948966), end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0]
    + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length, X3, Y3, XP3, YP3, K1_0,
              K_11, &Cobs[2]);
  G2splines_e(x_endpoint4, y_endpoint4, dist_op_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], End_x +
              yy_idx_0 * std::cos(count + 1.5707963267948966),
              seg_direction_tmp_tmp + yy_idx_0 * std::sin(count +
    1.5707963267948966), end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0]
    + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length, X4, Y4, XP4, YP4, K1_0,
              K_11, &Cobs[3]);
  G2splines_e(x_endpoint5, y_endpoint5, dist_op_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], End_x +
              target_k * std::cos(count + 1.5707963267948966),
              seg_direction_tmp_tmp + target_k * std::sin(count +
    1.5707963267948966), end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0]
    + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length, X5, Y5, XP5, YP5, K1_0,
              K_11, &Cobs[4]);
  G2splines_e(x_endpoint6, y_endpoint6, dist_op_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], End_x +
              delta_offset * std::cos(count + 1.5707963267948966),
              seg_direction_tmp_tmp + delta_offset * std::sin(count +
    1.5707963267948966), end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0]
    + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length, X6, Y6, XP6, YP6, K1_0,
              K_11, &Cobs[5]);
  G2splines_e(seg_direction, head_err, dist_op_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], End_x,
              seg_direction_tmp_tmp, end_heading_0_data[0], rtDW.Static_Path_0
              [(u_data_0[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length,
              X7, Y7, XP7, YP7, K1_0, K_11, &Cobs[6]);
  G2splines_e(x_endpoint8, y_endpoint8, dist_op_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], End_x +
              delta_offset * std::cos(count - 1.5707963267948966),
              seg_direction_tmp_tmp + delta_offset * std::sin(count -
    1.5707963267948966), end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0]
    + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length, X8, Y8, XP8, YP8, K1_0,
              K_11, &Cobs[7]);
  G2splines_e(x_endpoint9, y_endpoint9, dist_op_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], End_x +
              target_k * std::cos(count - 1.5707963267948966),
              seg_direction_tmp_tmp + target_k * std::sin(count -
    1.5707963267948966), end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0]
    + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length, X9, Y9, XP9, YP9, K1_0,
              K_11, &Cobs[8]);
  G2splines_e(x_endpoint10, y_endpoint10, dist_op_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], End_x +
              yy_idx_0 * std::cos(count - 1.5707963267948966),
              seg_direction_tmp_tmp + yy_idx_0 * std::sin(count -
    1.5707963267948966), end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0]
    + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length, X10, Y10, XP10, YP10,
              K1_0, K_11, &Cobs[9]);
  G2splines_e(x_endpoint11, y_endpoint11, dist_op_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], End_x +
              rtb_UnitDelay18 * std::cos(count - 1.5707963267948966),
              seg_direction_tmp_tmp + rtb_UnitDelay18 * std::sin(count -
    1.5707963267948966), end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0]
    + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length, X11, Y11, XP11, YP11,
              K1_0, K_11, &Cobs[10]);
  G2splines_e(x_endpoint12, y_endpoint12, dist_op_data[0], rtDW.Static_Path_0
              [(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) - 1], End_x + y *
              std::cos(count - 1.5707963267948966), seg_direction_tmp_tmp + y *
              std::sin(count - 1.5707963267948966), end_heading_0_data[0],
              rtDW.Static_Path_0[(u_data_0[0] + rtDW.SFunction_DIMS4_h[0] * 13)
              - 1], total_length, X12, Y12, XP12, YP12, K1_0, K_11, &Cobs[11]);
  G2splines_e(x_endpoint13, xy_end_point_idx_25, dist_op_data[0],
              rtDW.Static_Path_0[(u_data[0] + rtDW.SFunction_DIMS4_h[0] * 13) -
              1], End_x + offset_6 * std::cos(count - 1.5707963267948966),
              seg_direction_tmp_tmp + offset_6 * std::sin(count -
    1.5707963267948966), end_heading_0_data[0], rtDW.Static_Path_0[(u_data_0[0]
    + rtDW.SFunction_DIMS4_h[0] * 13) - 1], total_length, K1_0, K_11, XP13, YP13,
              K13, K_113, &Cobs[12]);
  for (b_ix = 0; b_ix < 11; b_ix++) {
    X_2[b_ix] = x[b_ix];
    X_2[b_ix + 11] = X2[b_ix];
    X_2[b_ix + 22] = X3[b_ix];
    X_2[b_ix + 33] = X4[b_ix];
    X_2[b_ix + 44] = X5[b_ix];
    X_2[b_ix + 55] = X6[b_ix];
    X_2[b_ix + 66] = X7[b_ix];
    X_2[b_ix + 77] = X8[b_ix];
    X_2[b_ix + 88] = X9[b_ix];
    X_2[b_ix + 99] = X10[b_ix];
    X_2[b_ix + 110] = X11[b_ix];
    X_2[b_ix + 121] = X12[b_ix];
    X_2[b_ix + 132] = K1_0[b_ix];
    Y[b_ix] = b_Path_dis_data[b_ix];
    Y[b_ix + 11] = Y2[b_ix];
    Y[b_ix + 22] = Y3[b_ix];
    Y[b_ix + 33] = Y4[b_ix];
    Y[b_ix + 44] = Y5[b_ix];
    Y[b_ix + 55] = Y6[b_ix];
    Y[b_ix + 66] = Y7[b_ix];
    Y[b_ix + 77] = Y8[b_ix];
    Y[b_ix + 88] = Y9[b_ix];
    Y[b_ix + 99] = Y10[b_ix];
    Y[b_ix + 110] = Y11[b_ix];
    Y[b_ix + 121] = Y12[b_ix];
    Y[b_ix + 132] = K_11[b_ix];
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
  for (b_ix = 0; b_ix < 5; b_ix++) {
    Path_col[3 + ((8 + b_ix) << 2)] = 1.0;
  }

  Path_col[3] = 1.0;
  Path_col[51] = 1.0;
  if ((rtU.Freespace_mode == 0.0) || (rtU.Freespace_mode == 2.0)) {
    memcpy(&OBXY_EL[0], &rtb_V_boundingbox[0], 400U * sizeof(real_T));
    for (n = 0; n < (int32_T)rtU.BB_num; n++) {
      delta_offset = (1.0 + (real_T)n) * 2.0;
      i = (int32_T)(delta_offset + -1.0);
      idx = i - 1;
      OBXY_EL[idx] = ((rtb_V_boundingbox[idx] - rtb_V_boundingbox[i + 99]) *
                      0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0
        + -1.0) - 1]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 +
        -1.0) - 1] - rtb_V_boundingbox[i + 299]) * 0.3;
      idx = (int32_T)delta_offset;
      b_ix = idx - 1;
      OBXY_EL[b_ix] = ((rtb_V_boundingbox[b_ix] - rtb_V_boundingbox[idx + 99]) *
                       0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
        2.0) - 1]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) - 1]
                      - rtb_V_boundingbox[idx + 299]) * 0.3;
      OBXY_EL[(int32_T)(delta_offset + -1.0) + 99] = ((rtb_V_boundingbox
        [(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 99] - rtb_V_boundingbox
        [(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) - 1]) * 0.15 +
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 99]) +
        (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 99] -
         rtb_V_boundingbox[i + 199]) * 0.3;
      OBXY_EL[(int32_T)delta_offset + 99] = ((rtb_V_boundingbox[(int32_T)((1.0 +
        (real_T)n) * 2.0) + 99] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
        2.0) - 1]) * 0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0)
        + 99]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) + 99] -
                  rtb_V_boundingbox[idx + 199]) * 0.3;
      OBXY_EL[(int32_T)(delta_offset + -1.0) + 199] = ((rtb_V_boundingbox
        [(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 199] - rtb_V_boundingbox
        [(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 299]) * 0.15 +
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 199]) +
        (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 199] -
         rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 99]) *
        0.3;
      OBXY_EL[(int32_T)delta_offset + 199] = ((rtb_V_boundingbox[(int32_T)((1.0
        + (real_T)n) * 2.0) + 199] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
        n) * 2.0) + 299]) * 0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n)
        * 2.0) + 199]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) +
                          199] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
        2.0) + 99]) * 0.3;
      OBXY_EL[(int32_T)(delta_offset + -1.0) + 299] = ((rtb_V_boundingbox
        [(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 299] - rtb_V_boundingbox
        [(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 199]) * 0.15 +
        rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 299]) +
        (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) + 299] -
         rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0 + -1.0) - 1]) * 0.3;
      OBXY_EL[(int32_T)delta_offset + 299] = ((rtb_V_boundingbox[(int32_T)((1.0
        + (real_T)n) * 2.0) + 299] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)
        n) * 2.0) + 199]) * 0.15 + rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n)
        * 2.0) + 299]) + (rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) * 2.0) +
                          299] - rtb_V_boundingbox[(int32_T)((1.0 + (real_T)n) *
        2.0) - 1]) * 0.3;
    }

    for (b_ix = 0; b_ix < 13; b_ix++) {
      for (iy = 0; iy < 10; iy++) {
        i = 11 * b_ix + iy;
        delta_offset = X_2[i + 1] - X_2[i];
        X_diff[iy + 11 * b_ix] = delta_offset;
        X_diff_0[iy + 10 * b_ix] = delta_offset;
      }

      idx = 10 + 11 * b_ix;
      X_diff[idx] = X_diff_0[10 * b_ix + 9];
      for (iy = 0; iy < 10; iy++) {
        i = 11 * b_ix + iy;
        delta_offset = Y[i + 1] - Y[i];
        Y_diff[iy + 11 * b_ix] = delta_offset;
        X_diff_0[iy + 10 * b_ix] = delta_offset;
      }

      Y_diff[idx] = X_diff_0[10 * b_ix + 9];
    }

    power_dw(X_diff, XY_difflen);
    power_dw(Y_diff, Path_vehFLY);
    for (b_ix = 0; b_ix < 143; b_ix++) {
      Path_vehFLX[b_ix] = XY_difflen[b_ix] + Path_vehFLY[b_ix];
    }

    power_dw3(Path_vehFLX, XY_difflen);
    for (b_ix = 0; b_ix < 143; b_ix++) {
      delta_offset = X_diff[b_ix] / XY_difflen[b_ix];
      target_k = Y_diff[b_ix] / XY_difflen[b_ix];
      yy_idx_0 = 1.1 * -target_k + X_2[b_ix];
      Path_vehFLX[b_ix] = yy_idx_0 + 1.4000000000000001 * delta_offset;
      rtb_UnitDelay18 = 1.1 * delta_offset + Y[b_ix];
      Path_vehFLY[b_ix] = rtb_UnitDelay18 + 1.4000000000000001 * target_k;
      y = X_2[b_ix] - 1.1 * -target_k;
      Path_vehFRX[b_ix] = y + 1.4000000000000001 * delta_offset;
      offset_6 = Y[b_ix] - 1.1 * delta_offset;
      Path_vehFRY[b_ix] = offset_6 + 1.4000000000000001 * target_k;
      Path_vehRLX[b_ix] = yy_idx_0 - 5.6000000000000005 * delta_offset;
      Path_vehRLY[b_ix] = rtb_UnitDelay18 - 5.6000000000000005 * target_k;
      Path_vehRRX[b_ix] = y - 5.6000000000000005 * delta_offset;
      Path_vehRRY[b_ix] = offset_6 - 5.6000000000000005 * target_k;
      X_diff[b_ix] = delta_offset;
      XY_difflen[b_ix] = -target_k;
      Y_diff[b_ix] = target_k;
    }

    for (ix = 0; ix < 13; ix++) {
      Path_col[ix << 2] = 0.0;
      if (!(Path_col[(ix << 2) + 3] == 1.0)) {
        c_ix = 0;
        exitg1 = false;
        while ((!exitg1) && (c_ix < 11)) {
          d_ix = 11 * ix + c_ix;
          OBXY_m[0] = Path_vehFLX[d_ix];
          OBXY_m[2] = Path_vehFRX[d_ix];
          OBXY_m[4] = Path_vehRLX[d_ix];
          OBXY_m[6] = Path_vehRRX[d_ix];
          OBXY_m[1] = Path_vehFLY[d_ix];
          OBXY_m[3] = Path_vehFRY[d_ix];
          OBXY_m[5] = Path_vehRLY[d_ix];
          OBXY_m[7] = Path_vehRRY[d_ix];
          i = 0;
          exitg3 = false;
          while ((!exitg3) && (i <= (int32_T)rtU.BB_num - 1)) {
            delta_offset = (1.0 + (real_T)i) * 2.0;
            idx = (int32_T)(delta_offset + -1.0);
            rtb_UnitDelay18 = OBXY_EL[idx + 99] - OBXY_EL[idx - 1];
            target_k = std::sqrt(rtb_UnitDelay18 * rtb_UnitDelay18 +
                                 rtb_UnitDelay18 * rtb_UnitDelay18);
            b_ix = (int32_T)delta_offset;
            rtb_Oi_near_l[0] = -(OBXY_EL[b_ix + 99] - OBXY_EL[b_ix - 1]) /
              target_k;
            rtb_Oi_near_l[1] = rtb_UnitDelay18 / target_k;
            yy_idx_0 = OBXY_EL[b_ix + 199] - OBXY_EL[(int32_T)((1.0 + (real_T)i)
              * 2.0) + 99];
            rtb_UnitDelay18 = OBXY_EL[idx + 199] - OBXY_EL[(int32_T)((1.0 +
              (real_T)i) * 2.0 + -1.0) + 99];
            offset_6 = std::sqrt(yy_idx_0 * yy_idx_0 + rtb_UnitDelay18 *
                                 rtb_UnitDelay18);
            yy_idx_0 = -yy_idx_0 / offset_6;
            target_k = rtb_UnitDelay18 / offset_6;
            rtb_Oi_near_o[0] = rtb_Oi_near_l[0];
            rtb_Oi_near_o[1] = yy_idx_0;
            rtb_Oi_near_o[4] = rtb_Oi_near_l[1];
            rtb_Oi_near_o[5] = target_k;
            rtb_Oi_near_o[2] = X_diff[d_ix];
            rtb_Oi_near_o[6] = Y_diff[d_ix];
            rtb_Oi_near_o[3] = XY_difflen[d_ix];
            rtb_Oi_near_o[7] = X_diff[11 * ix + c_ix];
            rtb_Oi_near_o_0[0] = rtb_Oi_near_l[0];
            rtb_Oi_near_o_0[1] = yy_idx_0;
            rtb_Oi_near_o_0[4] = rtb_Oi_near_l[1];
            rtb_Oi_near_o_0[5] = target_k;
            rtb_Oi_near_o_0[2] = X_diff[11 * ix + c_ix];
            rtb_Oi_near_o_0[6] = Y_diff[11 * ix + c_ix];
            rtb_Oi_near_o_0[3] = XY_difflen[11 * ix + c_ix];
            rtb_Oi_near_o_0[7] = X_diff[11 * ix + c_ix];
            for (b_ix = 0; b_ix < 4; b_ix++) {
              for (iy = 0; iy < 4; iy++) {
                proj_veh[b_ix + (iy << 2)] = 0.0;
                proj_veh[b_ix + (iy << 2)] += OBXY_m[iy << 1] *
                  rtb_Oi_near_o[b_ix];
                proj_veh[b_ix + (iy << 2)] += OBXY_m[(iy << 1) + 1] *
                  rtb_Oi_near_o[b_ix + 4];
              }

              OBXY_EL_0[b_ix << 1] = OBXY_EL[((int32_T)(delta_offset + -1.0) +
                100 * b_ix) - 1];
              OBXY_EL_0[1 + (b_ix << 1)] = OBXY_EL[(100 * b_ix + (int32_T)
                delta_offset) - 1];
            }

            for (b_idx = 0; b_idx < 4; b_idx++) {
              for (b_ix = 0; b_ix < 4; b_ix++) {
                proj_ob[b_idx + (b_ix << 2)] = 0.0;
                proj_ob[b_idx + (b_ix << 2)] += OBXY_EL_0[b_ix << 1] *
                  rtb_Oi_near_o_0[b_idx];
                proj_ob[b_idx + (b_ix << 2)] += OBXY_EL_0[(b_ix << 1) + 1] *
                  rtb_Oi_near_o_0[b_idx + 4];
              }

              K1[b_idx] = proj_veh[b_idx];
            }

            delta_offset = proj_veh[0];
            target_k = proj_veh[1];
            yy_idx_0 = proj_veh[2];
            rtb_UnitDelay18 = proj_veh[3];
            for (idx = 0; idx < 3; idx++) {
              if ((!rtIsNaN(proj_veh[(idx + 1) << 2])) && (rtIsNaN(K1[0]) ||
                   (K1[0] > proj_veh[(idx + 1) << 2]))) {
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

              y = delta_offset;
              if ((!rtIsNaN(proj_veh[(idx + 1) << 2])) && (rtIsNaN(delta_offset)
                   || (delta_offset < proj_veh[(idx + 1) << 2]))) {
                y = proj_veh[(idx + 1) << 2];
              }

              delta_offset = y;
              y = target_k;
              if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 1])) && (rtIsNaN
                   (target_k) || (target_k < proj_veh[((idx + 1) << 2) + 1]))) {
                y = proj_veh[((idx + 1) << 2) + 1];
              }

              target_k = y;
              y = yy_idx_0;
              if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 2])) && (rtIsNaN
                   (yy_idx_0) || (yy_idx_0 < proj_veh[((idx + 1) << 2) + 2]))) {
                y = proj_veh[((idx + 1) << 2) + 2];
              }

              yy_idx_0 = y;
              y = rtb_UnitDelay18;
              if ((!rtIsNaN(proj_veh[((idx + 1) << 2) + 3])) && (rtIsNaN
                   (rtb_UnitDelay18) || (rtb_UnitDelay18 < proj_veh[((idx + 1) <<
                     2) + 3]))) {
                y = proj_veh[((idx + 1) << 2) + 3];
              }

              rtb_UnitDelay18 = y;
            }

            minmax_veh[0] = K1[0];
            minmax_veh[4] = delta_offset;
            minmax_veh[1] = K1[1];
            minmax_veh[5] = target_k;
            minmax_veh[2] = K1[2];
            minmax_veh[6] = yy_idx_0;
            minmax_veh[3] = K1[3];
            minmax_veh[7] = rtb_UnitDelay18;
            K1[0] = proj_ob[0];
            K1[1] = proj_ob[1];
            K1[2] = proj_ob[2];
            K1[3] = proj_ob[3];
            delta_offset = proj_ob[0];
            target_k = proj_ob[1];
            yy_idx_0 = proj_ob[2];
            rtb_UnitDelay18 = proj_ob[3];
            for (idx = 0; idx < 3; idx++) {
              if ((!rtIsNaN(proj_ob[(idx + 1) << 2])) && (rtIsNaN(K1[0]) || (K1
                    [0] > proj_ob[(idx + 1) << 2]))) {
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

              y = delta_offset;
              if ((!rtIsNaN(proj_ob[(idx + 1) << 2])) && (rtIsNaN(delta_offset) ||
                   (delta_offset < proj_ob[(idx + 1) << 2]))) {
                y = proj_ob[(idx + 1) << 2];
              }

              delta_offset = y;
              y = target_k;
              if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 1])) && (rtIsNaN(target_k)
                   || (target_k < proj_ob[((idx + 1) << 2) + 1]))) {
                y = proj_ob[((idx + 1) << 2) + 1];
              }

              target_k = y;
              y = yy_idx_0;
              if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 2])) && (rtIsNaN(yy_idx_0)
                   || (yy_idx_0 < proj_ob[((idx + 1) << 2) + 2]))) {
                y = proj_ob[((idx + 1) << 2) + 2];
              }

              yy_idx_0 = y;
              y = rtb_UnitDelay18;
              if ((!rtIsNaN(proj_ob[((idx + 1) << 2) + 3])) && (rtIsNaN
                   (rtb_UnitDelay18) || (rtb_UnitDelay18 < proj_ob[((idx + 1) <<
                     2) + 3]))) {
                y = proj_ob[((idx + 1) << 2) + 3];
              }

              rtb_UnitDelay18 = y;
            }

            minmax_obj[0] = K1[0];
            minmax_obj[4] = delta_offset;
            minmax_obj[1] = K1[1];
            minmax_obj[5] = target_k;
            minmax_obj[2] = K1[2];
            minmax_obj[6] = yy_idx_0;
            minmax_obj[3] = K1[3];
            minmax_obj[7] = rtb_UnitDelay18;
            n = 0;
            exitg4 = false;
            while ((!exitg4) && (n < 4)) {
              if (minmax_veh[n] > minmax_obj[4 + n]) {
                Path_col[ix << 2] = 0.0;
                exitg4 = true;
              } else if (minmax_veh[4 + n] < minmax_obj[n]) {
                Path_col[ix << 2] = 0.0;
                exitg4 = true;
              } else {
                Path_col[ix << 2] = 1.0;
                n++;
              }
            }

            if (Path_col[ix << 2] == 1.0) {
              Path_col[2 + (ix << 2)] = 1.0 + (real_T)i;
              exitg3 = true;
            } else {
              i++;
            }
          }

          if (Path_col[ix << 2] == 1.0) {
            Path_col[1 + (ix << 2)] = 1.0 + (real_T)c_ix;
            exitg1 = true;
          } else {
            c_ix++;
          }
        }
      }
    }
  }

  seg_direction_tmp_tmp = target_k_tmp / total_length * 10.0;
  for (idx = 0; idx < 13; idx++) {
    delta_offset = Path_col[idx << 2];
    if (Path_col[(idx << 2) + 1] > seg_direction_tmp_tmp) {
      delta_offset = 0.0;
    }

    offset[idx] = delta_offset;
    Cobs_0[idx] = delta_offset;
  }

  target_k = std(Cobs_0);
  if (target_k != 0.0) {
    rtb_UnitDelay18 = target_k * target_k * 2.0;
    ang_1 = 2.5066282746310002 * target_k;
    for (ix = 0; ix < 13; ix++) {
      b_ix = 1 + ix;
      for (iy = 0; iy < 13; iy++) {
        Cc_0[iy] = (b_ix - iy) - 1;
      }

      power_dw3x(Cc_0, rtb_forward_length_free);
      for (b_ix = 0; b_ix < 13; b_ix++) {
        Cc_0[b_ix] = -rtb_forward_length_free[b_ix] / rtb_UnitDelay18;
      }

      exp_n(Cc_0);
      for (b_ix = 0; b_ix < 13; b_ix++) {
        Clane[b_ix] = Cc_0[b_ix] / ang_1 * Cobs_0[b_ix];
      }

      offset[ix] = sum_a(Clane);
      if ((1 + ix == 1) && (Cobs_0[0] == 1.0)) {
        offset[0] += std::exp(-1.0 / (target_k * target_k * 2.0)) /
          (2.5066282746310002 * target_k);
      } else {
        if ((1 + ix == 13) && (Cobs_0[12] == 1.0)) {
          offset[12] += std::exp(-1.0 / (target_k * target_k * 2.0)) /
            (2.5066282746310002 * target_k);
        }
      }
    }

    b_x_0 = rtIsNaN(offset[0]);
    if (!b_x_0) {
      idx = 1;
    } else {
      idx = 0;
      i = 2;
      exitg1 = false;
      while ((!exitg1) && (i < 14)) {
        if (!rtIsNaN(offset[i - 1])) {
          idx = i;
          exitg1 = true;
        } else {
          i++;
        }
      }
    }

    if (idx == 0) {
      delta_offset = offset[0];
    } else {
      delta_offset = offset[idx - 1];
      while (idx + 1 < 14) {
        if (delta_offset < offset[idx]) {
          delta_offset = offset[idx];
        }

        idx++;
      }
    }

    if (delta_offset != 1.0) {
      if (!b_x_0) {
        b_idx = 1;
      } else {
        b_idx = 0;
        i = 2;
        exitg1 = false;
        while ((!exitg1) && (i < 14)) {
          if (!rtIsNaN(offset[i - 1])) {
            b_idx = i;
            exitg1 = true;
          } else {
            i++;
          }
        }
      }

      if (b_idx == 0) {
        delta_offset = offset[0];
      } else {
        delta_offset = offset[b_idx - 1];
        while (b_idx + 1 < 14) {
          if (delta_offset < offset[b_idx]) {
            delta_offset = offset[b_idx];
          }

          b_idx++;
        }
      }

      for (b_ix = 0; b_ix < 13; b_ix++) {
        offset[b_ix] /= delta_offset;
      }
    }
  }

  for (b_ix = 0; b_ix < 13; b_ix++) {
    Clane[b_ix] = Path_col[(b_ix << 2) + 3];
    Cobs_0[b_ix] = Path_col[(b_ix << 2) + 3];
  }

  target_k = std(Cobs_0);
  if (target_k != 0.0) {
    rtb_UnitDelay18 = target_k * target_k * 2.0;
    delta_offset = 2.5066282746310002 * target_k;
    for (c_ix = 0; c_ix < 13; c_ix++) {
      b_ix = 1 + c_ix;
      for (iy = 0; iy < 13; iy++) {
        Cc_0[iy] = (b_ix - iy) - 1;
      }

      power_dw3x(Cc_0, rtb_forward_length_free);
      for (b_ix = 0; b_ix < 13; b_ix++) {
        Cc_0[b_ix] = -rtb_forward_length_free[b_ix] / rtb_UnitDelay18;
      }

      exp_n(Cc_0);
      for (b_ix = 0; b_ix < 13; b_ix++) {
        Cobs_0[b_ix] = Path_col[(b_ix << 2) + 3] * (Cc_0[b_ix] / delta_offset);
      }

      Clane[c_ix] = sum_a(Cobs_0);
      if ((1 + c_ix == 1) && (Path_col[3] == 1.0)) {
        Clane[0] += std::exp(-1.0 / (target_k * target_k * 2.0)) /
          (2.5066282746310002 * target_k);
      } else {
        if ((1 + c_ix == 13) && (Path_col[51] == 1.0)) {
          Clane[12] += std::exp(-1.0 / (target_k * target_k * 2.0)) /
            (2.5066282746310002 * target_k);
        }
      }
    }

    b_x_0 = rtIsNaN(Clane[0]);
    if (!b_x_0) {
      i = 1;
    } else {
      i = 0;
      b_idx = 2;
      exitg1 = false;
      while ((!exitg1) && (b_idx < 14)) {
        if (!rtIsNaN(Clane[b_idx - 1])) {
          i = b_idx;
          exitg1 = true;
        } else {
          b_idx++;
        }
      }
    }

    if (i == 0) {
      delta_offset = Clane[0];
    } else {
      delta_offset = Clane[i - 1];
      while (i + 1 < 14) {
        if (delta_offset < Clane[i]) {
          delta_offset = Clane[i];
        }

        i++;
      }
    }

    if (delta_offset != 1.0) {
      if (!b_x_0) {
        b_idx = 1;
      } else {
        b_idx = 0;
        i = 2;
        exitg1 = false;
        while ((!exitg1) && (i < 14)) {
          if (!rtIsNaN(Clane[i - 1])) {
            b_idx = i;
            exitg1 = true;
          } else {
            i++;
          }
        }
      }

      if (b_idx == 0) {
        delta_offset = Clane[0];
      } else {
        delta_offset = Clane[b_idx - 1];
        while (b_idx + 1 < 14) {
          if (delta_offset < Clane[b_idx]) {
            delta_offset = Clane[b_idx];
          }

          b_idx++;
        }
      }

      for (b_ix = 0; b_ix < 13; b_ix++) {
        Clane[b_ix] /= delta_offset;
      }
    }
  }

  for (b_ix = 0; b_ix < 11; b_ix++) {
    x[b_ix] = rtDW.UnitDelay6_DSTATE[b_ix] - seg_direction;
  }

  power_d(x, K1_0);
  for (b_ix = 0; b_ix < 11; b_ix++) {
    x[b_ix] = rtDW.UnitDelay6_DSTATE[11 + b_ix] - head_err;
  }

  power_d(x, X2);
  for (b_ix = 0; b_ix < 11; b_ix++) {
    K_11[b_ix] = K1_0[b_ix] + X2[b_ix];
  }

  sqrt_l(K_11);
  if (!rtIsNaN(K_11[0])) {
    b_idx = 1;
  } else {
    b_idx = 0;
    i = 2;
    exitg1 = false;
    while ((!exitg1) && (i < 12)) {
      if (!rtIsNaN(K_11[i - 1])) {
        b_idx = i;
        exitg1 = true;
      } else {
        i++;
      }
    }
  }

  if (b_idx == 0) {
    b_idx = 1;
  } else {
    delta_offset = K_11[b_idx - 1];
    for (i = b_idx; i + 1 < 12; i++) {
      if (delta_offset > K_11[i]) {
        delta_offset = K_11[i];
        b_idx = i + 1;
      }
    }
  }

  jj = 12 - b_idx;
  d_ix = -b_idx;
  for (b_ix = 0; b_ix <= d_ix + 11; b_ix++) {
    LastPath_overlap_data[b_ix] = rtDW.UnitDelay6_DSTATE[(b_idx + b_ix) - 1];
  }

  d_ix = -b_idx;
  for (b_ix = 0; b_ix <= d_ix + 11; b_ix++) {
    LastPath_overlap_data[b_ix + jj] = rtDW.UnitDelay6_DSTATE[(b_idx + b_ix) +
      10];
  }

  for (i = 0; i < 13; i++) {
    for (b_ix = 0; b_ix < 11; b_ix++) {
      b_Path_dis_data[b_ix] = X_2[11 * i + b_ix] - rtDW.UnitDelay6_DSTATE[10];
    }

    power_d(b_Path_dis_data, x);
    for (b_ix = 0; b_ix < 11; b_ix++) {
      b_Path_dis_data[b_ix] = Y[11 * i + b_ix] - rtDW.UnitDelay6_DSTATE[21];
    }

    power_d(b_Path_dis_data, K1_0);
    for (b_ix = 0; b_ix < 11; b_ix++) {
      K_11[b_ix] = x[b_ix] + K1_0[b_ix];
    }

    sqrt_l(K_11);
    if (!rtIsNaN(K_11[0])) {
      idx = 0;
    } else {
      idx = -1;
      c_ix = 2;
      exitg1 = false;
      while ((!exitg1) && (c_ix < 12)) {
        if (!rtIsNaN(K_11[c_ix - 1])) {
          idx = c_ix - 1;
          exitg1 = true;
        } else {
          c_ix++;
        }
      }
    }

    if (idx + 1 == 0) {
      idx = 0;
    } else {
      rtb_UnitDelay18 = K_11[idx];
      for (c_ix = idx + 1; c_ix + 1 < 12; c_ix++) {
        if (rtb_UnitDelay18 > K_11[c_ix]) {
          rtb_UnitDelay18 = K_11[c_ix];
          idx = c_ix;
        }
      }
    }

    Path_overlap_size[0] = idx + 1;
    if (0 <= idx) {
      memcpy(&Path_overlap_data[0], &X_2[i * 11], (idx + 1) * sizeof(real_T));
    }

    for (b_ix = 0; b_ix <= idx; b_ix++) {
      Path_overlap_data[b_ix + Path_overlap_size[0]] = Y[11 * i + b_ix];
    }

    if (12 - b_idx >= Path_overlap_size[0]) {
      idx = 13 - (b_idx + Path_overlap_size[0]);
      if (idx > 12 - b_idx) {
        idx = 1;
        ix = 0;
      } else {
        ix = 12 - b_idx;
      }

      b_ix = idx - 1;
      idx = ix - b_ix;
      LastPath_overlap_size_2[0] = idx;
      LastPath_overlap_size_2[1] = 2;
      for (iy = 0; iy < idx; iy++) {
        LastPath_overlap_data_0[iy] = LastPath_overlap_data[b_ix + iy] -
          Path_overlap_data[iy];
      }

      for (iy = 0; iy < idx; iy++) {
        LastPath_overlap_data_0[iy + idx] = LastPath_overlap_data[(b_ix + iy) +
          jj] - Path_overlap_data[iy + Path_overlap_size[0]];
      }

      power_egqso(LastPath_overlap_data_0, LastPath_overlap_size_2,
                  Path_overlap_data, Path_overlap_size);
      Path_overlap_size_3[0] = 2;
      Path_overlap_size_3[1] = Path_overlap_size[0];
      d_ix = Path_overlap_size[0];
      for (b_ix = 0; b_ix < d_ix; b_ix++) {
        LastPath_overlap_data_0[b_ix << 1] = Path_overlap_data[b_ix];
        LastPath_overlap_data_0[1 + (b_ix << 1)] = Path_overlap_data[b_ix +
          Path_overlap_size[0]];
      }

      sum_hx(LastPath_overlap_data_0, Path_overlap_size_3, b_Path_dis_data,
             oi_xy_size);
      sqrt_l5(b_Path_dis_data, oi_xy_size);
      d_ix = oi_xy_size[1];
      for (b_ix = 0; b_ix < d_ix; b_ix++) {
        K_11[b_ix] = b_Path_dis_data[oi_xy_size[0] * b_ix];
      }

      idx = oi_xy_size[1];
      iy = oi_xy_size[1];
      if (0 <= idx - 1) {
        memcpy(&Path_dis_data[0], &K_11[0], idx * sizeof(real_T));
      }
    } else {
      ix = 12 - b_idx;
      LastPath_overlap_size_1[0] = ix;
      LastPath_overlap_size_1[1] = 2;
      for (b_ix = 0; b_ix < ix; b_ix++) {
        LastPath_overlap_data_0[b_ix] = LastPath_overlap_data[b_ix] -
          Path_overlap_data[b_ix];
      }

      for (b_ix = 0; b_ix < ix; b_ix++) {
        LastPath_overlap_data_0[b_ix + ix] = LastPath_overlap_data[b_ix + jj] -
          Path_overlap_data[b_ix + Path_overlap_size[0]];
      }

      power_egqso(LastPath_overlap_data_0, LastPath_overlap_size_1,
                  Path_overlap_data, Path_overlap_size);
      Path_overlap_size_2[0] = 2;
      Path_overlap_size_2[1] = Path_overlap_size[0];
      d_ix = Path_overlap_size[0];
      for (b_ix = 0; b_ix < d_ix; b_ix++) {
        LastPath_overlap_data_0[b_ix << 1] = Path_overlap_data[b_ix];
        LastPath_overlap_data_0[1 + (b_ix << 1)] = Path_overlap_data[b_ix +
          Path_overlap_size[0]];
      }

      sum_hx(LastPath_overlap_data_0, Path_overlap_size_2, b_Path_dis_data,
             oi_xy_size);
      sqrt_l5(b_Path_dis_data, oi_xy_size);
      d_ix = oi_xy_size[1];
      for (b_ix = 0; b_ix < d_ix; b_ix++) {
        b_Path_dis_data_0[b_ix] = b_Path_dis_data[oi_xy_size[0] * b_ix];
      }

      idx = oi_xy_size[1];
      iy = oi_xy_size[1];
      if (0 <= idx - 1) {
        memcpy(&Path_dis_data[0], &b_Path_dis_data_0[0], idx * sizeof(real_T));
      }
    }

    if (iy > 1) {
      idx = iy;
    } else {
      idx = 1;
    }

    if (mod((real_T)idx) == 0.0) {
      if (iy > 1) {
        idx = iy - 1;
      } else {
        idx = 0;
      }

      oi_xy_size[1] = idx;
      d_ix = idx - 1;
      for (b_ix = 0; b_ix <= d_ix; b_ix++) {
        b_Path_dis_data[b_ix] = 4.0;
      }
    } else {
      if (iy > 1) {
        idx = iy;
      } else {
        idx = 1;
      }

      oi_xy_size[1] = idx;
      d_ix = idx - 1;
      for (b_ix = 0; b_ix <= d_ix; b_ix++) {
        b_Path_dis_data[b_ix] = 4.0;
      }
    }

    b_Path_dis_data[0] = 1.0;
    b_Path_dis_data[oi_xy_size[1] - 1] = 1.0;
    if (3 > oi_xy_size[1] - 2) {
      ix = 1;
      n = 1;
      c_ix = 0;
    } else {
      ix = 3;
      n = 2;
      c_ix = oi_xy_size[1] - 2;
    }

    idx = div_nde_s32_floor((int8_T)c_ix - ix, n);
    for (b_ix = 0; b_ix <= idx; b_ix++) {
      p_data[b_ix] = (int8_T)((int8_T)((int8_T)(n * (int8_T)b_ix) + ix) - 1);
    }

    for (b_ix = 0; b_ix <= idx; b_ix++) {
      b_Path_dis_data[p_data[b_ix]] = 2.0;
    }

    delta_offset = 0.0;
    for (b_ix = 0; b_ix < oi_xy_size[1]; b_ix++) {
      delta_offset += b_Path_dis_data[b_ix] * Path_dis_data[b_ix];
    }

    if (!(iy > 1)) {
      iy = 1;
    }

    Cobs_0[i] = Cobs[i] / 11.0 * delta_offset / 3.0 / (Cobs[i] * (real_T)iy /
      11.0);
  }

  for (i = 0; i < 13; i++) {
    Cobs[i] = 1.0;
    Cc_0[i] = 0.0;
    rtb_forward_length_free[i] = 0.0;
  }

  if ((rtU.Freespace_mode == 1.0) || (rtU.Freespace_mode == 2.0)) {
    for (c_ix = 0; c_ix < 13; c_ix++) {
      FreespaceDetectCollision_b(rtU.Freespace, &rtb_XP[6 * c_ix], &rtb_YP[6 *
        c_ix], rtb_TmpSignalConversionAtSFun_e, total_length, rtU.safe_range,
        rtConstP.pooled8, rtConstP.pooled7, &Cobs[c_ix], &Cc_0[c_ix],
        &rtb_forward_length_free[c_ix]);
    }
  }

  b_x_0 = rtIsNaN(Cobs_0[0]);
  if (!b_x_0) {
    b_idx = 1;
  } else {
    b_idx = 0;
    i = 2;
    exitg1 = false;
    while ((!exitg1) && (i < 14)) {
      if (!rtIsNaN(Cobs_0[i - 1])) {
        b_idx = i;
        exitg1 = true;
      } else {
        i++;
      }
    }
  }

  if (b_idx == 0) {
    target_k = Cobs_0[0];
  } else {
    target_k = Cobs_0[b_idx - 1];
    while (b_idx + 1 < 14) {
      if (target_k < Cobs_0[b_idx]) {
        target_k = Cobs_0[b_idx];
      }

      b_idx++;
    }
  }

  if (!(target_k == 0.0)) {
    if (!b_x_0) {
      idx = 1;
    } else {
      idx = 0;
      i = 2;
      exitg1 = false;
      while ((!exitg1) && (i < 14)) {
        if (!rtIsNaN(Cobs_0[i - 1])) {
          idx = i;
          exitg1 = true;
        } else {
          i++;
        }
      }
    }

    if (idx == 0) {
      delta_offset = Cobs_0[0];
    } else {
      delta_offset = Cobs_0[idx - 1];
      while (idx + 1 < 14) {
        if (delta_offset < Cobs_0[idx]) {
          delta_offset = Cobs_0[idx];
        }

        idx++;
      }
    }

    for (b_ix = 0; b_ix < 13; b_ix++) {
      Cobs_0[b_ix] /= delta_offset;
    }
  }

  for (b_ix = 0; b_ix < 13; b_ix++) {
    offset[b_ix] = (rtU.W_2[0] * offset[b_ix] + rtU.W_2[1] * Cobs_0[b_ix]) +
      rtU.W_2[2] * Clane[b_ix];
  }

  for (ix = 0; ix < 13; ix++) {
    // MATLAB Function: '<S2>/J_fsc_design' incorporates:
    //   Inport: '<Root>/w_fs'

    if (rtb_U_c_l[ix] == 1.0) {
      delta_offset = rtb_forward_length_free_o[ix] + rtb_forward_length_free[ix];
      if (Cobs[ix] == 1.0) {
        count = 0.0;
      } else {
        count = 2.0 - Cobs[ix];
      }

      count = count * rtU.w_fs + Cc_0[ix];
    } else {
      delta_offset = rtb_forward_length_free_o[ix];
      if (rtb_U_c_l[ix] == 1.0) {
        count = 0.0;
      } else {
        count = 3.0 - rtb_U_c_l[ix];
      }

      count = count * rtU.w_fs + rtb_safety_level_all_p[ix];
    }

    if (delta_offset > target_k_tmp) {
      count = 0.0;
    }

    rtb_forward_length_free_2[ix] = delta_offset;
    Clane[ix] = count;

    // End of MATLAB Function: '<S2>/J_fsc_design'

    // MATLAB Function: '<S2>/Fianl_Path_Decision'
    rtb_J_out_k[ix] = (rtb_J_out_k[ix] + offset[ix]) + count;
  }

  // MATLAB Function: '<S2>/Fianl_Path_Decision' incorporates:
  //   Inport: '<Root>/Freespace_mode'
  //   Inport: '<Root>/J_minvalue_diff_min'
  //   Inport: '<Root>/J_minvalue_index'
  //   Inport: '<Root>/Path_flag'
  //   Inport: '<Root>/W_1'
  //   Inport: '<Root>/w_fs'
  //   UnitDelay: '<S2>/Unit Delay11'
  //   UnitDelay: '<S2>/Unit Delay13'
  //   UnitDelay: '<S2>/Unit Delay7'

  if (!rtIsNaN(rtb_J_out_k[0])) {
    idx = 1;
  } else {
    idx = 0;
    i = 2;
    exitg1 = false;
    while ((!exitg1) && (i < 14)) {
      if (!rtIsNaN(rtb_J_out_k[i - 1])) {
        idx = i;
        exitg1 = true;
      } else {
        i++;
      }
    }
  }

  if (idx == 0) {
    delta_offset = rtb_J_out_k[0];
    idx = 1;
  } else {
    delta_offset = rtb_J_out_k[idx - 1];
    for (i = idx; i + 1 < 14; i++) {
      if (delta_offset > rtb_J_out_k[i]) {
        delta_offset = rtb_J_out_k[i];
        idx = i + 1;
      }
    }
  }

  count = std::abs(rtDW.UnitDelay11_DSTATE - delta_offset);
  ang_1 = 0.0;
  if (rtU.Path_flag == 1.0) {
    if ((rtU.Freespace_mode == 0.0) || (rtU.Freespace_mode == 2.0)) {
      if (delta_offset >= rtU.W_1[3]) {
        count = rtDW.UnitDelay7_DSTATE;
      } else if ((count < rtU.J_minvalue_diff_min) && (rtDW.UnitDelay13_DSTATE <
                  rtU.J_minvalue_index)) {
        count = rtDW.UnitDelay7_DSTATE;
        ang_1 = rtDW.UnitDelay13_DSTATE + 1.0;
      } else {
        count = idx;
      }
    } else if (delta_offset >= rtU.w_fs) {
      count = rtDW.UnitDelay7_DSTATE;
    } else if ((count < rtU.J_minvalue_diff_min) && (rtDW.UnitDelay13_DSTATE <
                rtU.J_minvalue_index)) {
      count = rtDW.UnitDelay7_DSTATE;
      ang_1 = rtDW.UnitDelay13_DSTATE + 1.0;
    } else {
      count = idx;
    }
  } else {
    count = 7.0;
  }

  b_idx = (int32_T)count;
  c_ix = (int32_T)count;
  ix = (int32_T)count;
  n = (int32_T)count;
  iy = (int32_T)count;
  d_ix = (int32_T)count;
  jj = (int32_T)count;
  xy_ends_POS_size_idx_0 = (int32_T)count;
  Path_RES_0_size_idx_1 = (int32_T)count;
  Path_RES_1_size_idx_0 = (int32_T)count;
  count_0 = (int32_T)count;
  count_2 = (int32_T)count;
  for (i = 0; i < 11; i++) {
    End_x = a[i] * a[i];

    // Update for UnitDelay: '<S2>/Unit Delay5'
    rtDW.UnitDelay5_DSTATE[i] = ((((rtb_XP_i[(c_ix - 1) * 6 + 1] * a[i] +
      rtb_XP_i[(b_idx - 1) * 6]) + rtb_XP_i[(ix - 1) * 6 + 2] * End_x) +
      rtb_XP_i[(n - 1) * 6 + 3] * rt_powd_snf(a[i], 3.0)) + rtb_XP_i[(iy - 1) *
      6 + 4] * rt_powd_snf(a[i], 4.0)) + rtb_XP_i[(d_ix - 1) * 6 + 5] *
      rt_powd_snf(a[i], 5.0);
    rtDW.UnitDelay5_DSTATE[i + 11] = ((((rtb_YP_g[(xy_ends_POS_size_idx_0 - 1) *
      6 + 1] * a[i] + rtb_YP_g[(jj - 1) * 6]) + rtb_YP_g[(Path_RES_0_size_idx_1
      - 1) * 6 + 2] * End_x) + rtb_YP_g[(Path_RES_1_size_idx_0 - 1) * 6 + 3] *
      rt_powd_snf(a[i], 3.0)) + rtb_YP_g[(count_0 - 1) * 6 + 4] * rt_powd_snf
      (a[i], 4.0)) + rtb_YP_g[(count_2 - 1) * 6 + 5] * rt_powd_snf(a[i], 5.0);
    x[i] = End_x;
    b_Path_dis_data[i] = rt_powd_snf(a[i], 3.0);
    K1_0[i] = rt_powd_snf(a[i], 4.0);
    K_11[i] = rt_powd_snf(a[i], 5.0);
    X2[i] = End_x;
    Y2[i] = rt_powd_snf(a[i], 3.0);
    K2[i] = rt_powd_snf(a[i], 4.0);
    K_12[i] = rt_powd_snf(a[i], 5.0);
  }

  // Outport: '<Root>/J'
  memcpy(&rtY.J[0], &rtb_J_out_k[0], 13U * sizeof(real_T));

  // Outport: '<Root>/J_fsc'
  memcpy(&rtY.J_fsc[0], &Clane[0], 13U * sizeof(real_T));

  // Outport: '<Root>/U_c_1'
  memcpy(&rtY.U_c_1[0], &Cobs[0], 13U * sizeof(real_T));

  // Outport: '<Root>/safety_level_all_1'
  memcpy(&rtY.safety_level_all_1[0], &Cc_0[0], 13U * sizeof(real_T));

  // Outport: '<Root>/U_c'
  memcpy(&rtY.U_c[0], &rtb_U_c_l[0], 13U * sizeof(real_T));

  // Outport: '<Root>/safety_level_all'
  memcpy(&rtY.safety_level_all[0], &rtb_safety_level_all_p[0], 13U * sizeof
         (real_T));

  // MATLAB Function: '<S2>/Fianl_Path_Decision'
  b_idx = (int32_T)count;
  c_ix = (int32_T)count;
  ix = (int32_T)count;
  n = (int32_T)count;
  for (b_ix = 0; b_ix < 6; b_ix++) {
    // Outport: '<Root>/XP_final' incorporates:
    //   MATLAB Function: '<S2>/Fianl_Path_Decision'

    rtY.XP_final[b_ix] = rtb_XP_i[(b_idx - 1) * 6 + b_ix];

    // Outport: '<Root>/YP_final' incorporates:
    //   MATLAB Function: '<S2>/Fianl_Path_Decision'

    rtY.YP_final[b_ix] = rtb_YP_g[(c_ix - 1) * 6 + b_ix];

    // Outport: '<Root>/XP_final_1' incorporates:
    //   MATLAB Function: '<S2>/Fianl_Path_Decision'

    rtY.XP_final_1[b_ix] = rtb_XP[(ix - 1) * 6 + b_ix];

    // Outport: '<Root>/YP_final_1' incorporates:
    //   MATLAB Function: '<S2>/Fianl_Path_Decision'

    rtY.YP_final_1[b_ix] = rtb_YP[(n - 1) * 6 + b_ix];
  }

  // Outport: '<Root>/X_UKF_SLAM'
  for (i = 0; i < 5; i++) {
    rtY.X_UKF_SLAM[i] = rtb_X[i];
  }

  // End of Outport: '<Root>/X_UKF_SLAM'

  // SignalConversion: '<S26>/TmpSignal ConversionAt SFunction Inport1' incorporates:
  //   Gain: '<S2>/Gain2'
  //   MATLAB Function: '<S2>/Target_Point_Decision'

  rtb_TmpSignalConversionAtSFun_e[2] = 0.017453292519943295 * ajj;

  // MATLAB Function: '<S2>/Target_Point_Decision' incorporates:
  //   MATLAB Function: '<S2>/EndPointDecision'
  //   MATLAB Function: '<S2>/EndPointDecision1'
  //   MATLAB Function: '<S2>/Fianl_Path_Decision'
  //   SignalConversion: '<S26>/TmpSignal ConversionAt SFunction Inport1'

  if (rtb_Gain_p <= seg_heading) {
    ajj = rtb_Gain_p / seg_heading;
    rtb_Gain_p = ((((rtb_XP_i[((int32_T)count - 1) * 6 + 1] * ajj + rtb_XP_i
                     [((int32_T)count - 1) * 6]) + rtb_XP_i[((int32_T)count - 1)
                    * 6 + 2] * (ajj * ajj)) + rtb_XP_i[((int32_T)count - 1) * 6
                   + 3] * rt_powd_snf(ajj, 3.0)) + rtb_XP_i[((int32_T)count - 1)
                  * 6 + 4] * rt_powd_snf(ajj, 4.0)) + rtb_XP_i[((int32_T)count -
      1) * 6 + 5] * rt_powd_snf(ajj, 5.0);
    ajj = ((((rtb_YP_g[((int32_T)count - 1) * 6 + 1] * ajj + rtb_YP_g[((int32_T)
               count - 1) * 6]) + rtb_YP_g[((int32_T)count - 1) * 6 + 2] * (ajj *
              ajj)) + rtb_YP_g[((int32_T)count - 1) * 6 + 3] * rt_powd_snf(ajj,
             3.0)) + rtb_YP_g[((int32_T)count - 1) * 6 + 4] * rt_powd_snf(ajj,
            4.0)) + rtb_YP_g[((int32_T)count - 1) * 6 + 5] * rt_powd_snf(ajj,
      5.0);
  } else if ((rtb_Gain_p > seg_heading) && (rtb_Gain_p <= seg_heading +
              total_length)) {
    ajj = (rtb_Gain_p - seg_heading) / total_length;
    seg_heading = ajj * ajj;
    rtb_Gain_p = ((((rtb_XP[((int32_T)count - 1) * 6 + 1] * ajj + rtb_XP
                     [((int32_T)count - 1) * 6]) + rtb_XP[((int32_T)count - 1) *
                    6 + 2] * seg_heading) + rtb_XP[((int32_T)count - 1) * 6 + 3]
                   * rt_powd_snf(ajj, 3.0)) + rtb_XP[((int32_T)count - 1) * 6 +
                  4] * rt_powd_snf(ajj, 4.0)) + rtb_XP[((int32_T)count - 1) * 6
      + 5] * rt_powd_snf(ajj, 5.0);
    ajj = ((((rtb_YP[((int32_T)count - 1) * 6 + 1] * ajj + rtb_YP[((int32_T)
               count - 1) * 6]) + rtb_YP[((int32_T)count - 1) * 6 + 2] *
             seg_heading) + rtb_YP[((int32_T)count - 1) * 6 + 3] * rt_powd_snf
            (ajj, 3.0)) + rtb_YP[((int32_T)count - 1) * 6 + 4] * rt_powd_snf(ajj,
            4.0)) + rtb_YP[((int32_T)count - 1) * 6 + 5] * rt_powd_snf(ajj, 5.0);
  } else {
    rtb_Gain_p = ((((rtb_XP[((int32_T)count - 1) * 6 + 1] + rtb_XP[((int32_T)
      count - 1) * 6]) + rtb_XP[((int32_T)count - 1) * 6 + 2]) + rtb_XP
                   [((int32_T)count - 1) * 6 + 3]) + rtb_XP[((int32_T)count - 1)
                  * 6 + 4]) + rtb_XP[((int32_T)count - 1) * 6 + 5];
    ajj = ((((rtb_YP[((int32_T)count - 1) * 6 + 1] + rtb_YP[((int32_T)count - 1)
              * 6]) + rtb_YP[((int32_T)count - 1) * 6 + 2]) + rtb_YP[((int32_T)
             count - 1) * 6 + 3]) + rtb_YP[((int32_T)count - 1) * 6 + 4]) +
      rtb_YP[((int32_T)count - 1) * 6 + 5];
  }

  rtb_Gain_p -= rtb_X[0];
  ajj -= rtb_X[1];
  seg_heading = std::sin(-rtb_TmpSignalConversionAtSFun_e[2]);
  total_length = std::cos(-rtb_TmpSignalConversionAtSFun_e[2]);

  // Outport: '<Root>/Vehicle_Target_x' incorporates:
  //   MATLAB Function: '<S2>/Target_Point_Decision'

  rtY.Vehicle_Target_x = total_length * rtb_Gain_p + -seg_heading * ajj;

  // Outport: '<Root>/Vehicle_Target_y' incorporates:
  //   MATLAB Function: '<S2>/Target_Point_Decision'

  rtY.Vehicle_Target_y = seg_heading * rtb_Gain_p + total_length * ajj;

  // Outport: '<Root>/J_minind' incorporates:
  //   MATLAB Function: '<S2>/Fianl_Path_Decision'

  rtY.J_minind = idx;

  // Outport: '<Root>/J_finalind' incorporates:
  //   MATLAB Function: '<S2>/Fianl_Path_Decision'

  rtY.J_finalind = count;

  // Outport: '<Root>/forward_length_free' incorporates:
  //   MATLAB Function: '<S2>/Fianl_Path_Decision'

  rtY.forward_length_free = rtb_forward_length_free_2[(int32_T)count - 1];

  // Outport: '<Root>/takeover_length'
  rtY.takeover_length = target_k_tmp;

  // Outport: '<Root>/takeoverlength_ind' incorporates:
  //   MATLAB Function: '<S2>/DynamicPathPlanning1'

  rtY.takeoverlength_ind = seg_direction_tmp_tmp;

  // Outport: '<Root>/OB_enlargescale' incorporates:
  //   MATLAB Function: '<S2>/DynamicPathPlanning1'

  rtY.OB_enlargescale = 0.15;

  // Outport: '<Root>/OB_enlargescale_frontbehind' incorporates:
  //   MATLAB Function: '<S2>/DynamicPathPlanning1'

  rtY.OB_enlargescale_frontbehind = 0.3;

  // Outport: '<Root>/avoidance_mode' incorporates:
  //   MATLAB Function: '<S2>/DangerousArea1'

  rtY.avoidance_mode = Length_1;

  // Outport: '<Root>/End_x' incorporates:
  //   MATLAB Function: '<S2>/EndPointDecision'

  rtY.End_x = seg_direction;

  // Outport: '<Root>/End_y' incorporates:
  //   MATLAB Function: '<S2>/EndPointDecision'

  rtY.End_y = head_err;

  // Outport: '<Root>/forward_length'
  rtY.forward_length = forward_length;

  // Outport: '<Root>/Target_seg_id' incorporates:
  //   MATLAB Function: '<S2>/target_seg_id_search'

  rtY.Target_seg_id = shortest_distance[rtDW.SFunction_DIMS4 - 1];

  // Outport: '<Root>/Look_ahead_time'
  rtY.Look_ahead_time = b_j;

  // Outport: '<Root>/seg_id_near' incorporates:
  //   MATLAB Function: '<S2>/MM'

  rtY.seg_id_near = rtb_Gain1;

  // Update for Memory: '<S2>/Memory1' incorporates:
  //   Constant: '<S2>/Constant7'

  rtDW.Memory1_PreviousInput = 1.0;

  // Update for Memory: '<S2>/Memory' incorporates:
  //   Constant: '<S2>/Constant8'

  rtDW.Memory_PreviousInput = 113.0;

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

  // Update for UnitDelay: '<S2>/Unit Delay14' incorporates:
  //   MATLAB Function: '<S2>/DangerousArea1'

  rtDW.UnitDelay14_DSTATE = Length_1;

  // Update for UnitDelay: '<S2>/Unit Delay16'
  rtDW.UnitDelay16_DSTATE = forward_length;

  // MATLAB Function: '<S2>/Fianl_Path_Decision'
  b_idx = (int32_T)count;
  c_ix = (int32_T)count;
  ix = (int32_T)count;
  n = (int32_T)count;
  iy = (int32_T)count;
  d_ix = (int32_T)count;
  jj = (int32_T)count;
  xy_ends_POS_size_idx_0 = (int32_T)count;
  Path_RES_0_size_idx_1 = (int32_T)count;
  Path_RES_1_size_idx_0 = (int32_T)count;
  count_0 = (int32_T)count;
  count_2 = (int32_T)count;
  for (b_ix = 0; b_ix < 11; b_ix++) {
    // Update for UnitDelay: '<S2>/Unit Delay6'
    rtDW.UnitDelay6_DSTATE[b_ix] = ((((rtb_XP[(c_ix - 1) * 6 + 1] * a[b_ix] +
      rtb_XP[(b_idx - 1) * 6]) + rtb_XP[(ix - 1) * 6 + 2] * x[b_ix]) + rtb_XP[(n
      - 1) * 6 + 3] * b_Path_dis_data[b_ix]) + rtb_XP[(iy - 1) * 6 + 4] *
      K1_0[b_ix]) + rtb_XP[(d_ix - 1) * 6 + 5] * K_11[b_ix];
    rtDW.UnitDelay6_DSTATE[b_ix + 11] = ((((rtb_YP[(xy_ends_POS_size_idx_0 - 1) *
      6 + 1] * a[b_ix] + rtb_YP[(jj - 1) * 6]) + rtb_YP[(Path_RES_0_size_idx_1 -
      1) * 6 + 2] * X2[b_ix]) + rtb_YP[(Path_RES_1_size_idx_0 - 1) * 6 + 3] *
      Y2[b_ix]) + rtb_YP[(count_0 - 1) * 6 + 4] * K2[b_ix]) + rtb_YP[(count_2 -
      1) * 6 + 5] * K_12[b_ix];
  }

  // Update for UnitDelay: '<S2>/Unit Delay18' incorporates:
  //   MATLAB Function: '<S2>/DangerousArea1'

  rtDW.UnitDelay18_DSTATE = Length_1;

  // Update for UnitDelay: '<S2>/Unit Delay17' incorporates:
  //   MATLAB Function: '<S2>/DangerousArea1'

  rtDW.UnitDelay17_DSTATE = count_1;

  // Update for UnitDelay: '<S2>/Unit Delay19'
  rtDW.UnitDelay19_DSTATE[0] = rtb_num_lane_direction_f[0];

  // Update for UnitDelay: '<S2>/Unit Delay15'
  rtDW.UnitDelay15_DSTATE[0] = rtb_H_y_out[0];

  // Update for UnitDelay: '<S2>/Unit Delay19'
  rtDW.UnitDelay19_DSTATE[1] = rtb_num_lane_direction_f[1];

  // Update for UnitDelay: '<S2>/Unit Delay15'
  rtDW.UnitDelay15_DSTATE[1] = rtb_H_y_out[1];

  // Update for UnitDelay: '<S2>/Unit Delay19'
  rtDW.UnitDelay19_DSTATE[2] = rtb_num_lane_direction_f[2];

  // Update for UnitDelay: '<S2>/Unit Delay15'
  rtDW.UnitDelay15_DSTATE[2] = rtb_H_y_out[2];

  // Update for UnitDelay: '<S2>/Unit Delay19'
  rtDW.UnitDelay19_DSTATE[3] = rtb_num_lane_direction_f[3];

  // Update for UnitDelay: '<S2>/Unit Delay15'
  rtDW.UnitDelay15_DSTATE[3] = rtb_H_y_out[3];

  // Update for UnitDelay: '<S2>/Unit Delay7' incorporates:
  //   MATLAB Function: '<S2>/Fianl_Path_Decision'

  rtDW.UnitDelay7_DSTATE = count;

  // Update for UnitDelay: '<S2>/Unit Delay11' incorporates:
  //   MATLAB Function: '<S2>/Fianl_Path_Decision'

  rtDW.UnitDelay11_DSTATE = delta_offset;

  // Update for UnitDelay: '<S2>/Unit Delay13'
  rtDW.UnitDelay13_DSTATE = ang_1;
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

  // InitializeConditions for UnitDelay: '<S2>/Unit Delay16'
  rtDW.UnitDelay16_DSTATE = 3.0;

  // InitializeConditions for UnitDelay: '<S2>/Unit Delay7'
  rtDW.UnitDelay7_DSTATE = 7.0;
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
