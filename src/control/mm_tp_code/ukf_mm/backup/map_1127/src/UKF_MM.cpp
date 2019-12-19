//
// File: UKF_MM.cpp
//
// Code generated for Simulink model 'UKF_MM'.
//
// Model version                  : 1.4
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Thu Dec  5 17:34:28 2019
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
// Code generation objectives:
//    1. Execution efficiency
//    2. RAM efficiency
// Validation result: Not run
//
#include "UKF_MM.h"
#define NumBitsPerChar                 8U

extern "C" {
  extern real_T rtGetInf(void);
  extern real32_T rtGetInfF(void);
  extern real_T rtGetMinusInf(void);
  extern real32_T rtGetMinusInfF(void);
}                                      // extern "C"
  extern "C"
{
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

extern "C" {
  real_T rtInf;
  real_T rtMinusInf;
  real_T rtNaN;
  real32_T rtInfF;
  real32_T rtMinusInfF;
  real32_T rtNaNF;
}
  extern "C"
{
  extern real_T rtGetNaN(void);
  extern real32_T rtGetNaNF(void);
}                                      // extern "C"

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

extern "C" {
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
// Function for MATLAB Function: '<S3>/SLAM_UKF'
  real_T UKF_MMModelClass::sum(const real_T x[10])
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
void UKF_MMModelClass::invNxN(const real_T x[25], real_T y[25])
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
void UKF_MMModelClass::merge(int32_T idx[301], real_T x[301], int32_T offset,
  int32_T np, int32_T nq, int32_T iwork[301], real_T xwork[301])
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
void UKF_MMModelClass::merge_block(int32_T idx[301], real_T x[301], int32_T
  offset, int32_T n, int32_T preSortLevel, int32_T iwork[301], real_T xwork[301])
{
  int32_T bLen;
  int32_T tailOffset;
  int32_T nTail;
  int32_T nPairs;
  nPairs = n >> preSortLevel;
  bLen = 1 << preSortLevel;
  while (nPairs > 1) {
    if ((nPairs & 1U) != 0U) {
      nPairs--;
      tailOffset = bLen * nPairs;
      nTail = n - tailOffset;
      if (nTail > bLen) {
        merge(idx, x, offset + tailOffset, bLen, nTail - bLen, iwork, xwork);
      }
    }

    tailOffset = bLen << 1;
    nPairs >>= 1;
    for (nTail = 1; nTail <= nPairs; nTail++) {
      merge(idx, x, offset + (nTail - 1) * tailOffset, bLen, bLen, iwork, xwork);
    }

    bLen = tailOffset;
  }

  if (n > bLen) {
    merge(idx, x, offset, bLen, n - bLen, iwork, xwork);
  }
}

// Function for MATLAB Function: '<S3>/SLAM_UKF_MM'
void UKF_MMModelClass::merge_pow2_block(int32_T idx[301], real_T x[301], int32_T
  offset)
{
  int32_T iwork[256];
  real_T xwork[256];
  int32_T bLen;
  int32_T bLen2;
  int32_T nPairs;
  int32_T blockOffset;
  int32_T p;
  int32_T q;
  int32_T b;
  int32_T k;
  int32_T exitg1;
  for (b = 0; b < 6; b++) {
    bLen = 1 << (b + 2);
    bLen2 = bLen << 1;
    nPairs = 256 >> (b + 3);
    for (k = 1; k <= nPairs; k++) {
      blockOffset = ((k - 1) * bLen2 + offset) - 1;
      for (p = 1; p <= bLen2; p++) {
        q = blockOffset + p;
        iwork[p - 1] = idx[q];
        xwork[p - 1] = x[q];
      }

      p = 0;
      q = bLen;
      do {
        exitg1 = 0;
        blockOffset++;
        if (xwork[p] <= xwork[q]) {
          idx[blockOffset] = iwork[p];
          x[blockOffset] = xwork[p];
          if (p + 1 < bLen) {
            p++;
          } else {
            exitg1 = 1;
          }
        } else {
          idx[blockOffset] = iwork[q];
          x[blockOffset] = xwork[q];
          if (q + 1 < bLen2) {
            q++;
          } else {
            blockOffset -= p;
            while (p + 1 <= bLen) {
              q = (blockOffset + p) + 1;
              idx[q] = iwork[p];
              x[q] = xwork[p];
              p++;
            }

            exitg1 = 1;
          }
        }
      } while (exitg1 == 0);
    }
  }
}

// Function for MATLAB Function: '<S3>/SLAM_UKF_MM'
void UKF_MMModelClass::sort(real_T x[301], int32_T idx[301])
{
  int32_T iwork[301];
  real_T xwork[301];
  int32_T nNaNs;
  real_T x4[4];
  int16_T idx4[4];
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
  memset(&idx[0], 0, 301U * sizeof(int32_T));
  memset(&xwork[0], 0, 301U * sizeof(real_T));
  nNaNs = 0;
  ib = 0;
  for (m = 0; m < 301; m++) {
    if (rtIsNaN(x[m])) {
      idx[300 - nNaNs] = m + 1;
      xwork[300 - nNaNs] = x[m];
      nNaNs++;
    } else {
      ib++;
      idx4[ib - 1] = (int16_T)(m + 1);
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

    for (m = 301; m - 300 <= ib; m++) {
      i1 = perm[m - 301] - 1;
      i2 = (m - nNaNs) - ib;
      idx[i2] = idx4[i1];
      x[i2] = x4[i1];
    }
  }

  m = nNaNs >> 1;
  for (ib = 1; ib <= m; ib++) {
    i2 = (ib - nNaNs) + 300;
    i1 = idx[i2];
    idx[i2] = idx[301 - ib];
    idx[301 - ib] = i1;
    x[i2] = xwork[301 - ib];
    x[301 - ib] = xwork[i2];
  }

  if ((nNaNs & 1U) != 0U) {
    x[(m - nNaNs) + 301] = xwork[(m - nNaNs) + 301];
  }

  memset(&iwork[0], 0, 301U * sizeof(int32_T));
  ib = 2;
  if (301 - nNaNs > 1) {
    m = (301 - nNaNs) >> 8;
    if (m > 0) {
      for (ib = 1; ib <= m; ib++) {
        merge_pow2_block(idx, x, (ib - 1) << 8);
      }

      m <<= 8;
      ib = 301 - (nNaNs + m);
      if (ib > 0) {
        memset(&iwork[0], 0, 301U * sizeof(int32_T));
        merge_block(idx, x, m, ib, 2, iwork, xwork);
      }

      ib = 8;
    }

    merge_block(idx, x, 0, 301 - nNaNs, ib, iwork, xwork);
  }
}

// Function for MATLAB Function: '<S3>/SLAM_UKF_MM'
void UKF_MMModelClass::power(const real_T a[301], real_T y[301])
{
  int32_T k;
  for (k = 0; k < 301; k++) {
    y[k] = a[k] * a[k];
  }
}

// Function for MATLAB Function: '<S3>/SLAM_UKF_MM'
void UKF_MMModelClass::rel_dist_xy(const real_T ref_xy[2], const real_T pt_xy
  [602], real_T dist[301])
{
  real_T pt_xy_0[301];
  real_T tmp[301];
  real_T tmp_0[301];
  int32_T k;
  for (k = 0; k < 301; k++) {
    pt_xy_0[k] = pt_xy[k] - ref_xy[0];
  }

  power(pt_xy_0, tmp);
  for (k = 0; k < 301; k++) {
    pt_xy_0[k] = pt_xy[301 + k] - ref_xy[1];
  }

  power(pt_xy_0, tmp_0);
  for (k = 0; k < 301; k++) {
    dist[k] = std::sqrt(tmp[k] + tmp_0[k]);
  }
}

// Function for MATLAB Function: '<S3>/SLAM_UKF_MM'
real_T UKF_MMModelClass::rel_dist_xy_g(const real_T ref_xy[2], const real_T
  pt_xy[2])
{
  real_T a;
  real_T b_a;
  a = pt_xy[0] - ref_xy[0];
  b_a = pt_xy[1] - ref_xy[1];
  return std::sqrt(a * a + b_a * b_a);
}

// Function for MATLAB Function: '<S3>/SLAM_UKF_MM'
void UKF_MMModelClass::MM(real_T heading, const real_T X_pos[2], const real_T
  oi_xy[602], const real_T dist_op[301], const real_T Map_data[6923], real_T
  *seg_id_near, real_T *op_distance, real_T oi_near[2], real_T *note, real_T
  *seg_direction, real_T *head_err, real_T num_lane_direction[4], real_T
  *seg_heading)
{
  real_T op_distance_n;
  real_T C;
  int32_T b_index[602];
  real_T dist_ini[301];
  real_T dist_end[301];
  int32_T iidx[301];
  boolean_T x[301];
  int16_T ii_data[301];
  int32_T idx;
  int32_T b_ii;
  boolean_T ex;
  int32_T d_k;
  real_T c_a;
  real_T d_a;
  real_T oi_xy_0[2];
  boolean_T exitg1;
  memcpy(&dist_ini[0], &dist_op[0], 301U * sizeof(real_T));
  sort(dist_ini, iidx);
  for (b_ii = 0; b_ii < 301; b_ii++) {
    b_index[b_ii] = iidx[b_ii];
    b_index[b_ii + 301] = 0;
    rtDW.SEG_GPS_HEAD[b_ii] = Map_data[b_ii];
    rtDW.SEG_GPS_HEAD[301 + b_ii] = Map_data[2107 + b_ii];
  }

  for (idx = 0; idx < 301; idx++) {
    op_distance_n = Map_data[b_index[idx] + 300] - oi_xy[b_index[idx] - 1];
    C = Map_data[b_index[idx] + 601] - oi_xy[b_index[idx] + 300];
    c_a = Map_data[b_index[idx] + 902] - oi_xy[b_index[idx] - 1];
    d_a = Map_data[b_index[idx] + 1203] - oi_xy[b_index[idx] + 300];
    if (std::sqrt(op_distance_n * op_distance_n + C * C) <= Map_data[b_index[idx]
        + 2407]) {
      b_index[301 + idx] = (std::sqrt(c_a * c_a + d_a * d_a) <=
                            Map_data[b_index[idx] + 2407]);
    } else {
      b_index[301 + idx] = 0;
    }

    x[idx] = (b_index[301 + idx] == 1);
  }

  idx = 0;
  b_ii = 1;
  exitg1 = false;
  while ((!exitg1) && (b_ii < 302)) {
    if (x[b_ii - 1]) {
      idx++;
      ii_data[idx - 1] = (int16_T)b_ii;
      if (idx >= 301) {
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
    rel_dist_xy(X_pos, &Map_data[301], dist_ini);
    rel_dist_xy(X_pos, &Map_data[903], dist_end);
    if (!rtIsNaN(dist_ini[0])) {
      idx = 0;
    } else {
      idx = -1;
      b_ii = 2;
      exitg1 = false;
      while ((!exitg1) && (b_ii < 302)) {
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
      for (b_ii = idx + 1; b_ii + 1 < 302; b_ii++) {
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
      while ((!exitg1) && (d_k < 302)) {
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
      for (d_k = b_ii + 1; d_k + 1 < 302; d_k++) {
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
    oi_near[1] = oi_xy[idx + 301];
    if ((idx + 1 > 1) && (idx + 1 < 301)) {
      oi_xy_0[0] = oi_xy[idx];
      oi_xy_0[1] = oi_xy[idx + 301];
      op_distance_n = rel_dist_xy_g(X_pos, oi_xy_0);
      oi_xy_0[0] = Map_data[idx + 301];
      oi_xy_0[1] = Map_data[idx + 602];
      if (op_distance_n < rel_dist_xy_g(X_pos, oi_xy_0)) {
        oi_xy_0[0] = Map_data[idx + 903];
        oi_xy_0[1] = Map_data[idx + 1204];
        if (op_distance_n < rel_dist_xy_g(X_pos, oi_xy_0)) {
          *note = 0.0;
        }
      }
    }

    for (b_ii = 0; b_ii < 301; b_ii++) {
      x[b_ii] = (rtDW.SEG_GPS_HEAD[b_ii] == Map_data[idx]);
    }

    idx = -1;
    ex = x[0];
    for (b_ii = 0; b_ii < 300; b_ii++) {
      if ((int32_T)ex < (int32_T)x[b_ii + 1]) {
        ex = x[b_ii + 1];
        idx = b_ii;
      }
    }

    *head_err = std::abs(heading - rtDW.SEG_GPS_HEAD[idx + 302]);
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
    oi_near[1] = oi_xy[b_index[ii_data[0] - 1] + 300];
    for (b_ii = 0; b_ii < 301; b_ii++) {
      x[b_ii] = (Map_data[b_index[ii_data[0] - 1] - 1] == rtDW.SEG_GPS_HEAD[b_ii]);
    }

    idx = -1;
    ex = x[0];
    for (b_ii = 0; b_ii < 300; b_ii++) {
      if ((int32_T)ex < (int32_T)x[b_ii + 1]) {
        ex = x[b_ii + 1];
        idx = b_ii;
      }
    }

    *head_err = std::abs(heading - rtDW.SEG_GPS_HEAD[idx + 302]);
    if (*head_err <= 0.78539816339744828) {
      *seg_direction = 1.0;
    } else if ((*head_err >= 0.78539816339744828) && (*head_err <= 90.0)) {
      *seg_direction = 0.0;
    } else {
      *seg_direction = 2.0;
    }

    rel_dist_xy(X_pos, &Map_data[301], dist_ini);
    rel_dist_xy(X_pos, &Map_data[903], dist_end);
    if (!rtIsNaN(dist_ini[0])) {
      idx = 0;
    } else {
      idx = -1;
      b_ii = 2;
      exitg1 = false;
      while ((!exitg1) && (b_ii < 302)) {
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
      for (b_ii = idx + 1; b_ii + 1 < 302; b_ii++) {
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
      while ((!exitg1) && (d_k < 302)) {
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
      for (d_k = b_ii + 1; d_k + 1 < 302; d_k++) {
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

      for (b_ii = 0; b_ii < 301; b_ii++) {
        x[b_ii] = (Map_data[b_index[ii_data[0] - 1] - 1] ==
                   rtDW.SEG_GPS_HEAD[b_ii]);
      }

      b_ii = -1;
      ex = x[0];
      for (d_k = 0; d_k < 300; d_k++) {
        if ((int32_T)ex < (int32_T)x[d_k + 1]) {
          ex = x[d_k + 1];
          b_ii = d_k;
        }
      }

      *head_err = std::abs(heading - rtDW.SEG_GPS_HEAD[b_ii + 302]);
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
      oi_near[1] = oi_xy[idx + 301];
    }
  }

  for (b_ii = 0; b_ii < 301; b_ii++) {
    dist_ini[b_ii] = Map_data[2107 + b_ii] * 3.1415926535897931 / 180.0;
    x[b_ii] = (Map_data[b_ii] == *seg_id_near);
  }

  idx = 0;
  ex = x[0];
  for (b_ii = 0; b_ii < 300; b_ii++) {
    if ((int32_T)ex < (int32_T)x[b_ii + 1]) {
      ex = x[b_ii + 1];
      idx = b_ii + 1;
    }
  }

  op_distance_n = oi_near[1] - Map_data[1505 + idx] * oi_near[0];
  if (Map_data[1505 + idx] < 0.0) {
    C = (-Map_data[1505 + idx] * X_pos[0] - op_distance_n) + X_pos[1];
    if (dist_ini[idx] > 4.71238898038469) {
      if (!(dist_ini[idx] < 6.2831853071795862)) {
        C = -C;
      }
    } else {
      C = -C;
    }
  } else if (Map_data[1505 + idx] == 0.0) {
    if (oi_near[1] < X_pos[1]) {
      C = -1.0;
    } else {
      C = 1.0;
    }
  } else {
    C = (Map_data[1505 + idx] * X_pos[0] + op_distance_n) - X_pos[1];
    if (dist_ini[idx] > 3.1415926535897931) {
      if (!(dist_ini[idx] < 4.71238898038469)) {
        C = -C;
      }
    } else {
      C = -C;
    }
  }

  num_lane_direction[0] = Map_data[1505 + idx];
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
int32_T UKF_MMModelClass::nonSingletonDim(const int32_T *x_size)
{
  int32_T dim;
  dim = 2;
  if (*x_size != 1) {
    dim = 1;
  }

  return dim;
}

// Function for MATLAB Function: '<S2>/MM'
void UKF_MMModelClass::merge_o(int32_T idx_data[], real_T x_data[], int32_T
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
void UKF_MMModelClass::merge_block_e(int32_T idx_data[], real_T x_data[],
  int32_T offset, int32_T n, int32_T preSortLevel, int32_T iwork_data[], real_T
  xwork_data[])
{
  int32_T bLen;
  int32_T tailOffset;
  int32_T nTail;
  int32_T nPairs;
  nPairs = n >> preSortLevel;
  bLen = 1 << preSortLevel;
  while (nPairs > 1) {
    if ((nPairs & 1U) != 0U) {
      nPairs--;
      tailOffset = bLen * nPairs;
      nTail = n - tailOffset;
      if (nTail > bLen) {
        merge_o(idx_data, x_data, offset + tailOffset, bLen, nTail - bLen,
                iwork_data, xwork_data);
      }
    }

    tailOffset = bLen << 1;
    nPairs >>= 1;
    for (nTail = 1; nTail <= nPairs; nTail++) {
      merge_o(idx_data, x_data, offset + (nTail - 1) * tailOffset, bLen, bLen,
              iwork_data, xwork_data);
    }

    bLen = tailOffset;
  }

  if (n > bLen) {
    merge_o(idx_data, x_data, offset, bLen, n - bLen, iwork_data, xwork_data);
  }
}

// Function for MATLAB Function: '<S2>/MM'
void UKF_MMModelClass::merge_pow2_block_c(int32_T idx_data[], real_T x_data[],
  int32_T offset)
{
  int32_T iwork[256];
  real_T xwork[256];
  int32_T bLen;
  int32_T bLen2;
  int32_T nPairs;
  int32_T blockOffset;
  int32_T p;
  int32_T q;
  int32_T b;
  int32_T k;
  int32_T exitg1;
  for (b = 0; b < 6; b++) {
    bLen = 1 << (b + 2);
    bLen2 = bLen << 1;
    nPairs = 256 >> (b + 3);
    for (k = 1; k <= nPairs; k++) {
      blockOffset = ((k - 1) * bLen2 + offset) - 1;
      for (p = 1; p <= bLen2; p++) {
        q = blockOffset + p;
        iwork[p - 1] = idx_data[q];
        xwork[p - 1] = x_data[q];
      }

      p = 0;
      q = bLen;
      do {
        exitg1 = 0;
        blockOffset++;
        if (xwork[p] <= xwork[q]) {
          idx_data[blockOffset] = iwork[p];
          x_data[blockOffset] = xwork[p];
          if (p + 1 < bLen) {
            p++;
          } else {
            exitg1 = 1;
          }
        } else {
          idx_data[blockOffset] = iwork[q];
          x_data[blockOffset] = xwork[q];
          if (q + 1 < bLen2) {
            q++;
          } else {
            blockOffset -= p;
            while (p + 1 <= bLen) {
              q = (blockOffset + p) + 1;
              idx_data[q] = iwork[p];
              x_data[q] = xwork[p];
              p++;
            }

            exitg1 = 1;
          }
        }
      } while (exitg1 == 0);
    }
  }
}

// Function for MATLAB Function: '<S2>/MM'
void UKF_MMModelClass::sortIdx(real_T x_data[], int32_T *x_size, int32_T
  idx_data[], int32_T *idx_size)
{
  int32_T iwork_data[301];
  real_T xwork_data[301];
  int32_T nBlocks;
  real_T c_x_data[301];
  real_T x4[4];
  int16_T idx4[4];
  int32_T ib;
  int32_T wOffset;
  int32_T itmp;
  int8_T perm[4];
  int32_T i1;
  int32_T i3;
  int32_T i4;
  int32_T c_x_size;
  int16_T b_x_idx_0;
  int16_T b_idx_0;
  b_x_idx_0 = (int16_T)*x_size;
  b_idx_0 = (int16_T)*x_size;
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
    b_idx_0 = (int16_T)*x_size;
    if (0 <= b_idx_0 - 1) {
      memset(&xwork_data[0], 0, b_idx_0 * sizeof(real_T));
    }

    nBlocks = 1;
    ib = 0;
    for (wOffset = 0; wOffset < *x_size; wOffset++) {
      if (rtIsNaN(c_x_data[wOffset])) {
        i3 = *x_size - nBlocks;
        idx_data[i3] = wOffset + 1;
        xwork_data[i3] = c_x_data[wOffset];
        nBlocks++;
      } else {
        ib++;
        idx4[ib - 1] = (int16_T)(wOffset + 1);
        x4[ib - 1] = c_x_data[wOffset];
        if (ib == 4) {
          ib = wOffset - nBlocks;
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

    wOffset = *x_size - nBlocks;
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

    ib = ((nBlocks - 1) >> 1) + 1;
    for (i1 = 1; i1 < ib; i1++) {
      i4 = wOffset + i1;
      itmp = idx_data[i4];
      i3 = *x_size - i1;
      idx_data[i4] = idx_data[i3];
      idx_data[i3] = itmp;
      c_x_data[i4] = xwork_data[i3];
      c_x_data[i3] = xwork_data[i4];
    }

    if (((nBlocks - 1) & 1U) != 0U) {
      c_x_data[wOffset + ib] = xwork_data[wOffset + ib];
    }

    if (0 <= b_x_idx_0 - 1) {
      memset(&iwork_data[0], 0, b_x_idx_0 * sizeof(int32_T));
    }

    ib = wOffset + 1;
    wOffset = 2;
    if (ib > 1) {
      if (*x_size >= 256) {
        nBlocks = ib >> 8;
        if (nBlocks > 0) {
          for (wOffset = 1; wOffset <= nBlocks; wOffset++) {
            merge_pow2_block_c(idx_data, c_x_data, (wOffset - 1) << 8);
          }

          nBlocks <<= 8;
          wOffset = ib - nBlocks;
          if (wOffset > 0) {
            if (0 <= b_x_idx_0 - 1) {
              memset(&iwork_data[0], 0, b_x_idx_0 * sizeof(int32_T));
            }

            merge_block_e(idx_data, c_x_data, nBlocks, wOffset, 2, iwork_data,
                          xwork_data);
          }

          wOffset = 8;
        }
      }

      merge_block_e(idx_data, c_x_data, 0, ib, wOffset, iwork_data, xwork_data);
    }

    if (0 <= c_x_size - 1) {
      memcpy(&x_data[0], &c_x_data[0], c_x_size * sizeof(real_T));
    }
  }
}

// Function for MATLAB Function: '<S2>/MM'
void UKF_MMModelClass::sort_m(real_T x_data[], int32_T *x_size, int32_T
  idx_data[], int32_T *idx_size)
{
  int32_T dim;
  real_T vwork_data[301];
  int32_T vstride;
  int32_T iidx_data[301];
  int32_T b;
  int32_T c_k;
  int32_T vwork_size;
  int32_T tmp;
  dim = nonSingletonDim(x_size);
  if (dim <= 1) {
    b = *x_size;
  } else {
    b = 1;
  }

  vwork_size = (int16_T)b;
  *idx_size = (int16_T)*x_size;
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
void UKF_MMModelClass::power_j(const real_T a_data[], const int32_T *a_size,
  real_T y_data[], int32_T *y_size)
{
  real_T z1_data[301];
  int32_T loop_ub;
  int16_T a_idx_0;
  a_idx_0 = (int16_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&z1_data[0], &y_data[0], a_idx_0 * sizeof(real_T));
  }

  for (loop_ub = 0; loop_ub < a_idx_0; loop_ub++) {
    z1_data[loop_ub] = a_data[loop_ub] * a_data[loop_ub];
  }

  *y_size = (int16_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&y_data[0], &z1_data[0], a_idx_0 * sizeof(real_T));
  }
}

// Function for MATLAB Function: '<S2>/MM'
void UKF_MMModelClass::power_jm(const real_T a_data[], const int32_T *a_size,
  real_T y_data[], int32_T *y_size)
{
  real_T z1_data[301];
  int32_T loop_ub;
  int16_T a_idx_0;
  a_idx_0 = (int16_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&z1_data[0], &y_data[0], a_idx_0 * sizeof(real_T));
  }

  for (loop_ub = 0; loop_ub < a_idx_0; loop_ub++) {
    z1_data[loop_ub] = std::sqrt(a_data[loop_ub]);
  }

  *y_size = (int16_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&y_data[0], &z1_data[0], a_idx_0 * sizeof(real_T));
  }
}

// Function for MATLAB Function: '<S2>/MM'
void UKF_MMModelClass::rel_dist_xy_o(const real_T ref_xy[2], const real_T
  pt_xy_data[], const int32_T pt_xy_size[2], real_T dist_data[], int32_T
  *dist_size)
{
  real_T pt_xy_data_0[301];
  real_T tmp_data[301];
  real_T tmp_data_0[301];
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

  power_j(pt_xy_data_0, &pt_xy_size_0, tmp_data, &tmp_size);
  loop_ub = pt_xy_size[0];
  pt_xy_size_1 = pt_xy_size[0];
  for (i = 0; i < loop_ub; i++) {
    pt_xy_data_0[i] = pt_xy_data[i + pt_xy_size[0]] - ref_xy[1];
  }

  power_j(pt_xy_data_0, &pt_xy_size_1, tmp_data_0, &pt_xy_size_0);
  for (i = 0; i < tmp_size; i++) {
    pt_xy_data_0[i] = tmp_data[i] + tmp_data_0[i];
  }

  power_jm(pt_xy_data_0, &tmp_size, dist_data, dist_size);
}

// Function for MATLAB Function: '<S2>/MM'
void UKF_MMModelClass::MM_o(real_T heading, const real_T X_pos[2], const real_T
  oi_xy_data[], const int32_T oi_xy_size[2], const real_T dist_op_data[], const
  int32_T *dist_op_size, const real_T Map_data_data[], const int32_T
  Map_data_size[2], real_T *seg_id_near, real_T *op_distance, real_T oi_near[2],
  real_T *note, real_T *seg_direction, real_T *head_err, real_T
  num_lane_direction[4], real_T *seg_heading)
{
  real_T seg_id_data[301];
  real_T ind_temp_data[301];
  real_T op_distance_n;
  real_T C;
  int32_T b_index_data[602];
  real_T dist_ini_data[301];
  real_T dist_end_data[301];
  boolean_T x_data[301];
  int32_T ii_data[301];
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

  sort_m(ind_temp_data, &g_idx, ii_data, &ii_size);
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
    rtDW.xy_ini_data[g_idx] = Map_data_data[g_idx + Map_data_size[0]];
  }

  for (g_idx = 0; g_idx < nx; g_idx++) {
    rtDW.xy_end_data[g_idx] = Map_data_data[Map_data_size[0] * 3 + g_idx];
  }

  for (g_idx = 0; g_idx < loop_ub; g_idx++) {
    rtDW.xy_ini_data[g_idx + loop_ub] = Map_data_data[(Map_data_size[0] << 1) +
      g_idx];
  }

  for (g_idx = 0; g_idx < nx; g_idx++) {
    rtDW.xy_end_data[g_idx + nx] = Map_data_data[(Map_data_size[0] << 2) + g_idx];
  }

  loop_ub = Map_data_size[0];
  if (0 <= loop_ub - 1) {
    memcpy(&seg_id_data[0], &Map_data_data[0], loop_ub * sizeof(real_T));
    memcpy(&rtDW.SEG_GPS_HEAD_data[0], &seg_id_data[0], loop_ub * sizeof(real_T));
  }

  nx = Map_data_size[0] - 1;
  for (g_idx = 0; g_idx <= nx; g_idx++) {
    rtDW.SEG_GPS_HEAD_data[g_idx + loop_ub] = Map_data_data[Map_data_size[0] * 7
      + g_idx];
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
    rel_dist_xy_o(X_pos, rtDW.xy_ini_data, xy_ini_size, dist_ini_data, &g_idx);
    rel_dist_xy_o(X_pos, rtDW.xy_end_data, xy_end_size, dist_end_data, &ii_size);
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
      op_distance_n = rel_dist_xy_g(X_pos, oi_xy);
      oi_xy[0] = rtDW.xy_ini_data[nx];
      g_idx = nx + Map_data_size[0];
      oi_xy[1] = rtDW.xy_ini_data[g_idx];
      if (op_distance_n < rel_dist_xy_g(X_pos, oi_xy)) {
        oi_xy[0] = rtDW.xy_end_data[nx];
        oi_xy[1] = rtDW.xy_end_data[g_idx];
        if (op_distance_n < rel_dist_xy_g(X_pos, oi_xy)) {
          *note = 0.0;
        }
      }
    }

    for (g_idx = 0; g_idx < loop_ub; g_idx++) {
      x_data[g_idx] = (rtDW.SEG_GPS_HEAD_data[g_idx] == Map_data_data[nx]);
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

    *head_err = std::abs(heading - rtDW.SEG_GPS_HEAD_data[(ii_data[0] +
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
                       - 1] == rtDW.SEG_GPS_HEAD_data[g_idx]);
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

    *head_err = std::abs(heading - rtDW.SEG_GPS_HEAD_data[(ii_data[0] +
      Map_data_size[0]) - 1]);
    if (*head_err <= 0.78539816339744828) {
      *seg_direction = 1.0;
    } else if ((*head_err >= 0.78539816339744828) && (*head_err <= 90.0)) {
      *seg_direction = 0.0;
    } else {
      *seg_direction = 2.0;
    }

    rel_dist_xy_o(X_pos, rtDW.xy_ini_data, xy_ini_size, dist_ini_data, &g_idx);
    rel_dist_xy_o(X_pos, rtDW.xy_end_data, xy_end_size, dist_end_data, &ii_size);
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
                         1] - 1] == rtDW.SEG_GPS_HEAD_data[g_idx]);
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

      *head_err = std::abs(heading - rtDW.SEG_GPS_HEAD_data[(ii_data[0] +
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

// Function for MATLAB Function: '<S9>/Dijkstra'
void UKF_MMModelClass::power_d(const real_T a[2], real_T y[2])
{
  y[0] = a[0] * a[0];
  y[1] = a[1] * a[1];
}

// Function for MATLAB Function: '<S9>/Dijkstra'
real_T UKF_MMModelClass::sum_k(const real_T x[2])
{
  return x[0] + x[1];
}

// Model step function
void UKF_MMModelClass::step()
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
  int32_T colj;
  int32_T ix;
  int32_T iy;
  int32_T b_ix;
  int8_T ii_data[5];
  int32_T d_ix;
  int32_T b_ia;
  real_T K1[4];
  int8_T I[25];
  real_T seg_heading;
  real_T dist_op_data[301];
  real_T total_length;
  int32_T end_ind_0;
  real_T Forward_Static_Path_id_0_data[301];
  int16_T cb_data[301];
  real_T shortest_distance[301];
  int8_T settled[301];
  int16_T pidx_data[301];
  int16_T zz_data[301];
  real_T tmp_path_data[301];
  int16_T nidx_data[301];
  int16_T c_data[301];
  int32_T ii_data_0[301];
  int32_T b_idx;
  int32_T n;
  boolean_T x_data[301];
  boolean_T b_x[301];
  int16_T f_ii_data[301];
  real_T rtb_Add;
  real_T rtb_Gain1;
  real_T rtb_Gain_k;
  real_T rtb_X[5];
  real_T rtb_UnitDelay34[5];
  real_T rtb_X_state[5];
  real_T rtb_Oi_near_b[2];
  real_T rtb_num_lane_direction_f[4];
  real_T rtb_Q_last_o[25];
  real_T rtb_R_last_o[25];
  real_T rtb_X_AUG[55];
  real_T rtb_K[25];
  int32_T i;
  real_T rtb_X_AUG_0[10];
  real_T p_sqrt_data_0[25];
  real_T rtb_X_state_0[2];
  int16_T tmp;
  int32_T oi_xy_size[2];
  boolean_T b_x_0;
  real_T y_idx_2;
  boolean_T exitg1;
  int32_T exitg2;

  // Sum: '<S2>/Add' incorporates:
  //   Abs: '<S2>/Abs'
  //   Abs: '<S2>/Abs1'
  //   Constant: '<S2>/Constant7'
  //   Constant: '<S2>/Constant8'
  //   Memory: '<S2>/Memory'
  //   Memory: '<S2>/Memory1'
  //   Sum: '<S2>/Sum'
  //   Sum: '<S2>/Sum1'

  rtb_Add = std::abs(1.0 - rtDW.Memory1_PreviousInput) + std::abs(301.0 -
    rtDW.Memory_PreviousInput);

  // Outputs for Enabled SubSystem: '<S2>/Enabled Subsystem' incorporates:
  //   EnablePort: '<S9>/Enable'

  if (rtb_Add > 0.0) {
    // MATLAB Function: '<S9>/Dijkstra' incorporates:
    //   Constant: '<S2>/Constant7'

    memset(&rtDW.table[0], 0, 602U * sizeof(real_T));
    for (i = 0; i < 301; i++) {
      shortest_distance[i] = (rtInf);
      settled[i] = 0;
    }

    memset(&rtDW.path[0], 0, 90601U * sizeof(real_T));
    i = 0;
    n = 1;
    exitg1 = false;
    while ((!exitg1) && (n < 302)) {
      if (rtConstP.Constant3_Value[n - 1] == 1.0) {
        i++;
        ii_data_0[i - 1] = n;
        if (i >= 301) {
          exitg1 = true;
        } else {
          n++;
        }
      } else {
        n++;
      }
    }

    if (1 > i) {
      colj = 0;
    } else {
      colj = i;
    }

    for (jj = 0; jj < colj; jj++) {
      pidx_data[jj] = (int16_T)ii_data_0[jj];
    }

    shortest_distance[ii_data_0[0] - 1] = 0.0;
    rtDW.table[ii_data_0[0] + 300] = 0.0;
    settled[ii_data_0[0] - 1] = 1;
    rtDW.path[ii_data_0[0] - 1] = 1.0;
    b_idx = 0;
    i = 1;
    exitg1 = false;
    while ((!exitg1) && (i < 302)) {
      if (rtConstP.Constant3_Value[i - 1] == 301.0) {
        b_idx++;
        ii_data_0[b_idx - 1] = i;
        if (b_idx >= 301) {
          exitg1 = true;
        } else {
          i++;
        }
      } else {
        i++;
      }
    }

    if (1 > b_idx) {
      colj = 0;
    } else {
      colj = b_idx;
    }

    for (jj = 0; jj < colj; jj++) {
      zz_data[jj] = (int16_T)ii_data_0[jj];
    }

    do {
      exitg2 = 0;
      jj = zz_data[0] - 1;
      if (settled[jj] == 0) {
        for (ix = 0; ix < 301; ix++) {
          rtDW.table[ix] = rtDW.table[301 + ix];
        }

        colj = pidx_data[0] + 300;
        rtDW.table[colj] = 0.0;
        b_idx = 0;
        for (n = 0; n < 301; n++) {
          b_x_0 = (rtConstP.Constant3_Value[pidx_data[0] - 1] ==
                   rtConstP.Constant5_Value[301 + n]);
          if (b_x_0) {
            b_idx++;
          }

          b_x[n] = b_x_0;
        }

        b_ix = b_idx;
        b_idx = 0;
        for (i = 0; i < 301; i++) {
          if (b_x[i]) {
            c_data[b_idx] = (int16_T)(i + 1);
            b_idx++;
          }
        }

        for (end_ind_0 = 0; end_ind_0 < b_ix; end_ind_0++) {
          for (ix = 0; ix < 301; ix++) {
            b_x[ix] = (rtConstP.Constant5_Value[c_data[end_ind_0] + 601] ==
                       rtConstP.Constant3_Value[ix]);
          }

          b_idx = -1;
          i = 1;
          exitg1 = false;
          while ((!exitg1) && (i < 302)) {
            if (b_x[i - 1]) {
              b_idx++;
              ii_data_0[b_idx] = i;
              if (b_idx + 1 >= 301) {
                exitg1 = true;
              } else {
                i++;
              }
            } else {
              i++;
            }
          }

          if (!(settled[ii_data_0[0] - 1] != 0)) {
            rtb_X_state_0[0] = rtConstP.Constant3_Value[colj] -
              rtConstP.Constant3_Value[ii_data_0[0] + 300];
            rtb_X_state_0[1] = rtConstP.Constant3_Value[pidx_data[0] + 601] -
              rtConstP.Constant3_Value[ii_data_0[0] + 601];
            power_d(rtb_X_state_0, rtb_Oi_near_b);
            ajj = std::sqrt(sum_k(rtb_Oi_near_b));
            if ((rtDW.table[ii_data_0[0] - 1] == 0.0) || (rtDW.table[ii_data_0[0]
                 - 1] > rtDW.table[pidx_data[0] - 1] + ajj)) {
              rtDW.table[ii_data_0[0] + 300] = rtDW.table[pidx_data[0] - 1] +
                ajj;
              for (ix = 0; ix < 301; ix++) {
                b_x[ix] = (rtDW.path[(301 * ix + pidx_data[0]) - 1] != 0.0);
              }

              b_idx = 0;
              i = 1;
              exitg1 = false;
              while ((!exitg1) && (i < 302)) {
                if (b_x[i - 1]) {
                  b_idx++;
                  f_ii_data[b_idx - 1] = (int16_T)i;
                  if (b_idx >= 301) {
                    exitg1 = true;
                  } else {
                    i++;
                  }
                } else {
                  i++;
                }
              }

              if (1 > b_idx) {
                n = 0;
              } else {
                n = b_idx;
              }

              iy = n - 1;
              if (0 <= iy) {
                memset(&tmp_path_data[0], 0, (iy + 1) * sizeof(real_T));
              }

              for (b_idx = 0; b_idx < n; b_idx++) {
                tmp_path_data[b_idx] = rtDW.path[((f_ii_data[b_idx] - 1) * 301 +
                  pidx_data[0]) - 1];
              }

              b_idx = ii_data_0[0] - 1;
              for (ix = 0; ix < n; ix++) {
                rtDW.path[b_idx + 301 * ix] = tmp_path_data[ix];
              }

              rtDW.path[b_idx + 301 * n] =
                rtConstP.Constant5_Value[c_data[end_ind_0] + 601];
            } else {
              rtDW.table[ii_data_0[0] + 300] = rtDW.table[ii_data_0[0] - 1];
            }
          }
        }

        b_idx = 0;
        i = 1;
        exitg1 = false;
        while ((!exitg1) && (i < 302)) {
          if (rtDW.table[i + 300] != 0.0) {
            b_idx++;
            ii_data_0[b_idx - 1] = i;
            if (b_idx >= 301) {
              exitg1 = true;
            } else {
              i++;
            }
          } else {
            i++;
          }
        }

        if (1 > b_idx) {
          colj = 0;
        } else {
          colj = b_idx;
        }

        for (ix = 0; ix < colj; ix++) {
          nidx_data[ix] = (int16_T)ii_data_0[ix];
        }

        if (colj <= 2) {
          if (colj == 1) {
            SLAM_X_out = rtDW.table[ii_data_0[0] + 300];
          } else if (rtDW.table[ii_data_0[0] + 300] > rtDW.table[ii_data_0[1] +
                     300]) {
            SLAM_X_out = rtDW.table[ii_data_0[1] + 300];
          } else if (rtIsNaN(rtDW.table[ii_data_0[0] + 300])) {
            if (!rtIsNaN(rtDW.table[ii_data_0[1] + 300])) {
              SLAM_X_out = rtDW.table[ii_data_0[1] + 300];
            } else {
              SLAM_X_out = rtDW.table[ii_data_0[0] + 300];
            }
          } else {
            SLAM_X_out = rtDW.table[ii_data_0[0] + 300];
          }
        } else {
          if (!rtIsNaN(rtDW.table[ii_data_0[0] + 300])) {
            b_idx = 1;
          } else {
            b_idx = 0;
            end_ind_0 = 2;
            exitg1 = false;
            while ((!exitg1) && (end_ind_0 <= colj)) {
              if (!rtIsNaN(rtDW.table[ii_data_0[end_ind_0 - 1] + 300])) {
                b_idx = end_ind_0;
                exitg1 = true;
              } else {
                end_ind_0++;
              }
            }
          }

          if (b_idx == 0) {
            SLAM_X_out = rtDW.table[ii_data_0[0] + 300];
          } else {
            SLAM_X_out = rtDW.table[ii_data_0[b_idx - 1] + 300];
            while (b_idx + 1 <= colj) {
              if (SLAM_X_out > rtDW.table[ii_data_0[b_idx] + 300]) {
                SLAM_X_out = rtDW.table[ii_data_0[b_idx] + 300];
              }

              b_idx++;
            }
          }
        }

        for (ix = 0; ix < colj; ix++) {
          x_data[ix] = (rtDW.table[ii_data_0[ix] + 300] == SLAM_X_out);
        }

        i = 0;
        n = 1;
        exitg1 = false;
        while ((!exitg1) && (n <= colj)) {
          if (x_data[n - 1]) {
            i++;
            ii_data_0[i - 1] = n;
            if (i >= colj) {
              exitg1 = true;
            } else {
              n++;
            }
          } else {
            n++;
          }
        }

        if (colj == 1) {
          if (i == 0) {
            colj = 0;
          }
        } else if (1 > i) {
          colj = 0;
        } else {
          colj = i;
        }

        if (colj == 0) {
          exitg2 = 1;
        } else {
          pidx_data[0] = nidx_data[ii_data_0[0] - 1];
          b_idx = nidx_data[ii_data_0[0] - 1] - 1;
          shortest_distance[b_idx] = rtDW.table[nidx_data[ii_data_0[0] - 1] +
            300];
          settled[b_idx] = 1;
        }
      } else {
        exitg2 = 1;
      }
    } while (exitg2 == 0);

    for (ix = 0; ix < 301; ix++) {
      b_x[ix] = (rtDW.path[(301 * ix + zz_data[0]) - 1] != 0.0);
    }

    b_idx = 0;
    i = 0;
    exitg1 = false;
    while ((!exitg1) && (i + 1 < 302)) {
      if (b_x[i]) {
        b_idx++;
        if (b_idx >= 301) {
          exitg1 = true;
        } else {
          i++;
        }
      } else {
        i++;
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

    rtDW.dist = shortest_distance[jj];
    rtDW.SFunction_DIMS3_g = n;
    for (jj = 0; jj < n; jj++) {
      rtDW.path_2[jj] = rtDW.path[(301 * jj + zz_data[0]) - 1];
    }

    // End of MATLAB Function: '<S9>/Dijkstra'
  }

  // End of Outputs for SubSystem: '<S2>/Enabled Subsystem'

  // MATLAB Function: '<S2>/Final_Static_Path' incorporates:
  //   Constant: '<S2>/Constant6'

  if (!rtDW.path_out1_not_empty) {
    if (rtb_Add > 0.0) {
      rtDW.path_out1.size = rtDW.SFunction_DIMS3_g;
      for (jj = 0; jj < rtDW.SFunction_DIMS3_g; jj++) {
        rtDW.path_out1.data[jj] = rtDW.path_2[jj];
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
    rtDW.path_out1.size = rtDW.SFunction_DIMS3_g;
    for (jj = 0; jj < rtDW.SFunction_DIMS3_g; jj++) {
      rtDW.path_out1.data[jj] = rtDW.path_2[jj];
    }

    rtDW.path_out1_not_empty = !(rtDW.path_out1.size == 0);
  }

  rtDW.SFunction_DIMS2_h = rtDW.path_out1.size;
  rtDW.SFunction_DIMS3_f = rtDW.path_out1.size;
  rtDW.SFunction_DIMS4_f[0] = 301;
  rtDW.SFunction_DIMS4_f[1] = 23;
  memcpy(&rtDW.Static_Path_0[0], &rtConstP.pooled2[0], 6923U * sizeof(real_T));
  rtDW.SFunction_DIMS6_c = rtDW.path_out1.size;

  // Gain: '<S1>/Gain' incorporates:
  //   Gain: '<Root>/Gain'
  //   Inport: '<Root>/angular_vz'

  rtb_Gain_k = -(0.017453292519943295 * rtU.angular_vz);

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
      rtb_X_state[2] = 0.0001;
    }

    rtb_X_state[3] = 0.001;
    rtb_X_state[4] = 0.0001;
    memset(&rtb_R_last_o[0], 0, 25U * sizeof(real_T));
    for (n = 0; n < 5; n++) {
      rtb_R_last_o[n + 5 * n] = rtb_X_state[n];
    }
  }

  rtb_X_state[0] = SLAM_X_out;
  rtb_X_state[1] = SLAM_Y_out;
  rtb_X_state[2] = SLAM_Heading_out;
  rtb_X_state[3] = rtb_Gain_k;
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
  i = 0;
  colj = 0;
  n = 1;
  exitg1 = false;
  while ((!exitg1) && (n <= 5)) {
    jj = (colj + n) - 1;
    ajj = 0.0;
    if (!(n - 1 < 1)) {
      ix = colj;
      iy = colj;
      for (end_ind_0 = 1; end_ind_0 < n; end_ind_0++) {
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
          ix = jj + 5;
          end_ind_0 = ((4 - n) * 5 + colj) + 6;
          for (iy = colj + 6; iy <= end_ind_0; iy += 5) {
            b_ix = colj;
            total_length = 0.0;
            d_ix = (iy + n) - 2;
            for (b_ia = iy; b_ia <= d_ix; b_ia++) {
              total_length += p_sqrt_data[b_ia - 1] * p_sqrt_data[b_ix];
              b_ix++;
            }

            p_sqrt_data[ix] += -total_length;
            ix += 5;
          }
        }

        ajj = 1.0 / ajj;
        ix = ((4 - n) * 5 + jj) + 6;
        for (end_ind_0 = jj + 5; end_ind_0 + 1 <= ix; end_ind_0 += 5) {
          p_sqrt_data[end_ind_0] *= ajj;
        }

        colj += 5;
      }

      n++;
    } else {
      p_sqrt_data[jj] = ajj;
      i = n;
      exitg1 = true;
    }
  }

  if (i == 0) {
    colj = 5;
  } else {
    colj = i - 1;
  }

  for (n = 1; n <= colj; n++) {
    for (end_ind_0 = n; end_ind_0 < colj; end_ind_0++) {
      p_sqrt_data[end_ind_0 + 5 * (n - 1)] = 0.0;
    }
  }

  if (1 > colj) {
    colj = 0;
    iy = 0;
  } else {
    iy = colj;
  }

  for (jj = 0; jj < iy; jj++) {
    for (ix = 0; ix < colj; ix++) {
      p_sqrt_data_0[ix + colj * jj] = p_sqrt_data[5 * jj + ix];
    }
  }

  for (jj = 0; jj < iy; jj++) {
    for (ix = 0; ix < colj; ix++) {
      n = colj * jj;
      p_sqrt_data[ix + n] = p_sqrt_data_0[n + ix];
    }
  }

  memset(&rtb_X_AUG[0], 0, 55U * sizeof(real_T));
  if (i != 0) {
    for (i = 0; i < 5; i++) {
      temp_dia[i] = std::abs(rtDW.UnitDelay33_DSTATE[5 * i + i]);
    }

    i = 0;
    n = 1;
    exitg1 = false;
    while ((!exitg1) && (n < 6)) {
      if (temp_dia[n - 1] < 1.0E-10) {
        i++;
        ii_data[i - 1] = (int8_T)n;
        if (i >= 5) {
          exitg1 = true;
        } else {
          n++;
        }
      } else {
        n++;
      }
    }

    if (!(1 > i)) {
      for (jj = 0; jj < i; jj++) {
        temp_dia[ii_data[jj] - 1] = 1.0E-10;
      }
    }

    memset(&p0[0], 0, 25U * sizeof(real_T));
    for (i = 0; i < 5; i++) {
      p0[i + 5 * i] = temp_dia[i];
    }

    i = 0;
    colj = 0;
    end_ind_0 = 1;
    exitg1 = false;
    while ((!exitg1) && (end_ind_0 < 6)) {
      jj = (colj + end_ind_0) - 1;
      ajj = 0.0;
      if (!(end_ind_0 - 1 < 1)) {
        n = colj;
        ix = colj;
        for (iy = 1; iy < end_ind_0; iy++) {
          ajj += p0[n] * p0[ix];
          n++;
          ix++;
        }
      }

      ajj = p0[jj] - ajj;
      if (ajj > 0.0) {
        ajj = std::sqrt(ajj);
        p0[jj] = ajj;
        if (end_ind_0 < 5) {
          if (end_ind_0 - 1 != 0) {
            iy = jj + 5;
            ix = ((4 - end_ind_0) * 5 + colj) + 6;
            for (b_ix = colj + 6; b_ix <= ix; b_ix += 5) {
              d_ix = colj;
              total_length = 0.0;
              n = (b_ix + end_ind_0) - 2;
              for (b_ia = b_ix; b_ia <= n; b_ia++) {
                total_length += p0[b_ia - 1] * p0[d_ix];
                d_ix++;
              }

              p0[iy] += -total_length;
              iy += 5;
            }
          }

          ajj = 1.0 / ajj;
          n = ((4 - end_ind_0) * 5 + jj) + 6;
          for (jj += 5; jj + 1 <= n; jj += 5) {
            p0[jj] *= ajj;
          }

          colj += 5;
        }

        end_ind_0++;
      } else {
        p0[jj] = ajj;
        i = end_ind_0;
        exitg1 = true;
      }
    }

    if (i == 0) {
      i = 5;
    } else {
      i--;
    }

    for (n = 0; n < i; n++) {
      for (colj = n + 1; colj < i; colj++) {
        p0[colj + 5 * n] = 0.0;
      }
    }

    colj = 5;
    iy = 5;
    memcpy(&p_sqrt_data[0], &p0[0], 25U * sizeof(real_T));
  }

  for (jj = 0; jj < colj; jj++) {
    for (ix = 0; ix < iy; ix++) {
      p_sqrt_data_0[ix + iy * jj] = p_sqrt_data[colj * ix + jj] *
        2.23606797749979;
    }
  }

  for (jj = 0; jj < colj; jj++) {
    for (ix = 0; ix < iy; ix++) {
      n = iy * jj;
      p_sqrt_data[ix + n] = p_sqrt_data_0[n + ix];
    }
  }

  for (jj = 0; jj < 5; jj++) {
    rtb_X_AUG[jj] = rtb_UnitDelay34[jj];
  }

  for (n = 0; n < 5; n++) {
    colj = iy - 1;
    for (jj = 0; jj <= colj; jj++) {
      temp_dia[jj] = p_sqrt_data[iy * n + jj];
    }

    jj = n + 2;
    for (ix = 0; ix < 5; ix++) {
      rtb_X_AUG[ix + 5 * (jj - 1)] = rtb_UnitDelay34[ix] + temp_dia[ix];
    }
  }

  for (i = 0; i < 5; i++) {
    colj = iy - 1;
    for (jj = 0; jj <= colj; jj++) {
      temp_dia[jj] = p_sqrt_data[iy * i + jj];
    }

    jj = i + 7;
    for (ix = 0; ix < 5; ix++) {
      rtb_X_AUG[ix + 5 * (jj - 1)] = rtb_UnitDelay34[ix] - temp_dia[ix];
    }
  }

  // End of MATLAB Function: '<S3>/SLAM_Generate_sigma_pt_UKF'

  // MATLAB Function: '<S3>/SLAM_UKF' incorporates:
  //   Constant: '<Root>/[Para] D_GC'
  //   Constant: '<S1>/Constant25'
  //   MATLAB Function: '<S3>/SLAM_Check'
  //   SignalConversion: '<S6>/TmpSignal ConversionAt SFunction Inport5'

  rtb_Gain_k = 0.01 * rtb_Gain_k * 3.8;
  for (end_ind_0 = 0; end_ind_0 < 11; end_ind_0++) {
    rtb_X_AUG[5 * end_ind_0] = (rtb_X_AUG[5 * end_ind_0 + 4] * 0.01 * std::cos
      (rtb_X_AUG[5 * end_ind_0 + 2]) + rtb_X_AUG[5 * end_ind_0]) + std::cos
      (rtb_X_AUG[5 * end_ind_0 + 2] + 1.5707963267948966) * rtb_Gain_k;
    rtb_X_AUG[1 + 5 * end_ind_0] = (rtb_X_AUG[5 * end_ind_0 + 4] * 0.01 * std::
      sin(rtb_X_AUG[5 * end_ind_0 + 2]) + rtb_X_AUG[5 * end_ind_0 + 1]) + std::
      sin(rtb_X_AUG[5 * end_ind_0 + 2] + 1.5707963267948966) * rtb_Gain_k;
    rtb_X_AUG[2 + 5 * end_ind_0] += rtb_X_AUG[5 * end_ind_0 + 3] * 0.01;
  }

  for (jj = 0; jj < 10; jj++) {
    rtb_X_AUG_0[jj] = rtb_X_AUG[(1 + jj) * 5];
  }

  rtb_X[0] = rtb_X_AUG[0] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (jj = 0; jj < 10; jj++) {
    rtb_X_AUG_0[jj] = rtb_X_AUG[(1 + jj) * 5 + 1];
  }

  rtb_X[1] = rtb_X_AUG[1] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (jj = 0; jj < 10; jj++) {
    rtb_X_AUG_0[jj] = rtb_X_AUG[(1 + jj) * 5 + 2];
  }

  rtb_X[2] = rtb_X_AUG[2] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (jj = 0; jj < 10; jj++) {
    rtb_X_AUG_0[jj] = rtb_X_AUG[(1 + jj) * 5 + 3];
  }

  rtb_X[3] = rtb_X_AUG[3] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (jj = 0; jj < 10; jj++) {
    rtb_X_AUG_0[jj] = rtb_X_AUG[(1 + jj) * 5 + 4];
  }

  rtb_X[4] = rtb_X_AUG[4] * 0.0 + sum(rtb_X_AUG_0) * 0.1;
  for (jj = 0; jj < 5; jj++) {
    rtb_Gain_k = rtb_X_AUG[jj] - rtb_X[jj];
    rtb_UnitDelay34[jj] = rtb_Gain_k;
    temp_dia[jj] = rtb_Gain_k;
  }

  for (jj = 0; jj < 5; jj++) {
    for (ix = 0; ix < 5; ix++) {
      p_sqrt_data[jj + 5 * ix] = rtb_UnitDelay34[jj] * temp_dia[ix];
    }
  }

  for (jj = 0; jj < 5; jj++) {
    for (ix = 0; ix < 5; ix++) {
      p0[ix + 5 * jj] = p_sqrt_data[5 * jj + ix] * 2.0;
    }
  }

  for (n = 0; n < 10; n++) {
    for (jj = 0; jj < 5; jj++) {
      rtb_Gain_k = rtb_X_AUG[(n + 1) * 5 + jj] - rtb_X[jj];
      rtb_UnitDelay34[jj] = rtb_Gain_k;
      temp_dia[jj] = rtb_Gain_k;
    }

    for (jj = 0; jj < 5; jj++) {
      for (ix = 0; ix < 5; ix++) {
        p_sqrt_data[jj + 5 * ix] = rtb_UnitDelay34[jj] * temp_dia[ix];
      }
    }

    for (jj = 0; jj < 5; jj++) {
      for (ix = 0; ix < 5; ix++) {
        i = 5 * jj + ix;
        p0[ix + 5 * jj] = p_sqrt_data[i] * 0.1 + p0[i];
      }
    }
  }

  for (jj = 0; jj < 25; jj++) {
    p0[jj] += rtb_Q_last_o[jj];
  }

  if (rtb_X[2] < 0.0) {
    rtb_X[2] += 6.2831853071795862;
  } else {
    if (rtb_X[2] >= 6.2831853071795862) {
      rtb_X[2] -= 6.2831853071795862;
    }
  }

  if (b_idx > 0) {
    for (jj = 0; jj < 25; jj++) {
      p_sqrt_data[jj] = p0[jj] + rtb_R_last_o[jj];
    }

    invNxN(p_sqrt_data, p_sqrt_data_0);
    for (jj = 0; jj < 5; jj++) {
      for (ix = 0; ix < 5; ix++) {
        i = jj + 5 * ix;
        rtb_K[i] = 0.0;
        for (b_idx = 0; b_idx < 5; b_idx++) {
          rtb_K[i] = p0[5 * b_idx + jj] * p_sqrt_data_0[5 * ix + b_idx] + rtb_K
            [5 * ix + jj];
        }
      }

      rtb_UnitDelay34[jj] = rtb_X_state[jj] - rtb_X[jj];
    }

    for (jj = 0; jj < 5; jj++) {
      rtb_Gain_k = 0.0;
      for (ix = 0; ix < 5; ix++) {
        rtb_Gain_k += rtb_K[5 * ix + jj] * rtb_UnitDelay34[ix];
      }

      rtb_X[jj] += rtb_Gain_k;
    }

    for (jj = 0; jj < 25; jj++) {
      I[jj] = 0;
    }

    for (b_idx = 0; b_idx < 5; b_idx++) {
      I[b_idx + 5 * b_idx] = 1;
    }

    for (jj = 0; jj < 5; jj++) {
      for (ix = 0; ix < 5; ix++) {
        i = 5 * jj + ix;
        p_sqrt_data[ix + 5 * jj] = (real_T)I[i] - rtb_K[i];
      }
    }

    for (jj = 0; jj < 5; jj++) {
      for (ix = 0; ix < 5; ix++) {
        i = ix + 5 * jj;
        rtb_K[i] = 0.0;
        for (b_idx = 0; b_idx < 5; b_idx++) {
          rtb_K[i] = p_sqrt_data[5 * b_idx + ix] * p0[5 * jj + b_idx] + rtb_K[5 *
            jj + ix];
        }
      }
    }

    for (jj = 0; jj < 5; jj++) {
      for (ix = 0; ix < 5; ix++) {
        p0[ix + 5 * jj] = rtb_K[5 * jj + ix];
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
      rtb_Gain_k = 1.0 / (rtb_Gain1 * rtb_num_lane_direction_f[3] -
                          rtb_num_lane_direction_f[2]);
      ajj = rtb_num_lane_direction_f[3] / rtb_num_lane_direction_f[1] *
        rtb_Gain_k;
      total_length = -rtb_Gain_k;
      y_idx_2 = -rtb_num_lane_direction_f[2] / rtb_num_lane_direction_f[1] *
        rtb_Gain_k;
      rtb_Gain_k *= rtb_Gain1;
    } else {
      rtb_Gain1 = rtb_num_lane_direction_f[1] / rtb_num_lane_direction_f[0];
      rtb_Gain_k = 1.0 / (rtb_num_lane_direction_f[3] - rtb_Gain1 *
                          rtb_num_lane_direction_f[2]);
      ajj = rtb_num_lane_direction_f[3] / rtb_num_lane_direction_f[0] *
        rtb_Gain_k;
      total_length = -rtb_Gain1 * rtb_Gain_k;
      y_idx_2 = -rtb_num_lane_direction_f[2] / rtb_num_lane_direction_f[0] *
        rtb_Gain_k;
    }

    for (jj = 0; jj < 2; jj++) {
      K1[jj] = 0.0;
      K1[jj] += p0[jj + 18] * ajj;
      K1[jj] += p0[jj + 23] * total_length;
      K1[jj + 2] = 0.0;
      K1[jj + 2] += p0[jj + 18] * y_idx_2;
      K1[jj + 2] += p0[jj + 23] * rtb_Gain_k;
      rtb_X_state_0[jj] = rtb_X_state[3 + jj] - rtb_X[3 + jj];
    }

    rtb_X[3] += K1[0] * rtb_X_state_0[0] + K1[2] * rtb_X_state_0[1];
    rtb_X[4] += K1[1] * rtb_X_state_0[0] + K1[3] * rtb_X_state_0[1];
    rtb_num_lane_direction_f[0] = 1.0 - K1[0];
    rtb_num_lane_direction_f[1] = 0.0 - K1[1];
    rtb_num_lane_direction_f[2] = 0.0 - K1[2];
    rtb_num_lane_direction_f[3] = 1.0 - K1[3];
    for (jj = 0; jj < 2; jj++) {
      K1[jj] = 0.0;
      K1[jj] += rtb_num_lane_direction_f[jj] * p0[18];
      K1[jj] += rtb_num_lane_direction_f[jj + 2] * p0[19];
      K1[jj + 2] = 0.0;
      K1[jj + 2] += rtb_num_lane_direction_f[jj] * p0[23];
      K1[jj + 2] += rtb_num_lane_direction_f[jj + 2] * p0[24];
    }

    p0[18] = K1[0];
    p0[19] = K1[1];
    p0[23] = K1[2];
    p0[24] = K1[3];
  }

  // End of MATLAB Function: '<S3>/SLAM_UKF'

  // MATLAB Function: '<S3>/SLAM_UKF_MM' incorporates:
  //   Constant: '<S3>/Constant4'

  for (n = 0; n < 301; n++) {
    if (rtConstP.pooled2[1505 + n] == (rtInf)) {
      rtDW.table[n] = rtConstP.pooled2[301 + n];
      rtDW.table[301 + n] = rtb_X[1];
    } else if (rtConstP.pooled2[1505 + n] == 0.0) {
      rtDW.table[n] = rtb_X[0];
      rtDW.table[301 + n] = rtConstP.pooled2[602 + n];
    } else {
      rtb_Gain1 = -1.0 / rtConstP.pooled2[1505 + n];
      rtb_Gain_k = rtb_X[1] - rtb_Gain1 * rtb_X[0];
      ajj = rtConstP.pooled2[1505 + n] - rtb_Gain1;
      rtDW.table[n] = (rtb_Gain_k - rtConstP.pooled2[1806 + n]) / ajj;
      rtDW.table[301 + n] = (rtConstP.pooled2[1505 + n] * rtb_Gain_k -
        rtConstP.pooled2[1806 + n] * rtb_Gain1) / ajj;
    }

    total_length = rtDW.table[n] - rtb_X[0];
    y_idx_2 = rtDW.table[301 + n] - rtb_X[1];
    shortest_distance[n] = std::sqrt(total_length * total_length + y_idx_2 *
      y_idx_2);
  }

  rtb_X_state_0[0] = rtb_X[0];
  rtb_X_state_0[1] = rtb_X[1];
  MM(rtb_X[2] * 180.0 / 3.1415926535897931, rtb_X_state_0, rtDW.table,
     shortest_distance, rtConstP.pooled2, &rtb_Gain1, &rtb_Gain_k, rtb_Oi_near_b,
     &ajj, &total_length, &y_idx_2, rtb_num_lane_direction_f, &seg_heading);

  // End of MATLAB Function: '<S3>/SLAM_UKF_MM'

  // MATLAB Function: '<S2>/MM' incorporates:
  //   Gain: '<S2>/Gain'
  //   Gain: '<S2>/Gain3'

  b_idx = rtDW.SFunction_DIMS4_f[0];
  oi_xy_size[0] = rtDW.SFunction_DIMS4_f[0];
  oi_xy_size[1] = 2;
  iy = (rtDW.SFunction_DIMS4_f[0] << 1) - 1;
  if (0 <= iy) {
    memset(&rtDW.oi_xy_data[0], 0, (iy + 1) * sizeof(real_T));
  }

  colj = rtDW.SFunction_DIMS4_f[0];
  if (0 <= b_idx - 1) {
    memset(&dist_op_data[0], 0, b_idx * sizeof(real_T));
  }

  for (n = 0; n < rtDW.SFunction_DIMS4_f[0]; n++) {
    if (rtDW.Static_Path_0[rtDW.SFunction_DIMS4_f[0] * 5 + n] == (rtInf)) {
      rtDW.oi_xy_data[n] = rtDW.Static_Path_0[n + rtDW.SFunction_DIMS4_f[0]];
      rtDW.oi_xy_data[n + b_idx] = rtb_X[1];
    } else if (rtDW.Static_Path_0[rtDW.SFunction_DIMS4_f[0] * 5 + n] == 0.0) {
      rtDW.oi_xy_data[n] = rtb_X[0];
      rtDW.oi_xy_data[n + b_idx] = rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_f[0]
        << 1) + n];
    } else {
      rtb_Gain1 = -1.0 / rtDW.Static_Path_0[rtDW.SFunction_DIMS4_f[0] * 5 + n];
      rtb_Gain_k = rtb_X[1] - rtb_Gain1 * rtb_X[0];
      ajj = rtDW.Static_Path_0[rtDW.SFunction_DIMS4_f[0] * 5 + n] - rtb_Gain1;
      rtDW.oi_xy_data[n] = (rtb_Gain_k -
                            rtDW.Static_Path_0[rtDW.SFunction_DIMS4_f[0] * 6 + n])
        / ajj;
      rtDW.oi_xy_data[n + b_idx] = (rtDW.Static_Path_0[rtDW.SFunction_DIMS4_f[0]
        * 5 + n] * rtb_Gain_k - rtDW.Static_Path_0[rtDW.SFunction_DIMS4_f[0] * 6
        + n] * rtb_Gain1) / ajj;
    }
  }

  for (i = 0; i < oi_xy_size[0]; i++) {
    total_length = rtDW.oi_xy_data[i] - rtb_X[0];
    y_idx_2 = rtDW.oi_xy_data[i + b_idx] - rtb_X[1];
    dist_op_data[i] = std::sqrt(total_length * total_length + y_idx_2 * y_idx_2);
  }

  rtb_X_state_0[0] = rtb_X[0];
  rtb_X_state_0[1] = rtb_X[1];
  MM_o(0.017453292519943295 * (57.295779513082323 * rtb_X[2]) * 180.0 /
       3.1415926535897931, rtb_X_state_0, rtDW.oi_xy_data, oi_xy_size,
       dist_op_data, &colj, rtDW.Static_Path_0, rtDW.SFunction_DIMS4_f,
       &rtb_Gain1, &rtb_Gain_k, rtb_Oi_near_b, &ajj, &total_length, &y_idx_2,
       rtb_num_lane_direction_f, &seg_heading);

  // MATLAB Function: '<S2>/MATLAB Function2' incorporates:
  //   Inport: '<Root>/ID_turn'
  //   Inport: '<Root>/Look_ahead_time_straight'
  //   Inport: '<Root>/Look_ahead_time_turn'
  //   MATLAB Function: '<S2>/MM'

  if ((rtb_Gain1 >= rtU.ID_turn[0]) && (rtb_Gain1 <= rtU.ID_turn[1])) {
    rtb_Gain_k = rtU.Look_ahead_time_turn;
  } else if ((rtb_Gain1 >= rtU.ID_turn[2]) && (rtb_Gain1 <= rtU.ID_turn[3])) {
    rtb_Gain_k = rtU.Look_ahead_time_turn;
  } else if ((rtb_Gain1 >= rtU.ID_turn[4]) && (rtb_Gain1 <= rtU.ID_turn[5])) {
    rtb_Gain_k = rtU.Look_ahead_time_turn;
  } else if (rtb_Gain1 >= rtU.ID_turn[6]) {
    if (rtb_Gain1 <= rtU.ID_turn[7]) {
      rtb_Gain_k = rtU.Look_ahead_time_turn;
    } else {
      rtb_Gain_k = rtU.Look_ahead_time_straight;
    }
  } else {
    rtb_Gain_k = rtU.Look_ahead_time_straight;
  }

  // End of MATLAB Function: '<S2>/MATLAB Function2'

  // MATLAB Function: '<S2>/target_seg_id_search' incorporates:
  //   Inport: '<Root>/Speed_mps'
  //   MATLAB Function: '<S2>/MM'

  ajj = rtU.Speed_mps * rtb_Gain_k + 3.0;
  iy = rtDW.SFunction_DIMS4_f[0];
  if (0 <= iy - 1) {
    memcpy(&dist_op_data[0], &rtDW.Static_Path_0[0], iy * sizeof(real_T));
  }

  if (rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_f[0] * 3 +
                          rtDW.SFunction_DIMS4_f[0]) - 1] ==
      rtDW.Static_Path_0[rtDW.SFunction_DIMS4_f[0]]) {
    colj = (rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_f[0] << 2) +
             rtDW.SFunction_DIMS4_f[0]) - 1] ==
            rtDW.Static_Path_0[rtDW.SFunction_DIMS4_f[0] << 1]);
  } else {
    colj = 0;
  }

  n = rtDW.SFunction_DIMS4_f[0];
  for (jj = 0; jj < n; jj++) {
    x_data[jj] = (dist_op_data[jj] == rtb_Gain1);
  }

  i = 1;
  b_x_0 = x_data[0];
  for (end_ind_0 = 2; end_ind_0 <= n; end_ind_0++) {
    if ((int32_T)b_x_0 < (int32_T)x_data[end_ind_0 - 1]) {
      b_x_0 = x_data[end_ind_0 - 1];
      i = end_ind_0;
    }
  }

  total_length = rtb_Oi_near_b[0] - rtDW.Static_Path_0[(rtDW.SFunction_DIMS4_f[0]
    * 3 + i) - 1];
  y_idx_2 = rtb_Oi_near_b[1] - rtDW.Static_Path_0[((rtDW.SFunction_DIMS4_f[0] <<
    2) + i) - 1];
  total_length = std::sqrt(total_length * total_length + y_idx_2 * y_idx_2);
  end_ind_0 = i;
  jj = 0;
  b_idx = 0;
  n = 0;
  exitg1 = false;
  while ((!exitg1) && (n <= rtDW.SFunction_DIMS4_f[0] - 1)) {
    if (total_length > ajj) {
      b_idx = end_ind_0;
      exitg1 = true;
    } else {
      jj = i + n;
      ix = jj + 1;
      if (ix <= rtDW.SFunction_DIMS4_f[0]) {
        total_length += rtDW.Static_Path_0[jj + (rtDW.SFunction_DIMS4_f[0] << 3)];
        end_ind_0 = ix;
        jj = 1;
        n++;
      } else if (colj == 1) {
        jj -= rtDW.SFunction_DIMS4_f[0];
        total_length += rtDW.Static_Path_0[jj + (rtDW.SFunction_DIMS4_f[0] << 3)];
        end_ind_0 = jj + 1;
        jj = 2;
        n++;
      } else {
        b_idx = end_ind_0;
        jj = 3;
        exitg1 = true;
      }
    }
  }

  n = rtDW.SFunction_DIMS4_f[0];
  if (0 <= n - 1) {
    memset(&Forward_Static_Path_id_0_data[0], 0, n * sizeof(real_T));
  }

  if ((jj == 1) || (jj == 0)) {
    if (i > b_idx) {
      colj = 0;
      n = 0;
    } else {
      colj = i - 1;
      n = b_idx;
    }

    b_ix = n - colj;
    if (i > b_idx) {
      end_ind_0 = 1;
      n = 0;
    } else {
      end_ind_0 = i;
      n = b_idx;
    }

    iy = n - end_ind_0;
    for (jj = 0; jj <= iy; jj++) {
      Forward_Static_Path_id_0_data[jj] = dist_op_data[(end_ind_0 + jj) - 1];
    }

    if (i > b_idx) {
      i = 1;
      b_idx = 0;
    }

    b_idx = (b_idx - i) + 1;
  } else if (jj == 2) {
    if (i > rtDW.SFunction_DIMS4_f[0]) {
      n = 0;
      colj = 0;
    } else {
      n = i - 1;
      colj = rtDW.SFunction_DIMS4_f[0];
    }

    if (1 > b_idx) {
      jj = 0;
    } else {
      jj = b_idx;
    }

    b_ix = (colj - n) + jj;
    if (i > rtDW.SFunction_DIMS4_f[0]) {
      n = 0;
      ix = 0;
    } else {
      n = i - 1;
      ix = rtDW.SFunction_DIMS4_f[0];
    }

    colj = ((rtDW.SFunction_DIMS4_f[0] - i) + b_idx) + 1;
    if (1 > colj) {
      tmp = 0;
    } else {
      tmp = (int16_T)colj;
    }

    end_ind_0 = tmp;
    iy = tmp - 1;
    for (jj = 0; jj <= iy; jj++) {
      cb_data[jj] = (int16_T)jj;
    }

    if (1 > b_idx) {
      jj = 0;
    } else {
      jj = b_idx;
    }

    iy = jj - 1;
    colj = ix - n;
    for (jj = 0; jj < colj; jj++) {
      rtDW.table[jj] = dist_op_data[n + jj];
    }

    for (jj = 0; jj <= iy; jj++) {
      rtDW.table[(jj + ix) - n] = dist_op_data[jj];
    }

    for (jj = 0; jj < end_ind_0; jj++) {
      Forward_Static_Path_id_0_data[cb_data[jj]] = rtDW.table[jj];
    }

    if (i > rtDW.SFunction_DIMS4_f[0]) {
      i = 1;
      n = 1;
    } else {
      n = rtDW.SFunction_DIMS4_f[0] + 1;
    }

    if (1 > b_idx) {
      b_idx = 0;
    }

    b_idx += n - i;
  } else {
    if (i > rtDW.SFunction_DIMS4_f[0]) {
      b_idx = 0;
      n = 0;
    } else {
      b_idx = i - 1;
      n = rtDW.SFunction_DIMS4_f[0];
    }

    b_ix = n - b_idx;
    if (i > rtDW.SFunction_DIMS4_f[0]) {
      b_idx = 1;
      n = 0;
    } else {
      b_idx = i;
      n = rtDW.SFunction_DIMS4_f[0];
    }

    iy = n - b_idx;
    for (jj = 0; jj <= iy; jj++) {
      Forward_Static_Path_id_0_data[jj] = dist_op_data[(b_idx + jj) - 1];
    }

    if (i > rtDW.SFunction_DIMS4_f[0]) {
      i = 1;
      b_idx = 1;
    } else {
      b_idx = rtDW.SFunction_DIMS4_f[0] + 1;
    }

    b_idx -= i;
  }

  if (1 > b_idx) {
    b_idx = 0;
  }

  rtDW.SFunction_DIMS4 = b_idx;
  if (0 <= b_idx - 1) {
    memcpy(&shortest_distance[0], &Forward_Static_Path_id_0_data[0], b_idx *
           sizeof(real_T));
  }

  b_idx = b_ix + 1;
  rtDW.SFunction_DIMS2 = b_idx;
  rtDW.SFunction_DIMS3 = b_idx;
  rtDW.SFunction_DIMS6[0] = rtDW.SFunction_DIMS4_f[0];
  rtDW.SFunction_DIMS6[1] = 1;

  // Outport: '<Root>/Target_seg_id' incorporates:
  //   MATLAB Function: '<S2>/target_seg_id_search'

  rtY.Target_seg_id = shortest_distance[rtDW.SFunction_DIMS4 - 1];

  // Outport: '<Root>/Look_ahead_time'
  rtY.Look_ahead_time = rtb_Gain_k;

  // Outport: '<Root>/seg_id_near' incorporates:
  //   MATLAB Function: '<S2>/MM'

  rtY.seg_id_near = rtb_Gain1;

  // Update for Memory: '<S2>/Memory1' incorporates:
  //   Constant: '<S2>/Constant7'

  rtDW.Memory1_PreviousInput = 1.0;

  // Update for Memory: '<S2>/Memory' incorporates:
  //   Constant: '<S2>/Constant8'

  rtDW.Memory_PreviousInput = 301.0;

  // MATLAB Function: '<S2>/Final_Static_Path'
  if (rtb_Add > 0.0) {
    // Update for UnitDelay: '<S2>/Unit Delay'
    rtDW.UnitDelay_DSTATE = rtDW.dist;
  }

  for (i = 0; i < 5; i++) {
    // Outport: '<Root>/X_UKF_SLAM'
    rtY.X_UKF_SLAM[i] = rtb_X[i];

    // Update for UnitDelay: '<S3>/Unit Delay1'
    rtDW.UnitDelay1_DSTATE[i] = rtb_X[i];
  }

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
}

// Model initialize function
void UKF_MMModelClass::initialize()
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
UKF_MMModelClass::UKF_MMModelClass()
{
}

// Destructor
UKF_MMModelClass::~UKF_MMModelClass()
{
  // Currently there is no destructor body generated.
}

// Real-Time Model get method
RT_MODEL * UKF_MMModelClass::getRTM()
{
  return (&rtM);
}

//
// File trailer for generated code.
//
// [EOF]
//
