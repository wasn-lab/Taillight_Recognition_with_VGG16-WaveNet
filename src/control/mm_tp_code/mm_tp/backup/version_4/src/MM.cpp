//
// File: MM.cpp
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
#include "MM.h"
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
// Function for MATLAB Function: '<S1>/MM'
  int32_T MMModelClass::nonSingletonDim(const int32_T *x_size)
{
  int32_T dim;
  dim = 2;
  if (*x_size != 1) {
    dim = 1;
  }

  return dim;
}

// Function for MATLAB Function: '<S1>/MM'
void MMModelClass::merge(int32_T idx_data[], real_T x_data[], int32_T offset,
  int32_T np, int32_T nq, int32_T iwork_data[], real_T xwork_data[])
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

// Function for MATLAB Function: '<S1>/MM'
void MMModelClass::merge_block(int32_T idx_data[], real_T x_data[], int32_T
  offset, int32_T n, int32_T preSortLevel, int32_T iwork_data[], real_T
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
        merge(idx_data, x_data, offset + tailOffset, bLen, nTail - bLen,
              iwork_data, xwork_data);
      }
    }

    tailOffset = bLen << 1;
    nPairs >>= 1;
    for (nTail = 1; nTail <= nPairs; nTail++) {
      merge(idx_data, x_data, offset + (nTail - 1) * tailOffset, bLen, bLen,
            iwork_data, xwork_data);
    }

    bLen = tailOffset;
  }

  if (n > bLen) {
    merge(idx_data, x_data, offset, bLen, n - bLen, iwork_data, xwork_data);
  }
}

// Function for MATLAB Function: '<S1>/MM'
void MMModelClass::merge_pow2_block(int32_T idx_data[], real_T x_data[], int32_T
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

// Function for MATLAB Function: '<S1>/MM'
void MMModelClass::sortIdx(real_T x_data[], int32_T *x_size, int32_T idx_data[],
  int32_T *idx_size)
{
  int32_T iwork_data[1000];
  int32_T nBlocks;
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
      memcpy(&rtDW.c_x_data[0], &x_data[0], *x_size * sizeof(real_T));
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
      memset(&rtDW.xwork_data[0], 0, b_idx_0 * sizeof(real_T));
    }

    nBlocks = 1;
    ib = 0;
    for (wOffset = 0; wOffset < *x_size; wOffset++) {
      if (rtIsNaN(rtDW.c_x_data[wOffset])) {
        i3 = *x_size - nBlocks;
        idx_data[i3] = wOffset + 1;
        rtDW.xwork_data[i3] = rtDW.c_x_data[wOffset];
        nBlocks++;
      } else {
        ib++;
        idx4[ib - 1] = (int16_T)(wOffset + 1);
        x4[ib - 1] = rtDW.c_x_data[wOffset];
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
          rtDW.c_x_data[ib - 2] = x4[i3];
          rtDW.c_x_data[ib - 1] = x4[itmp];
          rtDW.c_x_data[ib] = x4[i1];
          rtDW.c_x_data[ib + 1] = x4[i4];
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
        rtDW.c_x_data[itmp] = x4[i3];
      }
    }

    ib = ((nBlocks - 1) >> 1) + 1;
    for (i1 = 1; i1 < ib; i1++) {
      i4 = wOffset + i1;
      itmp = idx_data[i4];
      i3 = *x_size - i1;
      idx_data[i4] = idx_data[i3];
      idx_data[i3] = itmp;
      rtDW.c_x_data[i4] = rtDW.xwork_data[i3];
      rtDW.c_x_data[i3] = rtDW.xwork_data[i4];
    }

    if (((nBlocks - 1) & 1U) != 0U) {
      rtDW.c_x_data[wOffset + ib] = rtDW.xwork_data[wOffset + ib];
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
            merge_pow2_block(idx_data, rtDW.c_x_data, (wOffset - 1) << 8);
          }

          nBlocks <<= 8;
          wOffset = ib - nBlocks;
          if (wOffset > 0) {
            if (0 <= b_x_idx_0 - 1) {
              memset(&iwork_data[0], 0, b_x_idx_0 * sizeof(int32_T));
            }

            merge_block(idx_data, rtDW.c_x_data, nBlocks, wOffset, 2, iwork_data,
                        rtDW.xwork_data);
          }

          wOffset = 8;
        }
      }

      merge_block(idx_data, rtDW.c_x_data, 0, ib, wOffset, iwork_data,
                  rtDW.xwork_data);
    }

    if (0 <= c_x_size - 1) {
      memcpy(&x_data[0], &rtDW.c_x_data[0], c_x_size * sizeof(real_T));
    }
  }
}

// Function for MATLAB Function: '<S1>/MM'
void MMModelClass::sort(real_T x_data[], int32_T *x_size, int32_T idx_data[],
  int32_T *idx_size)
{
  int32_T dim;
  int32_T vstride;
  int32_T iidx_data[1000];
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
      rtDW.vwork_data[c_k] = x_data[c_k * vstride + dim];
    }

    sortIdx(rtDW.vwork_data, &vwork_size, iidx_data, &c_k);
    for (c_k = 0; c_k < b; c_k++) {
      tmp = dim + c_k * vstride;
      x_data[tmp] = rtDW.vwork_data[c_k];
      idx_data[tmp] = iidx_data[c_k];
    }
  }
}

// Function for MATLAB Function: '<S1>/MM'
void MMModelClass::power_n(const real_T a_data[], const int32_T *a_size, real_T
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

// Function for MATLAB Function: '<S1>/MM'
void MMModelClass::power_n5(const real_T a_data[], const int32_T *a_size, real_T
  y_data[], int32_T *y_size)
{
  int32_T loop_ub;
  int16_T a_idx_0;
  a_idx_0 = (int16_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&rtDW.z1_data_c[0], &y_data[0], a_idx_0 * sizeof(real_T));
  }

  for (loop_ub = 0; loop_ub < a_idx_0; loop_ub++) {
    rtDW.z1_data_c[loop_ub] = std::sqrt(a_data[loop_ub]);
  }

  *y_size = (int16_T)*a_size;
  if (0 <= a_idx_0 - 1) {
    memcpy(&y_data[0], &rtDW.z1_data_c[0], a_idx_0 * sizeof(real_T));
  }
}

// Function for MATLAB Function: '<S1>/MM'
void MMModelClass::rel_dist_xy(const real_T ref_xy[2], const real_T pt_xy_data[],
  const int32_T pt_xy_size[2], real_T dist_data[], int32_T *dist_size)
{
  int32_T loop_ub;
  int32_T i;
  int32_T pt_xy_size_0;
  int32_T tmp_size;
  int32_T pt_xy_size_1;
  loop_ub = pt_xy_size[0];
  pt_xy_size_0 = pt_xy_size[0];
  for (i = 0; i < loop_ub; i++) {
    rtDW.pt_xy_data[i] = pt_xy_data[i] - ref_xy[0];
  }

  power_n(rtDW.pt_xy_data, &pt_xy_size_0, rtDW.tmp_data, &tmp_size);
  loop_ub = pt_xy_size[0];
  pt_xy_size_1 = pt_xy_size[0];
  for (i = 0; i < loop_ub; i++) {
    rtDW.pt_xy_data[i] = pt_xy_data[i + pt_xy_size[0]] - ref_xy[1];
  }

  power_n(rtDW.pt_xy_data, &pt_xy_size_1, rtDW.tmp_data_m, &pt_xy_size_0);
  for (i = 0; i < tmp_size; i++) {
    rtDW.pt_xy_data[i] = rtDW.tmp_data[i] + rtDW.tmp_data_m[i];
  }

  power_n5(rtDW.pt_xy_data, &tmp_size, dist_data, dist_size);
}

// Function for MATLAB Function: '<S1>/MM'
real_T MMModelClass::rel_dist_xy_f(const real_T ref_xy[2], const real_T pt_xy[2])
{
  real_T a;
  real_T b_a;
  a = pt_xy[0] - ref_xy[0];
  b_a = pt_xy[1] - ref_xy[1];
  return std::sqrt(a * a + b_a * b_a);
}

// Function for MATLAB Function: '<S1>/MM'
void MMModelClass::MM_g(real_T heading, const real_T X_pos[2], const real_T
  oi_xy_data[], const int32_T oi_xy_size[2], const real_T dist_op_data[], const
  int32_T *dist_op_size, const real_T Map_data_data[], const int32_T
  Map_data_size[2], real_T *seg_id_near, real_T *op_distance, real_T oi_near[2],
  real_T *note, real_T *seg_direction, real_T *head_err, real_T
  num_lane_direction[4], real_T *seg_heading)
{
  real_T op_distance_n;
  real_T C;
  boolean_T x_data[1000];
  int32_T ii_data[1000];
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
    memcpy(&rtDW.ind_temp_data[0], &dist_op_data[0], *dist_op_size * sizeof
           (real_T));
  }

  sort(rtDW.ind_temp_data, &g_idx, ii_data, &ii_size);
  for (g_idx = 0; g_idx < ii_size; g_idx++) {
    rtDW.ind_temp_data[g_idx] = ii_data[g_idx];
  }

  loop_ub = ii_size - 1;
  for (g_idx = 0; g_idx < ii_size; g_idx++) {
    rtDW.b_index_data[g_idx] = (int32_T)rtDW.ind_temp_data[g_idx];
  }

  if (0 <= loop_ub) {
    memset(&rtDW.b_index_data[ii_size], 0, (((loop_ub + ii_size) - ii_size) + 1)
           * sizeof(int32_T));
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
    memcpy(&rtDW.seg_id_data[0], &Map_data_data[0], loop_ub * sizeof(real_T));
    memcpy(&rtDW.SEG_GPS_HEAD_data[0], &rtDW.seg_id_data[0], loop_ub * sizeof
           (real_T));
  }

  nx = Map_data_size[0] - 1;
  for (g_idx = 0; g_idx <= nx; g_idx++) {
    rtDW.SEG_GPS_HEAD_data[g_idx + loop_ub] = Map_data_data[Map_data_size[0] * 7
      + g_idx];
  }

  for (nx = 0; nx < Map_data_size[0]; nx++) {
    op_distance_n = Map_data_data[(rtDW.b_index_data[nx] + Map_data_size[0]) - 1]
      - oi_xy_data[rtDW.b_index_data[nx] - 1];
    C = Map_data_data[((Map_data_size[0] << 1) + rtDW.b_index_data[nx]) - 1] -
      oi_xy_data[(rtDW.b_index_data[nx] + oi_xy_size[0]) - 1];
    op_distance_n_0 = Map_data_data[(Map_data_size[0] * 3 + rtDW.b_index_data[nx])
      - 1] - oi_xy_data[rtDW.b_index_data[nx] - 1];
    d_a = Map_data_data[((Map_data_size[0] << 2) + rtDW.b_index_data[nx]) - 1] -
      oi_xy_data[(rtDW.b_index_data[nx] + oi_xy_size[0]) - 1];
    if (std::sqrt(op_distance_n * op_distance_n + C * C) <= Map_data_data
        [((Map_data_size[0] << 3) + rtDW.b_index_data[nx]) - 1]) {
      rtDW.b_index_data[nx + ii_size] = (std::sqrt(op_distance_n_0 *
        op_distance_n_0 + d_a * d_a) <= Map_data_data[((Map_data_size[0] << 3) +
        rtDW.b_index_data[nx]) - 1]);
    } else {
      rtDW.b_index_data[nx + ii_size] = 0;
    }
  }

  if (1 > ii_size) {
    nx = 0;
  } else {
    nx = ii_size;
  }

  for (g_idx = 0; g_idx < nx; g_idx++) {
    x_data[g_idx] = (rtDW.b_index_data[g_idx + ii_size] == 1);
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
    rtDW.ind_temp_data[g_idx] = ii_data[g_idx];
  }

  if (ii_size == 0) {
    *note = 1.0;
    rel_dist_xy(X_pos, rtDW.xy_ini_data, xy_ini_size, rtDW.dist_ini_data, &g_idx);
    rel_dist_xy(X_pos, rtDW.xy_end_data, xy_end_size, rtDW.dist_end_data,
                &ii_size);
    if (g_idx <= 2) {
      if (g_idx == 1) {
        op_distance_n = rtDW.dist_ini_data[0];
        nx = 0;
      } else if ((rtDW.dist_ini_data[0] > rtDW.dist_ini_data[1]) || (rtIsNaN
                  (rtDW.dist_ini_data[0]) && (!rtIsNaN(rtDW.dist_ini_data[1]))))
      {
        op_distance_n = rtDW.dist_ini_data[1];
        nx = 1;
      } else {
        op_distance_n = rtDW.dist_ini_data[0];
        nx = 0;
      }
    } else {
      if (!rtIsNaN(rtDW.dist_ini_data[0])) {
        nx = 0;
      } else {
        nx = -1;
        idx = 2;
        exitg1 = false;
        while ((!exitg1) && (idx <= g_idx)) {
          if (!rtIsNaN(rtDW.dist_ini_data[idx - 1])) {
            nx = idx - 1;
            exitg1 = true;
          } else {
            idx++;
          }
        }
      }

      if (nx + 1 == 0) {
        op_distance_n = rtDW.dist_ini_data[0];
        nx = 0;
      } else {
        op_distance_n = rtDW.dist_ini_data[nx];
        for (idx = nx + 1; idx < g_idx; idx++) {
          if (op_distance_n > rtDW.dist_ini_data[idx]) {
            op_distance_n = rtDW.dist_ini_data[idx];
            nx = idx;
          }
        }
      }
    }

    if (ii_size <= 2) {
      if (ii_size == 1) {
        C = rtDW.dist_end_data[0];
        idx = 0;
      } else if ((rtDW.dist_end_data[0] > rtDW.dist_end_data[1]) || (rtIsNaN
                  (rtDW.dist_end_data[0]) && (!rtIsNaN(rtDW.dist_end_data[1]))))
      {
        C = rtDW.dist_end_data[1];
        idx = 1;
      } else {
        C = rtDW.dist_end_data[0];
        idx = 0;
      }
    } else {
      if (!rtIsNaN(rtDW.dist_end_data[0])) {
        idx = 0;
      } else {
        idx = -1;
        g_idx = 2;
        exitg1 = false;
        while ((!exitg1) && (g_idx <= ii_size)) {
          if (!rtIsNaN(rtDW.dist_end_data[g_idx - 1])) {
            idx = g_idx - 1;
            exitg1 = true;
          } else {
            g_idx++;
          }
        }
      }

      if (idx + 1 == 0) {
        C = rtDW.dist_end_data[0];
        idx = 0;
      } else {
        C = rtDW.dist_end_data[idx];
        for (g_idx = idx + 1; g_idx < ii_size; g_idx++) {
          if (C > rtDW.dist_end_data[g_idx]) {
            C = rtDW.dist_end_data[g_idx];
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
      op_distance_n = rel_dist_xy_f(X_pos, oi_xy);
      oi_xy[0] = rtDW.xy_ini_data[nx];
      g_idx = nx + Map_data_size[0];
      oi_xy[1] = rtDW.xy_ini_data[g_idx];
      if (op_distance_n < rel_dist_xy_f(X_pos, oi_xy)) {
        oi_xy[0] = rtDW.xy_end_data[nx];
        oi_xy[1] = rtDW.xy_end_data[g_idx];
        if (op_distance_n < rel_dist_xy_f(X_pos, oi_xy)) {
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
    nx = rtDW.b_index_data[(int32_T)rtDW.ind_temp_data[0] - 1] - 1;
    *seg_id_near = Map_data_data[nx];
    *op_distance = dist_op_data[nx];
    oi_near[0] = oi_xy_data[nx];
    oi_near[1] = oi_xy_data[(rtDW.b_index_data[(int32_T)rtDW.ind_temp_data[0] -
      1] + oi_xy_size[0]) - 1];
    for (g_idx = 0; g_idx < loop_ub; g_idx++) {
      x_data[g_idx] = (Map_data_data[rtDW.b_index_data[(int32_T)
                       rtDW.ind_temp_data[0] - 1] - 1] ==
                       rtDW.SEG_GPS_HEAD_data[g_idx]);
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

    rel_dist_xy(X_pos, rtDW.xy_ini_data, xy_ini_size, rtDW.dist_ini_data, &g_idx);
    rel_dist_xy(X_pos, rtDW.xy_end_data, xy_end_size, rtDW.dist_end_data,
                &ii_size);
    if (g_idx <= 2) {
      if (g_idx == 1) {
        op_distance_n = rtDW.dist_ini_data[0];
        nx = 0;
      } else if ((rtDW.dist_ini_data[0] > rtDW.dist_ini_data[1]) || (rtIsNaN
                  (rtDW.dist_ini_data[0]) && (!rtIsNaN(rtDW.dist_ini_data[1]))))
      {
        op_distance_n = rtDW.dist_ini_data[1];
        nx = 1;
      } else {
        op_distance_n = rtDW.dist_ini_data[0];
        nx = 0;
      }
    } else {
      if (!rtIsNaN(rtDW.dist_ini_data[0])) {
        nx = 0;
      } else {
        nx = -1;
        idx = 2;
        exitg1 = false;
        while ((!exitg1) && (idx <= g_idx)) {
          if (!rtIsNaN(rtDW.dist_ini_data[idx - 1])) {
            nx = idx - 1;
            exitg1 = true;
          } else {
            idx++;
          }
        }
      }

      if (nx + 1 == 0) {
        op_distance_n = rtDW.dist_ini_data[0];
        nx = 0;
      } else {
        op_distance_n = rtDW.dist_ini_data[nx];
        for (idx = nx + 1; idx < g_idx; idx++) {
          if (op_distance_n > rtDW.dist_ini_data[idx]) {
            op_distance_n = rtDW.dist_ini_data[idx];
            nx = idx;
          }
        }
      }
    }

    if (ii_size <= 2) {
      if (ii_size == 1) {
        C = rtDW.dist_end_data[0];
        idx = 0;
      } else if ((rtDW.dist_end_data[0] > rtDW.dist_end_data[1]) || (rtIsNaN
                  (rtDW.dist_end_data[0]) && (!rtIsNaN(rtDW.dist_end_data[1]))))
      {
        C = rtDW.dist_end_data[1];
        idx = 1;
      } else {
        C = rtDW.dist_end_data[0];
        idx = 0;
      }
    } else {
      if (!rtIsNaN(rtDW.dist_end_data[0])) {
        idx = 0;
      } else {
        idx = -1;
        g_idx = 2;
        exitg1 = false;
        while ((!exitg1) && (g_idx <= ii_size)) {
          if (!rtIsNaN(rtDW.dist_end_data[g_idx - 1])) {
            idx = g_idx - 1;
            exitg1 = true;
          } else {
            g_idx++;
          }
        }
      }

      if (idx + 1 == 0) {
        C = rtDW.dist_end_data[0];
        idx = 0;
      } else {
        C = rtDW.dist_end_data[idx];
        for (g_idx = idx + 1; g_idx < ii_size; g_idx++) {
          if (C > rtDW.dist_end_data[g_idx]) {
            C = rtDW.dist_end_data[g_idx];
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

    if (op_distance_n_0 < dist_op_data[rtDW.b_index_data[(int32_T)
        rtDW.ind_temp_data[0] - 1] - 1]) {
      *note = 2.0;
      if (!(op_distance_n <= C)) {
        nx = idx;
      }

      for (g_idx = 0; g_idx < loop_ub; g_idx++) {
        x_data[g_idx] = (Map_data_data[rtDW.b_index_data[(int32_T)
                         rtDW.ind_temp_data[0] - 1] - 1] ==
                         rtDW.SEG_GPS_HEAD_data[g_idx]);
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
    rtDW.ind_temp_data[g_idx] = Map_data_data[Map_data_size[0] * 7 + g_idx] *
      3.1415926535897931 / 180.0;
  }

  loop_ub = Map_data_size[0];
  for (g_idx = 0; g_idx < loop_ub; g_idx++) {
    x_data[g_idx] = (rtDW.seg_id_data[g_idx] == *seg_id_near);
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
    if (rtDW.ind_temp_data[nx] > 4.71238898038469) {
      if (!(rtDW.ind_temp_data[nx] < 6.2831853071795862)) {
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
    if (rtDW.ind_temp_data[nx] > 3.1415926535897931) {
      if (!(rtDW.ind_temp_data[nx] < 4.71238898038469)) {
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

  *seg_heading = rtDW.ind_temp_data[nx];
}

// Function for MATLAB Function: '<S2>/Dijkstra'
void MMModelClass::power(const real_T a_data[], const int32_T a_size[2], real_T
  y_data[], int32_T y_size[2])
{
  real_T z1_data[3];
  int32_T loop_ub;
  y_size[1] = (int8_T)a_size[1];
  loop_ub = y_size[1] - 1;
  if (0 <= loop_ub) {
    memcpy(&z1_data[0], &y_data[0], (loop_ub + 1) * sizeof(real_T));
  }

  for (loop_ub = 0; loop_ub < y_size[1]; loop_ub++) {
    z1_data[loop_ub] = a_data[loop_ub] * a_data[loop_ub];
  }

  y_size[0] = 1;
  loop_ub = y_size[1] - 1;
  if (0 <= loop_ub) {
    memcpy(&y_data[0], &z1_data[0], (loop_ub + 1) * sizeof(real_T));
  }
}

// Function for MATLAB Function: '<S2>/Dijkstra'
real_T MMModelClass::sum(const real_T x_data[], const int32_T x_size[2])
{
  real_T y;
  int32_T k;
  if (x_size[1] == 0) {
    y = 0.0;
  } else {
    y = x_data[0];
    for (k = 2; k <= x_size[1]; k++) {
      y += x_data[k - 1];
    }
  }

  return y;
}

// Model step function
void MMModelClass::step()
{
  real_T a_j;
  real_T b_j;
  real_T note;
  real_T seg_direction;
  real_T head_err;
  real_T seg_heading;
  boolean_T ex;
  int8_T settled_data[1000];
  int32_T pidx_data[1000];
  int32_T zz_data[1000];
  int32_T nidx_data[1000];
  int32_T e;
  int32_T n_data[1000];
  boolean_T x_data[1000];
  int32_T ii_data[1000];
  int32_T nx;
  int32_T idx;
  boolean_T b_x_data[1000];
  int32_T f_ii_data[1000];
  int32_T e_nx;
  int32_T f_idx;
  int32_T g_idx;
  real_T rtb_Add;
  real_T rtb_Oi_near[2];
  real_T rtb_num_lane_direction[4];
  real_T tmp[2];
  real_T nodes_data[3];
  real_T tmp_data[3];
  int32_T oi_xy_size[2];
  int32_T nodes_size[2];
  int32_T tmp_size[2];
  int32_T path_size_idx_0;
  int32_T n_size_idx_0;
  int32_T f_ii_size_idx_1;
  int32_T nodes_size_tmp;
  boolean_T exitg1;
  boolean_T exitg2;

  // Sum: '<S1>/Add' incorporates:
  //   Abs: '<S1>/Abs'
  //   Abs: '<S1>/Abs1'
  //   Inport: '<Root>/finish_node_id_i'
  //   Inport: '<Root>/start_node_id_i'
  //   Memory: '<S1>/Memory'
  //   Memory: '<S1>/Memory1'
  //   Sum: '<S1>/Sum'
  //   Sum: '<S1>/Sum1'

  rtb_Add = std::abs(rtU.start_node_id_i - rtDW.Memory1_PreviousInput) + std::
    abs(rtU.finish_node_id_i - rtDW.Memory_PreviousInput);

  // Outputs for Enabled SubSystem: '<S1>/Enabled Subsystem' incorporates:
  //   EnablePort: '<S2>/Enable'

  if (rtb_Add > 0.0) {
    // MATLAB Function: '<S2>/Dijkstra' incorporates:
    //   Inport: '<Root>/finish_node_id_i'
    //   Inport: '<Root>/nodes_i'
    //   Inport: '<Root>/nodes_length1_i'
    //   Inport: '<Root>/nodes_length2_i'
    //   Inport: '<Root>/segments_i'
    //   Inport: '<Root>/segments_length1_i'
    //   Inport: '<Root>/segments_length2_i'
    //   Inport: '<Root>/start_node_id_i'

    if (1.0 > rtU.nodes_length1_i) {
      e_nx = 0;
    } else {
      e_nx = (int32_T)rtU.nodes_length1_i;
    }

    if (1.0 > rtU.nodes_length2_i) {
      f_idx = 0;
    } else {
      f_idx = (int32_T)rtU.nodes_length2_i;
    }

    for (idx = 0; idx < f_idx; idx++) {
      for (path_size_idx_0 = 0; path_size_idx_0 < e_nx; path_size_idx_0++) {
        rtDW.nodes_data[path_size_idx_0 + e_nx * idx] = rtU.nodes_i[1000 * idx +
          path_size_idx_0];
      }
    }

    if (1.0 > rtU.segments_length1_i) {
      e = 0;
    } else {
      e = (int32_T)rtU.segments_length1_i;
    }

    if (1.0 > rtU.segments_length2_i) {
      idx = 0;
    } else {
      idx = (int32_T)rtU.segments_length2_i;
    }

    nx = idx - 1;
    for (idx = 0; idx <= nx; idx++) {
      for (path_size_idx_0 = 0; path_size_idx_0 < e; path_size_idx_0++) {
        rtDW.segments_data[path_size_idx_0 + e * idx] = rtU.segments_i[1000 *
          idx + path_size_idx_0];
      }
    }

    if (0 <= e_nx - 1) {
      memcpy(&rtDW.node_ids_data[0], &rtDW.nodes_data[0], e_nx * sizeof(real_T));
    }

    g_idx = (e_nx << 1) - 1;
    if (0 <= g_idx) {
      memset(&rtDW.table_data[0], 0, (g_idx + 1) * sizeof(real_T));
    }

    for (idx = 0; idx < e_nx; idx++) {
      rtDW.shortest_distance_data[idx] = (rtInf);
    }

    if (0 <= e_nx - 1) {
      memset(&settled_data[0], 0, e_nx * sizeof(int8_T));
    }

    path_size_idx_0 = e_nx;
    g_idx = e_nx * e_nx - 1;
    if (0 <= g_idx) {
      memset(&rtDW.path_data[0], 0, (g_idx + 1) * sizeof(real_T));
    }

    for (idx = 0; idx < e_nx; idx++) {
      x_data[idx] = (rtU.start_node_id_i == rtDW.node_ids_data[idx]);
    }

    idx = 0;
    nx = e_nx;
    g_idx = 1;
    exitg1 = false;
    while ((!exitg1) && (g_idx <= e_nx)) {
      if (x_data[g_idx - 1]) {
        idx++;
        ii_data[idx - 1] = g_idx;
        if (idx >= e_nx) {
          exitg1 = true;
        } else {
          g_idx++;
        }
      } else {
        g_idx++;
      }
    }

    if (e_nx == 1) {
      if (idx == 0) {
        nx = 0;
      }
    } else if (1 > idx) {
      nx = 0;
    } else {
      nx = idx;
    }

    if (0 <= nx - 1) {
      memcpy(&pidx_data[0], &ii_data[0], nx * sizeof(int32_T));
    }

    rtDW.shortest_distance_data[ii_data[0] - 1] = 0.0;
    rtDW.table_data[(ii_data[0] + e_nx) - 1] = 0.0;
    settled_data[ii_data[0] - 1] = 1;
    rtDW.path_data[ii_data[0] - 1] = rtU.start_node_id_i;
    for (idx = 0; idx < e_nx; idx++) {
      x_data[idx] = (rtU.finish_node_id_i == rtDW.node_ids_data[idx]);
    }

    idx = 0;
    nx = e_nx;
    g_idx = 1;
    exitg1 = false;
    while ((!exitg1) && (g_idx <= e_nx)) {
      if (x_data[g_idx - 1]) {
        idx++;
        ii_data[idx - 1] = g_idx;
        if (idx >= e_nx) {
          exitg1 = true;
        } else {
          g_idx++;
        }
      } else {
        g_idx++;
      }
    }

    if (e_nx == 1) {
      if (idx == 0) {
        nx = 0;
      }
    } else if (1 > idx) {
      nx = 0;
    } else {
      nx = idx;
    }

    if (0 <= nx - 1) {
      memcpy(&zz_data[0], &ii_data[0], nx * sizeof(int32_T));
    }

    exitg1 = false;
    while ((!exitg1) && (settled_data[zz_data[0] - 1] == 0)) {
      nx = e_nx - 1;
      for (idx = 0; idx <= nx; idx++) {
        rtDW.table_data[idx] = rtDW.table_data[idx + e_nx];
      }

      rtDW.table_data[(pidx_data[0] + e_nx) - 1] = 0.0;
      for (idx = 0; idx < e; idx++) {
        x_data[idx] = (rtDW.nodes_data[pidx_data[0] - 1] ==
                       rtDW.segments_data[idx + e]);
      }

      nx = e - 1;
      idx = 0;
      for (g_idx = 0; g_idx <= nx; g_idx++) {
        if (x_data[g_idx]) {
          idx++;
        }
      }

      n_size_idx_0 = idx;
      idx = 0;
      for (g_idx = 0; g_idx <= nx; g_idx++) {
        if (x_data[g_idx]) {
          n_data[idx] = g_idx + 1;
          idx++;
        }
      }

      for (nx = 0; nx < n_size_idx_0; nx++) {
        for (idx = 0; idx < e_nx; idx++) {
          x_data[idx] = (rtDW.segments_data[((e << 1) + n_data[nx]) - 1] ==
                         rtDW.node_ids_data[idx]);
        }

        g_idx = -1;
        idx = 1;
        exitg2 = false;
        while ((!exitg2) && (idx <= e_nx)) {
          if (x_data[idx - 1]) {
            g_idx++;
            ii_data[g_idx] = idx;
            if (g_idx + 1 >= e_nx) {
              exitg2 = true;
            } else {
              idx++;
            }
          } else {
            idx++;
          }
        }

        if (!(settled_data[ii_data[0] - 1] != 0)) {
          if (2 > f_idx) {
            g_idx = 0;
            idx = 0;
            f_ii_size_idx_1 = 0;
          } else {
            g_idx = 1;
            idx = f_idx;
            f_ii_size_idx_1 = 1;
          }

          nodes_size[0] = 1;
          nodes_size_tmp = idx - g_idx;
          nodes_size[1] = nodes_size_tmp;
          for (idx = 0; idx < nodes_size_tmp; idx++) {
            nodes_data[idx] = rtDW.nodes_data[((g_idx + idx) * e_nx + pidx_data
              [0]) - 1] - rtDW.nodes_data[((f_ii_size_idx_1 + idx) * e_nx +
              ii_data[0]) - 1];
          }

          power(nodes_data, nodes_size, tmp_data, tmp_size);
          a_j = std::sqrt(sum(tmp_data, tmp_size));
          if ((rtDW.table_data[ii_data[0] - 1] == 0.0) ||
              (rtDW.table_data[ii_data[0] - 1] > rtDW.table_data[pidx_data[0] -
               1] + a_j)) {
            rtDW.table_data[(ii_data[0] + e_nx) - 1] =
              rtDW.table_data[pidx_data[0] - 1] + a_j;
            for (idx = 0; idx < e_nx; idx++) {
              b_x_data[idx] = (rtDW.path_data[(e_nx * idx + pidx_data[0]) - 1]
                               != 0.0);
            }

            g_idx = 0;
            f_ii_size_idx_1 = e_nx;
            idx = 1;
            exitg2 = false;
            while ((!exitg2) && (idx <= e_nx)) {
              if (b_x_data[idx - 1]) {
                g_idx++;
                f_ii_data[g_idx - 1] = idx;
                if (g_idx >= e_nx) {
                  exitg2 = true;
                } else {
                  idx++;
                }
              } else {
                idx++;
              }
            }

            if (e_nx == 1) {
              if (g_idx == 0) {
                f_ii_size_idx_1 = 0;
              }
            } else if (1 > g_idx) {
              f_ii_size_idx_1 = 0;
            } else {
              f_ii_size_idx_1 = g_idx;
            }

            g_idx = f_ii_size_idx_1 - 1;
            if (0 <= g_idx) {
              memset(&rtDW.tmp_path_data[0], 0, (g_idx + 1) * sizeof(real_T));
            }

            for (g_idx = 0; g_idx < f_ii_size_idx_1; g_idx++) {
              rtDW.tmp_path_data[g_idx] = rtDW.path_data[((f_ii_data[g_idx] - 1)
                * e_nx + pidx_data[0]) - 1];
            }

            g_idx = ii_data[0] - 1;
            for (idx = 0; idx < f_ii_size_idx_1; idx++) {
              rtDW.path_data[g_idx + e_nx * idx] = rtDW.tmp_path_data[idx];
            }

            rtDW.path_data[g_idx + e_nx * f_ii_size_idx_1] = rtDW.segments_data
              [((e << 1) + n_data[nx]) - 1];
          } else {
            rtDW.table_data[(ii_data[0] + e_nx) - 1] = rtDW.table_data[ii_data[0]
              - 1];
          }
        }
      }

      idx = 0;
      nx = e_nx;
      g_idx = 1;
      exitg2 = false;
      while ((!exitg2) && (g_idx <= e_nx)) {
        if (rtDW.table_data[(g_idx + e_nx) - 1] != 0.0) {
          idx++;
          ii_data[idx - 1] = g_idx;
          if (idx >= e_nx) {
            exitg2 = true;
          } else {
            g_idx++;
          }
        } else {
          g_idx++;
        }
      }

      if (e_nx == 1) {
        if (idx == 0) {
          nx = 0;
        }
      } else if (1 > idx) {
        nx = 0;
      } else {
        nx = idx;
      }

      if (0 <= nx - 1) {
        memcpy(&nidx_data[0], &ii_data[0], nx * sizeof(int32_T));
      }

      if (nx <= 2) {
        if (nx == 1) {
          a_j = rtDW.table_data[(ii_data[0] + e_nx) - 1];
        } else if (rtDW.table_data[(ii_data[0] + e_nx) - 1] > rtDW.table_data
                   [(ii_data[1] + e_nx) - 1]) {
          a_j = rtDW.table_data[(ii_data[1] + e_nx) - 1];
        } else if (rtIsNaN(rtDW.table_data[(ii_data[0] + e_nx) - 1])) {
          if (!rtIsNaN(rtDW.table_data[(ii_data[1] + e_nx) - 1])) {
            a_j = rtDW.table_data[(ii_data[1] + e_nx) - 1];
          } else {
            a_j = rtDW.table_data[(ii_data[0] + e_nx) - 1];
          }
        } else {
          a_j = rtDW.table_data[(ii_data[0] + e_nx) - 1];
        }
      } else {
        if (!rtIsNaN(rtDW.table_data[(ii_data[0] + e_nx) - 1])) {
          idx = 1;
        } else {
          idx = 0;
          g_idx = 2;
          exitg2 = false;
          while ((!exitg2) && (g_idx <= nx)) {
            if (!rtIsNaN(rtDW.table_data[(ii_data[g_idx - 1] + e_nx) - 1])) {
              idx = g_idx;
              exitg2 = true;
            } else {
              g_idx++;
            }
          }
        }

        if (idx == 0) {
          a_j = rtDW.table_data[(ii_data[0] + e_nx) - 1];
        } else {
          a_j = rtDW.table_data[(ii_data[idx - 1] + e_nx) - 1];
          while (idx + 1 <= nx) {
            if (a_j > rtDW.table_data[(ii_data[idx] + e_nx) - 1]) {
              a_j = rtDW.table_data[(ii_data[idx] + e_nx) - 1];
            }

            idx++;
          }
        }
      }

      for (idx = 0; idx < nx; idx++) {
        x_data[idx] = (rtDW.table_data[(ii_data[idx] + e_nx) - 1] == a_j);
      }

      idx = 0;
      g_idx = 1;
      exitg2 = false;
      while ((!exitg2) && (g_idx <= nx)) {
        if (x_data[g_idx - 1]) {
          idx++;
          ii_data[idx - 1] = g_idx;
          if (idx >= nx) {
            exitg2 = true;
          } else {
            g_idx++;
          }
        } else {
          g_idx++;
        }
      }

      if (nx == 1) {
        if (idx == 0) {
          nx = 0;
        }
      } else if (1 > idx) {
        nx = 0;
      } else {
        nx = idx;
      }

      if (nx == 0) {
        exitg1 = true;
      } else {
        pidx_data[0] = nidx_data[ii_data[0] - 1];
        idx = nidx_data[ii_data[0] - 1] - 1;
        rtDW.shortest_distance_data[idx] = rtDW.table_data[(nidx_data[ii_data[0]
          - 1] + e_nx) - 1];
        settled_data[idx] = 1;
      }
    }

    for (idx = 0; idx < e_nx; idx++) {
      b_x_data[idx] = (rtDW.path_data[(e_nx * idx + zz_data[0]) - 1] != 0.0);
    }

    f_idx = 0;
    f_ii_size_idx_1 = e_nx;
    e = 0;
    exitg1 = false;
    while ((!exitg1) && (e + 1 <= e_nx)) {
      if (b_x_data[e]) {
        f_idx++;
        if (f_idx >= e_nx) {
          exitg1 = true;
        } else {
          e++;
        }
      } else {
        e++;
      }
    }

    if (e_nx == 1) {
      if (f_idx == 0) {
        f_ii_size_idx_1 = 0;
      }
    } else if (1 > f_idx) {
      f_ii_size_idx_1 = 0;
    } else {
      f_ii_size_idx_1 = f_idx;
    }

    e_nx = f_ii_size_idx_1;
    if (1 > f_ii_size_idx_1) {
      e_nx = 0;
    }

    rtDW.dist = rtDW.shortest_distance_data[zz_data[0] - 1];
    rtDW.SFunction_DIMS3_l[0] = e_nx;
    rtDW.SFunction_DIMS3_l[1] = 1;
    for (idx = 0; idx < e_nx; idx++) {
      rtDW.shortest_distance_data[idx] = rtDW.path_data[(path_size_idx_0 * idx +
        zz_data[0]) - 1];
    }

    if (0 <= e_nx - 1) {
      memcpy(&rtDW.path_2[0], &rtDW.shortest_distance_data[0], e_nx * sizeof
             (real_T));
    }

    // End of MATLAB Function: '<S2>/Dijkstra'
  }

  // End of Outputs for SubSystem: '<S1>/Enabled Subsystem'

  // MATLAB Function: '<S1>/Final_Static_Path' incorporates:
  //   Inport: '<Root>/Map_data_i'
  //   Inport: '<Root>/Map_data_length1_i'
  //   Inport: '<Root>/Map_data_length2_i'

  if (1.0 > rtU.Map_data_length1_i) {
    e_nx = 0;
  } else {
    e_nx = (int32_T)rtU.Map_data_length1_i;
  }

  if (1.0 > rtU.Map_data_length2_i) {
    f_idx = 0;
  } else {
    f_idx = (int32_T)rtU.Map_data_length2_i;
  }

  if (!rtDW.path_out1_not_empty) {
    if (rtb_Add > 0.0) {
      rtDW.path_out1.size = rtDW.SFunction_DIMS3_l[0];
      g_idx = rtDW.SFunction_DIMS3_l[0];
      if (0 <= g_idx - 1) {
        memcpy(&rtDW.path_out1.data[0], &rtDW.path_2[0], g_idx * sizeof(real_T));
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
    rtDW.path_out1.size = rtDW.SFunction_DIMS3_l[0];
    g_idx = rtDW.SFunction_DIMS3_l[0];
    if (0 <= g_idx - 1) {
      memcpy(&rtDW.path_out1.data[0], &rtDW.path_2[0], g_idx * sizeof(real_T));
    }

    rtDW.path_out1_not_empty = !(rtDW.path_out1.size == 0);
  }

  rtDW.SFunction_DIMS2[0] = rtDW.path_out1.size;
  rtDW.SFunction_DIMS2[1] = 1;
  rtDW.SFunction_DIMS3[0] = rtDW.path_out1.size;
  rtDW.SFunction_DIMS3[1] = 1;
  rtDW.SFunction_DIMS4[0] = e_nx;
  rtDW.SFunction_DIMS4[1] = f_idx;
  for (idx = 0; idx < f_idx; idx++) {
    for (path_size_idx_0 = 0; path_size_idx_0 < e_nx; path_size_idx_0++) {
      rtDW.Static_Path_0[path_size_idx_0 + rtDW.SFunction_DIMS4[0] * idx] =
        rtU.Map_data_i[1000 * idx + path_size_idx_0];
    }
  }

  rtDW.SFunction_DIMS8[0] = rtDW.path_out1.size;
  rtDW.SFunction_DIMS8[1] = 1;

  // Outport: '<Root>/Static_Path_0'
  g_idx = rtDW.SFunction_DIMS4[0] * rtDW.SFunction_DIMS4[1];
  if (0 <= g_idx - 1) {
    memcpy(&rtY.Static_Path_0[0], &rtDW.Static_Path_0[0], g_idx * sizeof(real_T));
  }

  // End of Outport: '<Root>/Static_Path_0'

  // MATLAB Function: '<S1>/MM' incorporates:
  //   Gain: '<S1>/Gain'
  //   Gain: '<S1>/Gain3'
  //   Inport: '<Root>/X_UKF_SLAM_i'
  //   MATLAB Function: '<S1>/MATLAB Function'

  e = rtDW.SFunction_DIMS4[0];
  oi_xy_size[0] = rtDW.SFunction_DIMS4[0];
  oi_xy_size[1] = 2;
  g_idx = (rtDW.SFunction_DIMS4[0] << 1) - 1;
  if (0 <= g_idx) {
    memset(&rtDW.oi_xy_data[0], 0, (g_idx + 1) * sizeof(real_T));
  }

  path_size_idx_0 = rtDW.SFunction_DIMS4[0];
  if (0 <= e - 1) {
    memset(&rtDW.node_ids_data[0], 0, e * sizeof(real_T));
  }

  for (idx = 0; idx < rtDW.SFunction_DIMS4[0]; idx++) {
    if (rtDW.Static_Path_0[rtDW.SFunction_DIMS4[0] * 5 + idx] == (rtInf)) {
      rtDW.oi_xy_data[idx] = rtDW.Static_Path_0[idx + rtDW.SFunction_DIMS4[0]];
      rtDW.oi_xy_data[idx + e] = rtU.X_UKF_SLAM_i[1];
    } else if (rtDW.Static_Path_0[rtDW.SFunction_DIMS4[0] * 5 + idx] == 0.0) {
      rtDW.oi_xy_data[idx] = rtU.X_UKF_SLAM_i[0];
      rtDW.oi_xy_data[idx + e] = rtDW.Static_Path_0[(rtDW.SFunction_DIMS4[0] <<
        1) + idx];
    } else {
      a_j = -1.0 / rtDW.Static_Path_0[rtDW.SFunction_DIMS4[0] * 5 + idx];
      b_j = rtU.X_UKF_SLAM_i[1] - a_j * rtU.X_UKF_SLAM_i[0];
      note = rtDW.Static_Path_0[rtDW.SFunction_DIMS4[0] * 5 + idx] - a_j;
      rtDW.oi_xy_data[idx] = (b_j - rtDW.Static_Path_0[rtDW.SFunction_DIMS4[0] *
        6 + idx]) / note;
      rtDW.oi_xy_data[idx + e] = (rtDW.Static_Path_0[rtDW.SFunction_DIMS4[0] * 5
        + idx] * b_j - rtDW.Static_Path_0[rtDW.SFunction_DIMS4[0] * 6 + idx] *
        a_j) / note;
    }
  }

  for (g_idx = 0; g_idx < oi_xy_size[0]; g_idx++) {
    a_j = rtDW.oi_xy_data[g_idx] - rtU.X_UKF_SLAM_i[0];
    b_j = rtDW.oi_xy_data[g_idx + e] - rtU.X_UKF_SLAM_i[1];
    rtDW.node_ids_data[g_idx] = std::sqrt(a_j * a_j + b_j * b_j);
  }

  tmp[0] = rtU.X_UKF_SLAM_i[0];
  tmp[1] = rtU.X_UKF_SLAM_i[1];
  MM_g(0.017453292519943295 * (57.295779513082323 * rtU.X_UKF_SLAM_i[2]) * 180.0
       / 3.1415926535897931, tmp, rtDW.oi_xy_data, oi_xy_size,
       rtDW.node_ids_data, &path_size_idx_0, rtDW.Static_Path_0,
       rtDW.SFunction_DIMS4, &a_j, &b_j, rtb_Oi_near, &note, &seg_direction,
       &head_err, rtb_num_lane_direction, &seg_heading);

  // Outport: '<Root>/Oi_near' incorporates:
  //   MATLAB Function: '<S1>/MM'

  rtY.Oi_near[0] = rtb_Oi_near[0];
  rtY.Oi_near[1] = rtb_Oi_near[1];

  // MATLAB Function: '<S1>/MM'
  g_idx = rtDW.SFunction_DIMS4[0];
  for (idx = 0; idx < g_idx; idx++) {
    x_data[idx] = (rtDW.Static_Path_0[idx] == a_j);
  }

  idx = 0;
  ex = x_data[0];
  for (nx = 1; nx < rtDW.SFunction_DIMS4[0]; nx++) {
    if ((int32_T)ex < (int32_T)x_data[nx]) {
      ex = x_data[nx];
      idx = nx;
    }
  }

  // Outport: '<Root>/seg_id_near' incorporates:
  //   MATLAB Function: '<S1>/MM'

  rtY.seg_id_near = a_j;

  // Outport: '<Root>/seg_Curvature' incorporates:
  //   MATLAB Function: '<S1>/MM'

  rtY.seg_Curvature = rtDW.Static_Path_0[rtDW.SFunction_DIMS4[0] * 13 + idx];

  // Outport: '<Root>/Static_Path_0_length1' incorporates:
  //   MATLAB Function: '<S1>/Final_Static_Path'

  rtY.Static_Path_0_length1 = f_idx;

  // Outport: '<Root>/Static_Path_0_length2' incorporates:
  //   MATLAB Function: '<S1>/Final_Static_Path'

  rtY.Static_Path_0_length2 = e_nx;

  // Update for Memory: '<S1>/Memory1' incorporates:
  //   Inport: '<Root>/start_node_id_i'

  rtDW.Memory1_PreviousInput = rtU.start_node_id_i;

  // Update for Memory: '<S1>/Memory' incorporates:
  //   Inport: '<Root>/finish_node_id_i'

  rtDW.Memory_PreviousInput = rtU.finish_node_id_i;

  // MATLAB Function: '<S1>/Final_Static_Path'
  if (rtb_Add > 0.0) {
    // Update for UnitDelay: '<S1>/Unit Delay'
    rtDW.UnitDelay_DSTATE = rtDW.dist;
  }
}

// Model initialize function
void MMModelClass::initialize()
{
  // Registration code

  // initialize non-finites
  rt_InitInfAndNaN(sizeof(real_T));
}

// Constructor
MMModelClass::MMModelClass()
{
}

// Destructor
MMModelClass::~MMModelClass()
{
  // Currently there is no destructor body generated.
}

// Real-Time Model get method
RT_MODEL * MMModelClass::getRTM()
{
  return (&rtM);
}

//
// File trailer for generated code.
//
// [EOF]
//
