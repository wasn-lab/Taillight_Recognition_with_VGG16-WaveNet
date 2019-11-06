//
// File: Localization_UKF.cpp
//
// Code generated for Simulink model 'Localization_UKF'.
//
// Model version                  : 1.1
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Fri Oct 25 18:26:59 2019
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
// Code generation objectives:
//    1. Execution efficiency
//    2. RAM efficiency
// Validation result: Not run
//
#include "Localization_UKF.h"

// Function for MATLAB Function: '<S1>/SLAM_UKF'
real_T Localization_UKFModelClass::sum(const real_T x[10])
{
  real_T y;
  int32_T k;
  y = x[0];
  for (k = 0; k < 9; k++) {
    y += x[k + 1];
  }

  return y;
}

// Function for MATLAB Function: '<S1>/SLAM_UKF'
void Localization_UKFModelClass::invNxN(const real_T x[25], real_T y[25])
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

// Model step function
void Localization_UKFModelClass::step()
{
  real_T p_sqrt_data[25];
  real_T temp_dia[5];
  int32_T jj;
  real_T ajj;
  int32_T colj;
  int32_T ix;
  int32_T iy;
  int8_T ii_data[5];
  int32_T b_colj;
  int32_T c_ix;
  int32_T d_ix;
  int32_T m;
  int32_T b_ia;
  real_T SLAM_Y_out;
  real_T SLAM_Heading_out;
  static const real_T b[5] = { 0.00025, 0.00025, 1.0E-7, 1.0E-5, 0.0001 };

  real_T x_aug[55];
  real_T K[25];
  real_T K1[4];
  real_T x[11];
  real_T b_x[11];
  int8_T I[25];
  real_T rtb_Gain2;
  real_T rtb_Gain1;
  real_T rtb_X[5];
  real_T rtb_UnitDelay34[5];
  real_T rtb_X_AUG[55];
  real_T rtb_R_last_o[25];
  real_T rtb_P[25];
  int32_T i;
  real_T tmp[25];
  real_T x_aug_0[10];
  real_T x_aug_1[5];
  real_T x_aug_2[25];
  real_T rtb_UnitDelay34_0[2];
  real_T c_x[4];
  real_T c_x_idx_1;
  real_T c_x_idx_0;
  real_T c_x_idx_3;
  real_T c_x_idx_2;
  real_T y_idx_1;
  boolean_T exitg1;

  // UnitDelay: '<S1>/Unit Delay34'
  for (i = 0; i < 5; i++) {
    rtb_UnitDelay34[i] = rtDW.UnitDelay34_DSTATE[i];
  }

  // End of UnitDelay: '<S1>/Unit Delay34'

  // MATLAB Function: '<S1>/SLAM_Generate_sigma_pt_UKF' incorporates:
  //   MATLAB Function: '<S1>/UKF_para'
  //   UnitDelay: '<S1>/Unit Delay33'

  memcpy(&p_sqrt_data[0], &rtDW.UnitDelay33_DSTATE[0], 25U * sizeof(real_T));
  i = 0;
  colj = 0;
  b_colj = 1;
  exitg1 = false;
  while ((!exitg1) && (b_colj <= 5)) {
    jj = (colj + b_colj) - 1;
    ajj = 0.0;
    if (!(b_colj - 1 < 1)) {
      ix = colj;
      iy = colj;
      for (c_ix = 1; c_ix < b_colj; c_ix++) {
        ajj += p_sqrt_data[ix] * p_sqrt_data[iy];
        ix++;
        iy++;
      }
    }

    ajj = p_sqrt_data[jj] - ajj;
    if (ajj > 0.0) {
      ajj = std::sqrt(ajj);
      p_sqrt_data[jj] = ajj;
      if (b_colj < 5) {
        if (b_colj - 1 != 0) {
          c_ix = jj + 5;
          ix = ((4 - b_colj) * 5 + colj) + 6;
          for (iy = colj + 6; iy <= ix; iy += 5) {
            d_ix = colj;
            SLAM_Y_out = 0.0;
            m = (iy + b_colj) - 2;
            for (b_ia = iy; b_ia <= m; b_ia++) {
              SLAM_Y_out += p_sqrt_data[b_ia - 1] * p_sqrt_data[d_ix];
              d_ix++;
            }

            p_sqrt_data[c_ix] += -SLAM_Y_out;
            c_ix += 5;
          }
        }

        ajj = 1.0 / ajj;
        c_ix = ((4 - b_colj) * 5 + jj) + 6;
        for (jj += 5; jj + 1 <= c_ix; jj += 5) {
          p_sqrt_data[jj] *= ajj;
        }

        colj += 5;
      }

      b_colj++;
    } else {
      p_sqrt_data[jj] = ajj;
      i = b_colj;
      exitg1 = true;
    }
  }

  if (i == 0) {
    colj = 5;
  } else {
    colj = i - 1;
  }

  for (b_colj = 1; b_colj <= colj; b_colj++) {
    for (jj = b_colj; jj < colj; jj++) {
      p_sqrt_data[jj + 5 * (b_colj - 1)] = 0.0;
    }
  }

  if (1 > colj) {
    colj = 0;
    ix = 0;
  } else {
    ix = colj;
  }

  for (c_ix = 0; c_ix < ix; c_ix++) {
    for (jj = 0; jj < colj; jj++) {
      rtb_R_last_o[jj + colj * c_ix] = p_sqrt_data[5 * c_ix + jj];
    }
  }

  for (c_ix = 0; c_ix < ix; c_ix++) {
    for (jj = 0; jj < colj; jj++) {
      b_colj = colj * c_ix;
      p_sqrt_data[jj + b_colj] = rtb_R_last_o[b_colj + jj];
    }
  }

  memset(&rtb_X_AUG[0], 0, 55U * sizeof(real_T));
  if (i != 0) {
    for (b_colj = 0; b_colj < 5; b_colj++) {
      temp_dia[b_colj] = std::abs(rtDW.UnitDelay33_DSTATE[5 * b_colj + b_colj]);
    }

    i = 0;
    b_colj = 1;
    exitg1 = false;
    while ((!exitg1) && (b_colj < 6)) {
      if (temp_dia[b_colj - 1] < 1.0E-10) {
        i++;
        ii_data[i - 1] = (int8_T)b_colj;
        if (i >= 5) {
          exitg1 = true;
        } else {
          b_colj++;
        }
      } else {
        b_colj++;
      }
    }

    if (!(1 > i)) {
      for (c_ix = 0; c_ix < i; c_ix++) {
        temp_dia[ii_data[c_ix] - 1] = 1.0E-10;
      }
    }

    memset(&p_sqrt_data[0], 0, 25U * sizeof(real_T));
    for (i = 0; i < 5; i++) {
      p_sqrt_data[i + 5 * i] = temp_dia[i];
    }

    i = 0;
    b_colj = 0;
    colj = 1;
    exitg1 = false;
    while ((!exitg1) && (colj < 6)) {
      jj = (b_colj + colj) - 1;
      ajj = 0.0;
      if (!(colj - 1 < 1)) {
        c_ix = b_colj;
        ix = b_colj;
        for (iy = 1; iy < colj; iy++) {
          ajj += p_sqrt_data[c_ix] * p_sqrt_data[ix];
          c_ix++;
          ix++;
        }
      }

      ajj = p_sqrt_data[jj] - ajj;
      if (ajj > 0.0) {
        ajj = std::sqrt(ajj);
        p_sqrt_data[jj] = ajj;
        if (colj < 5) {
          if (colj - 1 != 0) {
            c_ix = jj + 5;
            ix = ((4 - colj) * 5 + b_colj) + 6;
            for (iy = b_colj + 6; iy <= ix; iy += 5) {
              d_ix = b_colj;
              SLAM_Y_out = 0.0;
              m = (iy + colj) - 2;
              for (b_ia = iy; b_ia <= m; b_ia++) {
                SLAM_Y_out += p_sqrt_data[b_ia - 1] * p_sqrt_data[d_ix];
                d_ix++;
              }

              p_sqrt_data[c_ix] += -SLAM_Y_out;
              c_ix += 5;
            }
          }

          ajj = 1.0 / ajj;
          c_ix = ((4 - colj) * 5 + jj) + 6;
          for (jj += 5; jj + 1 <= c_ix; jj += 5) {
            p_sqrt_data[jj] *= ajj;
          }

          b_colj += 5;
        }

        colj++;
      } else {
        p_sqrt_data[jj] = ajj;
        i = colj;
        exitg1 = true;
      }
    }

    if (i == 0) {
      i = 5;
    } else {
      i--;
    }

    for (b_colj = 0; b_colj < i; b_colj++) {
      for (colj = b_colj + 1; colj < i; colj++) {
        p_sqrt_data[colj + 5 * b_colj] = 0.0;
      }
    }

    colj = 5;
    ix = 5;
  }

  for (c_ix = 0; c_ix < colj; c_ix++) {
    for (jj = 0; jj < ix; jj++) {
      rtb_R_last_o[jj + ix * c_ix] = p_sqrt_data[colj * jj + c_ix] *
        2.23606797749979;
    }
  }

  for (c_ix = 0; c_ix < colj; c_ix++) {
    for (jj = 0; jj < ix; jj++) {
      b_colj = ix * c_ix;
      p_sqrt_data[jj + b_colj] = rtb_R_last_o[b_colj + jj];
    }
  }

  for (c_ix = 0; c_ix < 5; c_ix++) {
    rtb_X_AUG[c_ix] = rtb_UnitDelay34[c_ix];
  }

  colj = ix - 1;
  for (b_colj = 0; b_colj < 5; b_colj++) {
    for (c_ix = 0; c_ix <= colj; c_ix++) {
      temp_dia[c_ix] = p_sqrt_data[ix * b_colj + c_ix];
    }

    c_ix = b_colj + 2;
    for (jj = 0; jj < 5; jj++) {
      rtb_X_AUG[jj + 5 * (c_ix - 1)] = rtb_UnitDelay34[jj] + temp_dia[jj];
    }
  }

  colj = ix - 1;
  for (i = 0; i < 5; i++) {
    for (c_ix = 0; c_ix <= colj; c_ix++) {
      temp_dia[c_ix] = p_sqrt_data[ix * i + c_ix];
    }

    c_ix = i + 7;
    for (jj = 0; jj < 5; jj++) {
      rtb_X_AUG[jj + 5 * (c_ix - 1)] = rtb_UnitDelay34[jj] - temp_dia[jj];
    }
  }

  // End of MATLAB Function: '<S1>/SLAM_Generate_sigma_pt_UKF'

  // Gain: '<S1>/Gain1' incorporates:
  //   Gain: '<S1>/Gain'
  //   Inport: '<Root>/angular_vz'

  rtb_Gain1 = -(0.017453292519943295 * rtU.angular_vz);

  // MATLAB Function: '<S1>/SLAM_Check' incorporates:
  //   Gain: '<S1>/Gain2'
  //   Inport: '<Root>/SLAM_counter'
  //   Inport: '<Root>/SLAM_fault'
  //   Inport: '<Root>/SLAM_heading'
  //   Inport: '<Root>/SLAM_x'
  //   Inport: '<Root>/SLAM_y'
  //   Inport: '<Root>/Speed_mps'
  //   SignalConversion: '<S2>/TmpSignal ConversionAt SFunction Inport6'
  //   UnitDelay: '<S1>/Unit Delay1'
  //   UnitDelay: '<S1>/Unit Delay35'
  //   UnitDelay: '<S1>/Unit Delay36'
  //   UnitDelay: '<S1>/Unit Delay37'

  i = 0;
  if (rtU.SLAM_counter != rtDW.UnitDelay35_DSTATE[3]) {
    i = 1;

    // Update for UnitDelay: '<S1>/Unit Delay38'
    rtDW.UnitDelay38_DSTATE = 0.0;
  } else {
    // Update for UnitDelay: '<S1>/Unit Delay38'
    rtDW.UnitDelay38_DSTATE++;
  }

  if (rtU.SLAM_fault == 1.0) {
    i = 0;
  }

  if (i == 0) {
    ajj = rtDW.UnitDelay35_DSTATE[0];
    SLAM_Y_out = rtDW.UnitDelay35_DSTATE[1];
    SLAM_Heading_out = rtDW.UnitDelay35_DSTATE[2];
    memcpy(&p_sqrt_data[0], &rtDW.UnitDelay37_DSTATE[0], 25U * sizeof(real_T));
    memcpy(&rtb_R_last_o[0], &rtDW.UnitDelay36_DSTATE[0], 25U * sizeof(real_T));
  } else {
    ajj = rtU.SLAM_x;
    SLAM_Y_out = rtU.SLAM_y;
    SLAM_Heading_out = rtU.SLAM_heading;
    memset(&p_sqrt_data[0], 0, 25U * sizeof(real_T));
    for (b_colj = 0; b_colj < 5; b_colj++) {
      p_sqrt_data[b_colj + 5 * b_colj] = b[b_colj];
    }

    rtb_UnitDelay34[0] = 0.0001;
    rtb_UnitDelay34[1] = 0.0001;
    if (std::abs(rtU.SLAM_heading - rtDW.UnitDelay1_DSTATE[2]) > 5.5) {
      rtb_UnitDelay34[2] = 100.0;
    } else {
      rtb_UnitDelay34[2] = 0.1;
    }

    rtb_UnitDelay34[3] = 0.1;
    rtb_UnitDelay34[4] = 0.0001;
    memset(&rtb_R_last_o[0], 0, 25U * sizeof(real_T));
    for (b_colj = 0; b_colj < 5; b_colj++) {
      rtb_R_last_o[b_colj + 5 * b_colj] = rtb_UnitDelay34[b_colj];
    }
  }

  rtb_UnitDelay34[0] = ajj;
  rtb_UnitDelay34[1] = SLAM_Y_out;
  rtb_UnitDelay34[2] = SLAM_Heading_out;
  rtb_UnitDelay34[3] = rtb_Gain1;
  rtb_UnitDelay34[4] = 1.025 * rtU.Speed_mps;

  // MATLAB Function: '<S1>/SLAM_UKF' incorporates:
  //   Constant: '<Root>/[Para] D_GC'
  //   Inport: '<Root>/dt'
  //   MATLAB Function: '<S1>/SLAM_Check'
  //   SignalConversion: '<S4>/TmpSignal ConversionAt SFunction Inport5'

  memcpy(&x_aug[0], &rtb_X_AUG[0], 55U * sizeof(real_T));
  rtb_Gain1 = rtU.dt * rtb_Gain1 * 3.8;
  for (c_ix = 0; c_ix < 11; c_ix++) {
    b_colj = 5 * c_ix + 2;
    x_aug[5 * c_ix] = (rtb_X_AUG[5 * c_ix + 4] * rtU.dt * std::cos
                       (rtb_X_AUG[b_colj]) + rtb_X_AUG[5 * c_ix]) + std::cos
      (rtb_X_AUG[5 * c_ix + 2] + 1.5707963267948966) * rtb_Gain1;
    x[c_ix] = std::sin(x_aug[b_colj]);
    b_x[c_ix] = std::sin(x_aug[5 * c_ix + 2] + 1.5707963267948966);
  }

  for (c_ix = 0; c_ix < 11; c_ix++) {
    x_aug[1 + 5 * c_ix] = (x_aug[5 * c_ix + 4] * rtU.dt * x[c_ix] + x_aug[5 *
      c_ix + 1]) + rtb_Gain1 * b_x[c_ix];
    x_aug[2 + 5 * c_ix] += x_aug[5 * c_ix + 3] * rtU.dt;
  }

  for (c_ix = 0; c_ix < 10; c_ix++) {
    x_aug_0[c_ix] = x_aug[(1 + c_ix) * 5];
  }

  rtb_X[0] = x_aug[0] * 0.0 + sum(x_aug_0) * 0.1;
  for (c_ix = 0; c_ix < 10; c_ix++) {
    x_aug_0[c_ix] = x_aug[(1 + c_ix) * 5 + 1];
  }

  rtb_X[1] = x_aug[1] * 0.0 + sum(x_aug_0) * 0.1;
  for (c_ix = 0; c_ix < 10; c_ix++) {
    x_aug_0[c_ix] = x_aug[(1 + c_ix) * 5 + 2];
  }

  rtb_X[2] = x_aug[2] * 0.0 + sum(x_aug_0) * 0.1;
  for (c_ix = 0; c_ix < 10; c_ix++) {
    x_aug_0[c_ix] = x_aug[(1 + c_ix) * 5 + 3];
  }

  rtb_X[3] = x_aug[3] * 0.0 + sum(x_aug_0) * 0.1;
  for (c_ix = 0; c_ix < 10; c_ix++) {
    x_aug_0[c_ix] = x_aug[(1 + c_ix) * 5 + 4];
  }

  rtb_X[4] = x_aug[4] * 0.0 + sum(x_aug_0) * 0.1;
  for (c_ix = 0; c_ix < 5; c_ix++) {
    rtb_Gain1 = x_aug[c_ix] - rtb_X[c_ix];
    temp_dia[c_ix] = rtb_Gain1;
    x_aug_1[c_ix] = rtb_Gain1;
  }

  for (c_ix = 0; c_ix < 5; c_ix++) {
    for (jj = 0; jj < 5; jj++) {
      x_aug_2[c_ix + 5 * jj] = temp_dia[c_ix] * x_aug_1[jj];
    }
  }

  for (c_ix = 0; c_ix < 5; c_ix++) {
    for (jj = 0; jj < 5; jj++) {
      rtb_P[jj + 5 * c_ix] = x_aug_2[5 * c_ix + jj] * 2.0;
    }
  }

  for (b_colj = 0; b_colj < 10; b_colj++) {
    for (c_ix = 0; c_ix < 5; c_ix++) {
      rtb_Gain1 = x_aug[(b_colj + 1) * 5 + c_ix] - rtb_X[c_ix];
      temp_dia[c_ix] = rtb_Gain1;
      x_aug_1[c_ix] = rtb_Gain1;
    }

    for (c_ix = 0; c_ix < 5; c_ix++) {
      for (jj = 0; jj < 5; jj++) {
        x_aug_2[c_ix + 5 * jj] = temp_dia[c_ix] * x_aug_1[jj];
      }
    }

    for (c_ix = 0; c_ix < 5; c_ix++) {
      for (jj = 0; jj < 5; jj++) {
        colj = 5 * c_ix + jj;
        rtb_P[jj + 5 * c_ix] = x_aug_2[colj] * 0.1 + rtb_P[colj];
      }
    }
  }

  for (c_ix = 0; c_ix < 25; c_ix++) {
    rtb_P[c_ix] += p_sqrt_data[c_ix];
  }

  if (rtb_X[2] < 0.0) {
    rtb_X[2] += 6.2831853071795862;
  } else {
    if (rtb_X[2] >= 6.2831853071795862) {
      rtb_X[2] -= 6.2831853071795862;
    }
  }

  if (i > 0) {
    for (c_ix = 0; c_ix < 25; c_ix++) {
      x_aug_2[c_ix] = rtb_P[c_ix] + rtb_R_last_o[c_ix];
    }

    invNxN(x_aug_2, tmp);
    for (c_ix = 0; c_ix < 5; c_ix++) {
      for (jj = 0; jj < 5; jj++) {
        b_colj = c_ix + 5 * jj;
        K[b_colj] = 0.0;
        for (i = 0; i < 5; i++) {
          K[b_colj] = rtb_P[5 * i + c_ix] * tmp[5 * jj + i] + K[5 * jj + c_ix];
        }
      }

      temp_dia[c_ix] = rtb_UnitDelay34[c_ix] - rtb_X[c_ix];
    }

    for (c_ix = 0; c_ix < 5; c_ix++) {
      rtb_Gain1 = 0.0;
      for (jj = 0; jj < 5; jj++) {
        rtb_Gain1 += K[5 * jj + c_ix] * temp_dia[jj];
      }

      rtb_X[c_ix] += rtb_Gain1;
    }

    for (c_ix = 0; c_ix < 25; c_ix++) {
      I[c_ix] = 0;
    }

    for (i = 0; i < 5; i++) {
      I[i + 5 * i] = 1;
    }

    for (c_ix = 0; c_ix < 5; c_ix++) {
      for (jj = 0; jj < 5; jj++) {
        b_colj = 5 * c_ix + jj;
        x_aug_2[jj + 5 * c_ix] = (real_T)I[b_colj] - K[b_colj];
      }
    }

    for (c_ix = 0; c_ix < 5; c_ix++) {
      for (jj = 0; jj < 5; jj++) {
        b_colj = jj + 5 * c_ix;
        K[b_colj] = 0.0;
        for (i = 0; i < 5; i++) {
          K[b_colj] = x_aug_2[5 * i + jj] * rtb_P[5 * c_ix + i] + K[5 * c_ix +
            jj];
        }
      }
    }

    for (c_ix = 0; c_ix < 5; c_ix++) {
      for (jj = 0; jj < 5; jj++) {
        rtb_P[jj + 5 * c_ix] = K[5 * c_ix + jj];
      }
    }
  } else {
    c_x_idx_0 = rtb_P[18] + rtb_R_last_o[18];
    c_x_idx_1 = rtb_P[19] + rtb_R_last_o[19];
    c_x_idx_2 = rtb_P[23] + rtb_R_last_o[23];
    c_x_idx_3 = rtb_P[24] + rtb_R_last_o[24];
    if (std::abs(c_x_idx_1) > std::abs(c_x_idx_0)) {
      rtb_Gain2 = c_x_idx_0 / c_x_idx_1;
      rtb_Gain1 = 1.0 / (rtb_Gain2 * c_x_idx_3 - c_x_idx_2);
      c_x_idx_3 = c_x_idx_3 / c_x_idx_1 * rtb_Gain1;
      y_idx_1 = -rtb_Gain1;
      c_x_idx_0 = -c_x_idx_2 / c_x_idx_1 * rtb_Gain1;
      rtb_Gain1 *= rtb_Gain2;
    } else {
      rtb_Gain2 = c_x_idx_1 / c_x_idx_0;
      rtb_Gain1 = 1.0 / (c_x_idx_3 - rtb_Gain2 * c_x_idx_2);
      c_x_idx_3 = c_x_idx_3 / c_x_idx_0 * rtb_Gain1;
      y_idx_1 = -rtb_Gain2 * rtb_Gain1;
      c_x_idx_0 = -c_x_idx_2 / c_x_idx_0 * rtb_Gain1;
    }

    for (c_ix = 0; c_ix < 2; c_ix++) {
      K1[c_ix] = 0.0;
      K1[c_ix] += rtb_P[c_ix + 18] * c_x_idx_3;
      K1[c_ix] += rtb_P[c_ix + 23] * y_idx_1;
      K1[c_ix + 2] = 0.0;
      K1[c_ix + 2] += rtb_P[c_ix + 18] * c_x_idx_0;
      K1[c_ix + 2] += rtb_P[c_ix + 23] * rtb_Gain1;
      rtb_UnitDelay34_0[c_ix] = rtb_UnitDelay34[3 + c_ix] - rtb_X[3 + c_ix];
    }

    rtb_X[3] += K1[0] * rtb_UnitDelay34_0[0] + K1[2] * rtb_UnitDelay34_0[1];
    rtb_X[4] += K1[1] * rtb_UnitDelay34_0[0] + K1[3] * rtb_UnitDelay34_0[1];
    c_x[0] = 1.0 - K1[0];
    c_x[1] = 0.0 - K1[1];
    c_x[2] = 0.0 - K1[2];
    c_x[3] = 1.0 - K1[3];
    for (c_ix = 0; c_ix < 2; c_ix++) {
      K1[c_ix] = 0.0;
      K1[c_ix] += c_x[c_ix] * rtb_P[18];
      K1[c_ix] += c_x[c_ix + 2] * rtb_P[19];
      K1[c_ix + 2] = 0.0;
      K1[c_ix + 2] += c_x[c_ix] * rtb_P[23];
      K1[c_ix + 2] += c_x[c_ix + 2] * rtb_P[24];
    }

    rtb_P[18] = K1[0];
    rtb_P[19] = K1[1];
    rtb_P[23] = K1[2];
    rtb_P[24] = K1[3];
  }

  // End of MATLAB Function: '<S1>/SLAM_UKF'
  for (i = 0; i < 5; i++) {
    // Outport: '<Root>/X_UKF_SLAM'
    rtY.X_UKF_SLAM[i] = rtb_X[i];

    // Update for UnitDelay: '<S1>/Unit Delay34'
    rtDW.UnitDelay34_DSTATE[i] = rtb_X[i];
  }

  // Update for UnitDelay: '<S1>/Unit Delay33'
  memcpy(&rtDW.UnitDelay33_DSTATE[0], &rtb_P[0], 25U * sizeof(real_T));

  // Update for UnitDelay: '<S1>/Unit Delay1'
  for (i = 0; i < 5; i++) {
    rtDW.UnitDelay1_DSTATE[i] = rtb_X[i];
  }

  // End of Update for UnitDelay: '<S1>/Unit Delay1'

  // Update for UnitDelay: '<S1>/Unit Delay35' incorporates:
  //   Inport: '<Root>/SLAM_counter'
  //   MATLAB Function: '<S1>/SLAM_Check'

  rtDW.UnitDelay35_DSTATE[0] = ajj;
  rtDW.UnitDelay35_DSTATE[1] = SLAM_Y_out;
  rtDW.UnitDelay35_DSTATE[2] = SLAM_Heading_out;
  rtDW.UnitDelay35_DSTATE[3] = rtU.SLAM_counter;

  // Update for UnitDelay: '<S1>/Unit Delay37'
  memcpy(&rtDW.UnitDelay37_DSTATE[0], &p_sqrt_data[0], 25U * sizeof(real_T));

  // Update for UnitDelay: '<S1>/Unit Delay36'
  memcpy(&rtDW.UnitDelay36_DSTATE[0], &rtb_R_last_o[0], 25U * sizeof(real_T));
}

// Model initialize function
void Localization_UKFModelClass::initialize()
{
  // InitializeConditions for UnitDelay: '<S1>/Unit Delay33'
  memcpy(&rtDW.UnitDelay33_DSTATE[0], &rtConstP.UnitDelay33_InitialCondition[0],
         25U * sizeof(real_T));

  // InitializeConditions for UnitDelay: '<S1>/Unit Delay37'
  memcpy(&rtDW.UnitDelay37_DSTATE[0], &rtConstP.UnitDelay37_InitialCondition[0],
         25U * sizeof(real_T));

  // InitializeConditions for UnitDelay: '<S1>/Unit Delay36'
  memcpy(&rtDW.UnitDelay36_DSTATE[0], &rtConstP.UnitDelay36_InitialCondition[0],
         25U * sizeof(real_T));
}

// Constructor
Localization_UKFModelClass::Localization_UKFModelClass()
{
}

// Destructor
Localization_UKFModelClass::~Localization_UKFModelClass()
{
  // Currently there is no destructor body generated.
}

// Real-Time Model get method
RT_MODEL * Localization_UKFModelClass::getRTM()
{
  return (&rtM);
}

//
// File trailer for generated code.
//
// [EOF]
//
