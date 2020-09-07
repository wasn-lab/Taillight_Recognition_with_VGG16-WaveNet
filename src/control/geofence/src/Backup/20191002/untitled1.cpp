//
// File: untitled1.cpp
//
// Code generated for Simulink model 'untitled1'.
//
// Model version                  : 1.7
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Wed Sep 25 14:31:33 2019
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
// Code generation objectives:
//    1. Execution efficiency
//    2. RAM efficiency
// Validation result: Not run
//
#include "untitled1.h"
#define NumBitsPerChar                 8U

extern real_T rt_powd_snf(real_T u0, real_T u1);
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

// Model step function
void untitled1ModelClass::step()
{
  real_T t;
  int8_T Trigger_matrix[100];
  real_T Min_Range_matrix[100];
  real_T Object_X[5];
  real_T Object_Y[5];
  real_T Nearest_distance[4];
  real_T Distance;
  int32_T b_i;
  int32_T c_idx;
  int32_T i;
  real_T X_points_one_tmp;
  int32_T X_points_tmp;
  int32_T X_points_tmp_0;
  boolean_T exitg1;

  // MATLAB Function: '<Root>/Boundary' incorporates:
  //   Inport: '<Root>/Input'
  //   Inport: '<Root>/Input1'
  //   Inport: '<Root>/Input2'
  //   Inport: '<Root>/Input3'

  memset(&rtDW.X_points_one[0], 0, 10201U * sizeof(real_T));
  memset(&rtDW.Y_points_one[0], 0, 10201U * sizeof(real_T));
  memset(&rtDW.X_points_two[0], 0, 10201U * sizeof(real_T));
  memset(&rtDW.Y_points_two[0], 0, 10201U * sizeof(real_T));
  t = 1.0;
  for (b_i = 0; b_i < 101; b_i++) {
    Distance = (real_T)b_i * 0.01;
    X_points_one_tmp = Distance * Distance;
    i = (int32_T)t - 1;
    rtDW.X_points_one[i] = ((((rtU.Input[1] * Distance + rtU.Input[0]) +
      X_points_one_tmp * rtU.Input[2]) + rtU.Input[3] * rt_powd_snf(Distance,
      3.0)) + rtU.Input[4] * rt_powd_snf(Distance, 4.0)) + rtU.Input[5] *
      rt_powd_snf(Distance, 5.0);
    rtDW.Y_points_one[i] = ((((rtU.Input1[1] * Distance + rtU.Input1[0]) +
      X_points_one_tmp * rtU.Input1[2]) + rtU.Input1[3] * rt_powd_snf(Distance,
      3.0)) + rtU.Input1[4] * rt_powd_snf(Distance, 4.0)) + rtU.Input1[5] *
      rt_powd_snf(Distance, 5.0);
    rtDW.X_points_two[i] = ((((rtU.Input[7] * Distance + rtU.Input[6]) +
      X_points_one_tmp * rtU.Input[8]) + rtU.Input[9] * rt_powd_snf(Distance,
      3.0)) + rtU.Input[10] * rt_powd_snf(Distance, 4.0)) + rtU.Input[11] *
      rt_powd_snf(Distance, 5.0);
    rtDW.Y_points_two[i] = ((((rtU.Input1[7] * Distance + rtU.Input1[6]) +
      X_points_one_tmp * rtU.Input1[8]) + rtU.Input1[9] * rt_powd_snf(Distance,
      3.0)) + rtU.Input1[10] * rt_powd_snf(Distance, 4.0)) + rtU.Input1[11] *
      rt_powd_snf(Distance, 5.0);
    t++;
  }

  for (i = 0; i < 101; i++) {
    for (b_i = 0; b_i < 101; b_i++) {
      c_idx = 101 * i + b_i;
      X_points_tmp = b_i + 202 * i;
      rtDW.X_points[X_points_tmp] = rtDW.X_points_one[c_idx];
      X_points_tmp_0 = X_points_tmp + 101;
      rtDW.X_points[X_points_tmp_0] = rtDW.X_points_two[c_idx];
      rtDW.Y_points[X_points_tmp] = rtDW.Y_points_one[c_idx];
      rtDW.Y_points[X_points_tmp_0] = rtDW.Y_points_two[c_idx];
    }
  }

  memset(&rtDW.Line_length_one[0], 0, 10201U * sizeof(real_T));
  memset(&rtDW.Line_length_two[0], 0, 10201U * sizeof(real_T));
  for (b_i = 0; b_i < 100; b_i++) {
    t = rtDW.X_points_one[b_i] - rtDW.X_points_one[b_i + 1];
    Distance = rtDW.Y_points_one[b_i] - rtDW.Y_points_one[b_i + 1];
    rtDW.Line_length_one[b_i + 1] = std::sqrt(t * t + Distance * Distance) +
      rtDW.Line_length_one[b_i];
  }

  rtDW.Line_length_two[0] = rtDW.Line_length_one[100];
  for (b_i = 0; b_i < 100; b_i++) {
    t = rtDW.X_points_two[b_i] - rtDW.X_points_two[b_i + 1];
    Distance = rtDW.Y_points_two[b_i] - rtDW.Y_points_two[b_i + 1];
    rtDW.Line_length_two[b_i + 1] = std::sqrt(t * t + Distance * Distance) +
      rtDW.Line_length_two[b_i];
  }

  for (i = 0; i < 101; i++) {
    for (b_i = 0; b_i < 101; b_i++) {
      c_idx = 101 * i + b_i;
      X_points_tmp = b_i + 202 * i;
      rtDW.Line_length[X_points_tmp] = rtDW.Line_length_one[c_idx];
      rtDW.Line_length[X_points_tmp + 101] = rtDW.Line_length_two[c_idx];
    }
  }

  for (i = 0; i < 100; i++) {
    Trigger_matrix[i] = 0;
    Min_Range_matrix[i] = 100.0;
  }

  for (b_i = 0; b_i < (int32_T)rtU.Input3; b_i++) {
    t = (1.0 + (real_T)b_i) * 2.0;
    i = (int32_T)(t - 1.0);
    Object_X[0] = rtU.Input2[i - 1];
    Object_X[1] = rtU.Input2[i + 199];
    Object_X[2] = rtU.Input2[i + 399];
    Object_X[3] = rtU.Input2[i + 599];
    Object_X[4] = (((rtU.Input2[i - 1] + rtU.Input2[i + 199]) + rtU.Input2[i +
                    399]) + rtU.Input2[i + 599]) / 4.0;
    i = (int32_T)t;
    Object_Y[0] = rtU.Input2[i - 1];
    Nearest_distance[0] = 100.0;
    Object_Y[1] = rtU.Input2[i + 199];
    Nearest_distance[1] = 100.0;
    Object_Y[2] = rtU.Input2[i + 399];
    Nearest_distance[2] = 100.0;
    Object_Y[3] = rtU.Input2[i + 599];
    Nearest_distance[3] = 100.0;
    Object_Y[4] = (((rtU.Input2[i - 1] + rtU.Input2[i + 199]) + rtU.Input2[i +
                    399]) + rtU.Input2[i + 599]) / 4.0;
    i = 0;
    for (c_idx = 0; c_idx < 5; c_idx++) {
      t = 100.0;
      for (X_points_tmp = 0; X_points_tmp < 202; X_points_tmp++) {
        Distance = rtDW.X_points[X_points_tmp] - Object_X[c_idx];
        X_points_one_tmp = rtDW.Y_points[X_points_tmp] - Object_Y[c_idx];
        Distance = std::sqrt(Distance * Distance + X_points_one_tmp *
                             X_points_one_tmp);
        if (Distance < t) {
          t = Distance;
          Nearest_distance[c_idx] = rtDW.Line_length[X_points_tmp];
        }
      }

      if (t < 1.5) {
        i = 1;
      }
    }

    Min_Range_matrix[b_i] = 100.0;
    if (i == 1) {
      if (!rtIsNaN(Nearest_distance[0])) {
        c_idx = 1;
      } else {
        c_idx = 0;
        X_points_tmp = 2;
        exitg1 = false;
        while ((!exitg1) && (X_points_tmp < 5)) {
          if (!rtIsNaN(Nearest_distance[X_points_tmp - 1])) {
            c_idx = X_points_tmp;
            exitg1 = true;
          } else {
            X_points_tmp++;
          }
        }
      }

      if (c_idx == 0) {
        Min_Range_matrix[b_i] = Nearest_distance[0];
      } else {
        t = Nearest_distance[c_idx - 1];
        while (c_idx + 1 < 5) {
          if (t > Nearest_distance[c_idx]) {
            t = Nearest_distance[c_idx];
          }

          c_idx++;
        }

        Min_Range_matrix[b_i] = t;
      }
    }

    Trigger_matrix[b_i] = (int8_T)i;
  }

  b_i = Trigger_matrix[0];
  for (i = 1; i + 1 < 101; i++) {
    if (b_i < Trigger_matrix[i]) {
      b_i = Trigger_matrix[i];
    }
  }

  if (!rtIsNaN(Min_Range_matrix[0])) {
    i = 1;
  } else {
    i = 0;
    c_idx = 2;
    exitg1 = false;
    while ((!exitg1) && (c_idx < 101)) {
      if (!rtIsNaN(Min_Range_matrix[c_idx - 1])) {
        i = c_idx;
        exitg1 = true;
      } else {
        c_idx++;
      }
    }
  }

  if (i == 0) {
    // Outport: '<Root>/Output1'
    rtY.Output1 = Min_Range_matrix[0];
    i = 1;
  } else {
    t = Min_Range_matrix[i - 1];
    for (c_idx = i; c_idx + 1 < 101; c_idx++) {
      if (t > Min_Range_matrix[c_idx]) {
        t = Min_Range_matrix[c_idx];
        i = c_idx + 1;
      }
    }

    // Outport: '<Root>/Output1'
    rtY.Output1 = t;
  }

  // Outport: '<Root>/Output' incorporates:
  //   MATLAB Function: '<Root>/Boundary'

  rtY.Output = b_i;

  // Outport: '<Root>/Output2' incorporates:
  //   MATLAB Function: '<Root>/Boundary'

  rtY.Output2 = i;
}

// Model initialize function
void untitled1ModelClass::initialize()
{
  // Registration code

  // initialize non-finites
  rt_InitInfAndNaN(sizeof(real_T));
}

// Constructor
untitled1ModelClass::untitled1ModelClass()
{
}

// Destructor
untitled1ModelClass::~untitled1ModelClass()
{
  // Currently there is no destructor body generated.
}

// Real-Time Model get method
RT_MODEL * untitled1ModelClass::getRTM()
{
  return (&rtM);
}

//
// File trailer for generated code.
//
// [EOF]
//
