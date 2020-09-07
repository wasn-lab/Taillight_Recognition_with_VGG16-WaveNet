//
// File: untitled.cpp
//
// Code generated for Simulink model 'untitled'.
//
// Model version                  : 1.6
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Fri Aug 30 16:08:25 2019
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
// Code generation objectives:
//    1. Execution efficiency
//    2. RAM efficiency
// Validation result: Not run
//
#include "untitled.h"
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
void untitledModelClass::step()
{
  real_T matrix[66];
  real_T X_points_one[11];
  real_T Y_points_one[11];
  real_T X_points_two[11];
  real_T Y_points_two[11];
  int8_T Trigger[10];
  real_T Min_Range[10];
  int8_T Nearest_i[4];
  real_T Nearest_range[4];
  int32_T Trigger_two;
  real_T Accumulation;
  real_T minimum;
  real_T Distance;
  real_T Min_Range_one;
  int32_T c_i;
  int32_T idx;
  int32_T d_k;
  real_T b_a;
  int32_T i;
  int32_T b_j;
  boolean_T exitg1;

  // MATLAB Function: '<Root>/Function' incorporates:
  //   Inport: '<Root>/Input'
  //   Inport: '<Root>/Input1'
  //   Inport: '<Root>/Input2'
  //   Inport: '<Root>/Input3'

  memset(&matrix[0], 0, 66U * sizeof(real_T));
  for (i = 0; i < 11; i++) {
    Accumulation = (real_T)i * 0.1;
    for (idx = 0; idx < 6; idx++) {
      matrix[((int32_T)(Accumulation / 0.1 + 1.0) + 11 * idx) - 1] = rt_powd_snf
        (Accumulation, (real_T)idx);
    }
  }

  for (i = 0; i < 11; i++) {
    X_points_one[i] = 0.0;
    Y_points_one[i] = 0.0;
    X_points_two[i] = 0.0;
    Y_points_two[i] = 0.0;
    for (idx = 0; idx < 6; idx++) {
      X_points_one[i] += matrix[11 * idx + i] * rtU.Input[idx];
      Y_points_one[i] += matrix[11 * idx + i] * rtU.Input1[idx];
      X_points_two[i] += matrix[11 * idx + i] * rtU.Input[6 + idx];
      Y_points_two[i] += matrix[11 * idx + i] * rtU.Input1[6 + idx];
    }
  }

  for (i = 0; i < 10; i++) {
    Trigger[i] = 0;
    Min_Range[i] = 100.0;
  }

  for (i = 0; i < (int32_T)rtU.Input3; i++) {
    idx = 0;
    Trigger_two = 0;
    for (b_j = 0; b_j < 4; b_j++) {
      Nearest_i[b_j] = 10;
      Nearest_range[b_j] = 100.0;
      Accumulation = 0.0;
      minimum = 100.0;
      for (c_i = 0; c_i < 10; c_i++) {
        Min_Range_one = X_points_one[c_i + 1] - X_points_one[c_i];
        Distance = Y_points_one[c_i + 1] - Y_points_one[c_i];
        Accumulation += std::sqrt(Min_Range_one * Min_Range_one + Distance *
          Distance);
        Distance = (1.0 + (real_T)i) * 2.0;
        Min_Range_one = X_points_one[c_i] - rtU.Input2[((int32_T)(Distance - 1.0)
          + 100 * b_j) - 1];
        Distance = Y_points_one[c_i] - rtU.Input2[((int32_T)Distance + 100 * b_j)
          - 1];
        Distance = std::sqrt(Min_Range_one * Min_Range_one + Distance * Distance);
        if (Distance < minimum) {
          minimum = Distance;
          Nearest_i[b_j] = (int8_T)(1 + c_i);
          Nearest_range[b_j] = Accumulation;
        }
      }

      if (minimum < 1.5) {
        idx = 1;
      }
    }

    Min_Range_one = 100.0;
    if (idx == 1) {
      b_j = Nearest_i[0];
      c_i = 0;
      for (d_k = 2; d_k < 5; d_k++) {
        if (b_j > Nearest_i[d_k - 1]) {
          b_j = Nearest_i[d_k - 1];
          c_i = d_k - 1;
        }
      }

      Min_Range_one = Nearest_range[c_i];
    }

    for (b_j = 0; b_j < 4; b_j++) {
      Accumulation = 0.0;
      minimum = 100.0;
      for (c_i = 0; c_i < 10; c_i++) {
        Distance = X_points_two[c_i + 1] - X_points_two[c_i];
        b_a = Y_points_two[c_i + 1] - Y_points_two[c_i];
        Accumulation += std::sqrt(Distance * Distance + b_a * b_a);
        Distance = X_points_two[c_i] - rtU.Input2[((int32_T)((1.0 + (real_T)i) *
          2.0 - 1.0) + 100 * b_j) - 1];
        b_a = Y_points_two[c_i] - rtU.Input2[((int32_T)((1.0 + (real_T)i) * 2.0)
          + 100 * b_j) - 1];
        Distance = std::sqrt(Distance * Distance + b_a * b_a);
        if (Distance < minimum) {
          minimum = Distance;
          Nearest_i[b_j] = (int8_T)(1 + c_i);
          Nearest_range[b_j] = Accumulation;
        }
      }

      if (minimum < 1.5) {
        Trigger_two = 1;
      }
    }

    Accumulation = 100.0;
    if (Trigger_two == 1) {
      b_j = Nearest_i[0];
      c_i = 0;
      for (d_k = 2; d_k < 5; d_k++) {
        if (b_j > Nearest_i[d_k - 1]) {
          b_j = Nearest_i[d_k - 1];
          c_i = d_k - 1;
        }
      }

      Accumulation = Nearest_range[c_i];
    }

    if ((real_T)idx > Trigger_two) {
      Trigger[i] = (int8_T)idx;
    } else {
      Trigger[i] = (int8_T)Trigger_two;
    }

    if ((Min_Range_one < Accumulation) || rtIsNaN(Accumulation)) {
      Min_Range[i] = Min_Range_one;
    } else {
      Min_Range[i] = Accumulation;
    }
  }

  i = Trigger[0];
  for (idx = 1; idx + 1 < 11; idx++) {
    if (i < Trigger[idx]) {
      i = Trigger[idx];
    }
  }

  if (!rtIsNaN(Min_Range[0])) {
    idx = 1;
  } else {
    idx = 0;
    Trigger_two = 2;
    exitg1 = false;
    while ((!exitg1) && (Trigger_two < 11)) {
      if (!rtIsNaN(Min_Range[Trigger_two - 1])) {
        idx = Trigger_two;
        exitg1 = true;
      } else {
        Trigger_two++;
      }
    }
  }

  if (idx == 0) {
    // Outport: '<Root>/Output1'
    rtY.Output1 = Min_Range[0];
  } else {
    Accumulation = Min_Range[idx - 1];
    while (idx + 1 < 11) {
      if (Accumulation > Min_Range[idx]) {
        Accumulation = Min_Range[idx];
      }

      idx++;
    }

    // Outport: '<Root>/Output1'
    rtY.Output1 = Accumulation;
  }

  // Outport: '<Root>/Output' incorporates:
  //   MATLAB Function: '<Root>/Function'

  rtY.Output = i;
}

// Model initialize function
void untitledModelClass::initialize()
{
  // Registration code

  // initialize non-finites
  rt_InitInfAndNaN(sizeof(real_T));
}

// Constructor
untitledModelClass::untitledModelClass()
{
}

// Destructor
untitledModelClass::~untitledModelClass()
{
  // Currently there is no destructor body generated.
}

// Real-Time Model get method
RT_MODEL * untitledModelClass::getRTM()
{
  return (&rtM);
}

//
// File trailer for generated code.
//
// [EOF]
//
