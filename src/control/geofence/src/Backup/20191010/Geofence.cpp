//
// File: Geofence.cpp
//
// Code generated for Simulink model 'Geofence'.
//
// Model version                  : 1.9
// Simulink Coder version         : 8.14 (R2018a) 06-Feb-2018
// C/C++ source code generated on : Wed Oct  2 13:29:17 2019
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
// Code generation objectives:
//    1. Execution efficiency
//    2. RAM efficiency
// Validation result: Not run
//
#include "Geofence.h"
#define NumBitsPerChar 8U

extern real_T rt_powd_snf(real_T u0, real_T u1);
extern "C" {
extern real_T rtGetInf(void);
extern real32_T rtGetInfF(void);
extern real_T rtGetMinusInf(void);
extern real32_T rtGetMinusInfF(void);
}  // extern "C"
extern "C" {
extern real_T rtGetNaN(void);
extern real32_T rtGetNaNF(void);
}  // extern "C"

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
typedef struct
{
  struct
  {
    uint32_T wordH;
    uint32_T wordL;
  } words;
} BigEndianIEEEDouble;

typedef struct
{
  struct
  {
    uint32_T wordL;
    uint32_T wordH;
  } words;
} LittleEndianIEEEDouble;

typedef struct
{
  union
  {
    real32_T wordLreal;
    uint32_T wordLuint;
  } wordL;
} IEEESingle;
}  // extern "C"
extern "C" {
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
  if (bitsPerReal == 32U)
  {
    inf = rtGetInfF();
  }
  else
  {
    union
    {
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
  if (bitsPerReal == 32U)
  {
    minf = rtGetMinusInfF();
  }
  else
  {
    union
    {
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
extern "C" {
//
// Initialize rtNaN needed by the generated code.
// NaN is initialized as non-signaling. Assumes IEEE.
//
real_T rtGetNaN(void)
{
  size_t bitsPerReal = sizeof(real_T) * (NumBitsPerChar);
  real_T nan = 0.0;
  if (bitsPerReal == 32U)
  {
    nan = rtGetNaNF();
  }
  else
  {
    union
    {
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
  (void)(realSize);
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
  return (boolean_T)((value == rtInf || value == rtMinusInf) ? 1U : 0U);
}

// Test if single-precision value is infinite
boolean_T rtIsInfF(real32_T value)
{
  return (boolean_T)(((value) == rtInfF || (value) == rtMinusInfF) ? 1U : 0U);
}

// Test if value is not a number
boolean_T rtIsNaN(real_T value)
{
  return (boolean_T)((value != value) ? 1U : 0U);
}

// Test if single-precision value is not a number
boolean_T rtIsNaNF(real32_T value)
{
  return (boolean_T)(((value != value) ? 1U : 0U));
}
}
real_T rt_powd_snf(real_T u0, real_T u1)
{
  real_T y;
  real_T tmp;
  real_T tmp_0;
  if (rtIsNaN(u0) || rtIsNaN(u1))
  {
    y = (rtNaN);
  }
  else
  {
    tmp = std::abs(u0);
    tmp_0 = std::abs(u1);
    if (rtIsInf(u1))
    {
      if (tmp == 1.0)
      {
        y = 1.0;
      }
      else if (tmp > 1.0)
      {
        if (u1 > 0.0)
        {
          y = (rtInf);
        }
        else
        {
          y = 0.0;
        }
      }
      else if (u1 > 0.0)
      {
        y = 0.0;
      }
      else
      {
        y = (rtInf);
      }
    }
    else if (tmp_0 == 0.0)
    {
      y = 1.0;
    }
    else if (tmp_0 == 1.0)
    {
      if (u1 > 0.0)
      {
        y = u0;
      }
      else
      {
        y = 1.0 / u0;
      }
    }
    else if (u1 == 2.0)
    {
      y = u0 * u0;
    }
    else if ((u1 == 0.5) && (u0 >= 0.0))
    {
      y = std::sqrt(u0);
    }
    else if ((u0 < 0.0) && (u1 > std::floor(u1)))
    {
      y = (rtNaN);
    }
    else
    {
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
  real_T Speed_matrix[100];
  real_T Object_X[5];
  real_T Object_Y[5];
  real_T Nearest_distance[4];
  int32_T Trigger;
  real_T Distance;
  int32_T c_idx;
  int32_T i;
  real_T X_points_one_tmp;
  int32_T X_points_tmp;
  int32_T X_points_tmp_0;
  boolean_T exitg1;

  // MATLAB Function: '<Root>/Calculator' incorporates:
  //   Inport: '<Root>/BoundingBox'
  //   Inport: '<Root>/OB_num'
  //   Inport: '<Root>/X_Poly'
  //   Inport: '<Root>/Y_Poly'

  memset(&rtDW.X_points_one[0], 0, 10201U * sizeof(real_T));
  memset(&rtDW.Y_points_one[0], 0, 10201U * sizeof(real_T));
  memset(&rtDW.X_points_two[0], 0, 10201U * sizeof(real_T));
  memset(&rtDW.Y_points_two[0], 0, 10201U * sizeof(real_T));
  t = 1.0;
  for (i = 0; i < 101; i++)
  {
    Distance = (real_T)i * 0.01;
    X_points_one_tmp = Distance * Distance;
    Trigger = (int32_T)t - 1;
    rtDW.X_points_one[Trigger] = ((((rtU.X_Poly[1] * Distance + rtU.X_Poly[0]) + X_points_one_tmp * rtU.X_Poly[2]) +
                                   rtU.X_Poly[3] * rt_powd_snf(Distance, 3.0)) +
                                  rtU.X_Poly[4] * rt_powd_snf(Distance, 4.0)) +
                                 rtU.X_Poly[5] * rt_powd_snf(Distance, 5.0);
    rtDW.Y_points_one[Trigger] = ((((rtU.Y_Poly[1] * Distance + rtU.Y_Poly[0]) + X_points_one_tmp * rtU.Y_Poly[2]) +
                                   rtU.Y_Poly[3] * rt_powd_snf(Distance, 3.0)) +
                                  rtU.Y_Poly[4] * rt_powd_snf(Distance, 4.0)) +
                                 rtU.Y_Poly[5] * rt_powd_snf(Distance, 5.0);
    rtDW.X_points_two[Trigger] = ((((rtU.X_Poly[7] * Distance + rtU.X_Poly[6]) + X_points_one_tmp * rtU.X_Poly[8]) +
                                   rtU.X_Poly[9] * rt_powd_snf(Distance, 3.0)) +
                                  rtU.X_Poly[10] * rt_powd_snf(Distance, 4.0)) +
                                 rtU.X_Poly[11] * rt_powd_snf(Distance, 5.0);
    rtDW.Y_points_two[Trigger] = ((((rtU.Y_Poly[7] * Distance + rtU.Y_Poly[6]) + X_points_one_tmp * rtU.Y_Poly[8]) +
                                   rtU.Y_Poly[9] * rt_powd_snf(Distance, 3.0)) +
                                  rtU.Y_Poly[10] * rt_powd_snf(Distance, 4.0)) +
                                 rtU.Y_Poly[11] * rt_powd_snf(Distance, 5.0);
    t++;
  }

  for (i = 0; i < 101; i++)
  {
    for (Trigger = 0; Trigger < 101; Trigger++)
    {
      c_idx = 101 * i + Trigger;
      X_points_tmp = Trigger + 202 * i;
      rtDW.X_points[X_points_tmp] = rtDW.X_points_one[c_idx];
      X_points_tmp_0 = X_points_tmp + 101;
      rtDW.X_points[X_points_tmp_0] = rtDW.X_points_two[c_idx];
      rtDW.Y_points[X_points_tmp] = rtDW.Y_points_one[c_idx];
      rtDW.Y_points[X_points_tmp_0] = rtDW.Y_points_two[c_idx];
    }
  }

  memset(&rtDW.Line_length_one[0], 0, 10201U * sizeof(real_T));
  memset(&rtDW.Line_length_two[0], 0, 10201U * sizeof(real_T));
  for (i = 0; i < 100; i++)
  {
    t = rtDW.X_points_one[i] - rtDW.X_points_one[i + 1];
    Distance = rtDW.Y_points_one[i] - rtDW.Y_points_one[i + 1];
    rtDW.Line_length_one[i + 1] = std::sqrt(t * t + Distance * Distance) + rtDW.Line_length_one[i];
  }

  rtDW.Line_length_two[0] = rtDW.Line_length_one[100];
  for (i = 0; i < 100; i++)
  {
    t = rtDW.X_points_two[i] - rtDW.X_points_two[i + 1];
    Distance = rtDW.Y_points_two[i] - rtDW.Y_points_two[i + 1];
    rtDW.Line_length_two[i + 1] = std::sqrt(t * t + Distance * Distance) + rtDW.Line_length_two[i];
  }

  for (i = 0; i < 101; i++)
  {
    for (Trigger = 0; Trigger < 101; Trigger++)
    {
      c_idx = 101 * i + Trigger;
      X_points_tmp = Trigger + 202 * i;
      rtDW.Line_length[X_points_tmp] = rtDW.Line_length_one[c_idx];
      rtDW.Line_length[X_points_tmp + 101] = rtDW.Line_length_two[c_idx];
    }
  }

  for (i = 0; i < 100; i++)
  {
    Trigger_matrix[i] = 0;
    Min_Range_matrix[i] = 100.0;
    Speed_matrix[i] = 100.0;
  }

  for (i = 0; i < (int32_T)rtU.OB_num; i++)
  {
    Object_X[0] = rtU.BoundingBox[i];
    Object_Y[0] = rtU.BoundingBox[i + 400];
    Nearest_distance[0] = 100.0;
    Object_X[1] = rtU.BoundingBox[i + 100];
    Object_Y[1] = rtU.BoundingBox[i + 500];
    Nearest_distance[1] = 100.0;
    Object_X[2] = rtU.BoundingBox[i + 200];
    Object_Y[2] = rtU.BoundingBox[i + 600];
    Nearest_distance[2] = 100.0;
    Object_X[3] = rtU.BoundingBox[i + 300];
    Object_Y[3] = rtU.BoundingBox[i + 700];
    Nearest_distance[3] = 100.0;
    Object_X[4] =
        (((rtU.BoundingBox[i + 100] + rtU.BoundingBox[i]) + rtU.BoundingBox[i + 200]) + rtU.BoundingBox[i + 300]) / 4.0;
    Object_Y[4] = (((rtU.BoundingBox[i + 400] + rtU.BoundingBox[i + 500]) + rtU.BoundingBox[i + 600]) +
                   rtU.BoundingBox[i + 700]) /
                  4.0;
    Trigger = 0;
    for (c_idx = 0; c_idx < 5; c_idx++)
    {
      t = 100.0;
      for (X_points_tmp = 0; X_points_tmp < 202; X_points_tmp++)
      {
        Distance = rtDW.X_points[X_points_tmp] - Object_X[c_idx];
        X_points_one_tmp = rtDW.Y_points[X_points_tmp] - Object_Y[c_idx];
        Distance = std::sqrt(Distance * Distance + X_points_one_tmp * X_points_one_tmp);
        if (Distance < t)
        {
          t = Distance;
          Nearest_distance[c_idx] = rtDW.Line_length[X_points_tmp];
        }
      }

      if (t < 1.5)
      {
        Trigger = 1;
      }
    }

    Min_Range_matrix[i] = 100.0;
    if (Trigger == 1)
    {
      if (!rtIsNaN(Nearest_distance[0]))
      {
        c_idx = 1;
      }
      else
      {
        c_idx = 0;
        X_points_tmp = 2;
        exitg1 = false;
        while ((!exitg1) && (X_points_tmp < 5))
        {
          if (!rtIsNaN(Nearest_distance[X_points_tmp - 1]))
          {
            c_idx = X_points_tmp;
            exitg1 = true;
          }
          else
          {
            X_points_tmp++;
          }
        }
      }

      if (c_idx == 0)
      {
        Min_Range_matrix[i] = Nearest_distance[0];
      }
      else
      {
        t = Nearest_distance[c_idx - 1];
        while (c_idx + 1 < 5)
        {
          if (t > Nearest_distance[c_idx])
          {
            t = Nearest_distance[c_idx];
          }

          c_idx++;
        }

        Min_Range_matrix[i] = t;
      }
    }

    Trigger_matrix[i] = (int8_T)Trigger;
    Speed_matrix[i] = rtU.BoundingBox[800 + i];
  }

  i = Trigger_matrix[0];
  for (Trigger = 1; Trigger + 1 < 101; Trigger++)
  {
    if (i < Trigger_matrix[Trigger])
    {
      i = Trigger_matrix[Trigger];
    }
  }

  if (!rtIsNaN(Min_Range_matrix[0]))
  {
    Trigger = 0;
  }
  else
  {
    Trigger = -1;
    c_idx = 2;
    exitg1 = false;
    while ((!exitg1) && (c_idx < 101))
    {
      if (!rtIsNaN(Min_Range_matrix[c_idx - 1]))
      {
        Trigger = c_idx - 1;
        exitg1 = true;
      }
      else
      {
        c_idx++;
      }
    }
  }

  if (Trigger + 1 == 0)
  {
    // Outport: '<Root>/Range'
    rtY.Range = Min_Range_matrix[0];
    Trigger = 0;
  }
  else
  {
    t = Min_Range_matrix[Trigger];
    for (c_idx = Trigger + 1; c_idx + 1 < 101; c_idx++)
    {
      if (t > Min_Range_matrix[c_idx])
      {
        t = Min_Range_matrix[c_idx];
        Trigger = c_idx;
      }
    }

    // Outport: '<Root>/Range'
    rtY.Range = t;
  }

  // Outport: '<Root>/Trigger' incorporates:
  //   MATLAB Function: '<Root>/Calculator'

  rtY.Trigger = i;

  // Outport: '<Root>/Obj_Speed' incorporates:
  //   MATLAB Function: '<Root>/Calculator'

  rtY.Obj_Speed = Speed_matrix[Trigger];
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
RT_MODEL* untitled1ModelClass::getRTM()
{
  return (&rtM);
}

//
// File trailer for generated code.
//
// [EOF]
//
