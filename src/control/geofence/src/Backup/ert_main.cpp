
//
// File: ert_main.cpp
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
#include <stddef.h>
#include <stdio.h>                     // This ert_main.c example uses printf/fflush 
#include "untitled.h"                  // Model's header file
#include "rtwtypes.h"
#include <iostream>  

static untitledModelClass rtObj;       // Instance of model class
using namespace std;
//


// Associating rt_OneStep with a real-time clock or interrupt service routine
// is what makes the generated code "real-time".  The function rt_OneStep is
// always associated with the base rate of the model.  Subrates are managed
// by the base rate from inside the generated code.  Enabling/disabling
// interrupts and floating point context switches are target specific.  This
// example code indicates where these should take place relative to executing
// the generated code step function.  Overrun behavior should be tailored to
// your application needs.  This example simply sets an error status in the
// real-time model and returns from rt_OneStep.
//
void rt_OneStep(void);
void rt_OneStep(void)
{
  static boolean_T OverrunFlag = false;

  // Disable interrupts here

  // Check for overrun
  if (OverrunFlag) {
    rtmSetErrorStatus(rtObj.getRTM(), "Overrun");
    return;
  }

  OverrunFlag = true;

  // Save FPU context here (if necessary)
  // Re-enable timer or interrupt here
  // Set model inputs here

  // Step the model
  //real_T Input[12];                    
  //real_T Input1[12];                  
  //real_T Input2[400];                 
  //real_T Input3;

	rtObj.rtU.Input2[0] = -8.5756778717; 
	rtObj.rtU.Input2[1] = -107.649917603; 
	rtObj.rtU.Input[0] = -8.68999958038;
	rtObj.rtU.Input[1] = 0.331709235907;
	rtObj.rtU.Input[2] = 0.00338661414571;
	rtObj.rtU.Input[3] = -2.53303170204;
	rtObj.rtU.Input[4] = 3.8808734417;
	rtObj.rtU.Input[5] = -1.56861579418;
	rtObj.rtU.Input1[0] = -110.559997559;
	rtObj.rtU.Input1[1] = 2.98160505295;
	rtObj.rtU.Input1[2] = -0.000376767246053;
	rtObj.rtU.Input1[3] = -0.670077860355;
	rtObj.rtU.Input1[4] = 0.994797825813;
	rtObj.rtU.Input1[5] = -0.395865321159;
	rtObj.rtU.Input[6] = -8.5756778717;
	rtObj.rtU.Input[7] = 2.09900903702;
	rtObj.rtU.Input[8] = 0.0843495130539;
	rtObj.rtU.Input[9] = -0.374515295029;
	rtObj.rtU.Input[10] = 0.379369705915;
	rtObj.rtU.Input[11] = -0.115266129375;
	rtObj.rtU.Input1[6] = -107.649917603;
	rtObj.rtU.Input1[7] = 14.8524122238;
	rtObj.rtU.Input1[8] = -0.0119206495583;
	rtObj.rtU.Input1[9] = -0.971147060394;
	rtObj.rtU.Input1[10] = 1.48245251179;
	rtObj.rtU.Input1[11] = -0.598119616508;
/*
	rtObj.rtU.Input2[0] = 0; 
	rtObj.rtU.Input2[1] = 0; 
	rtObj.rtU.Input[0] = 0;
	rtObj.rtU.Input[1] = 0;
	rtObj.rtU.Input[2] = 0;
	rtObj.rtU.Input[3] = 0;
	rtObj.rtU.Input[4] = 0;
	rtObj.rtU.Input[5] = 0;
	rtObj.rtU.Input1[0] = 0;
	rtObj.rtU.Input1[1] = 0;
	rtObj.rtU.Input1[2] = 0;
	rtObj.rtU.Input1[3] = 0;
	rtObj.rtU.Input1[4] = 0;
	rtObj.rtU.Input1[5] = 0;
	rtObj.rtU.Input[6] = 0;
	rtObj.rtU.Input[7] = 0;
	rtObj.rtU.Input[8] = 0;
	rtObj.rtU.Input[9] = 0;
	rtObj.rtU.Input[10] = 0;
	rtObj.rtU.Input[11] = 0;
	rtObj.rtU.Input1[6] = 0;
	rtObj.rtU.Input1[7] = 0;
	rtObj.rtU.Input1[8] = 0;
	rtObj.rtU.Input1[9] = 0;
	rtObj.rtU.Input1[10] = 0;
	rtObj.rtU.Input1[11] = 0;
*/
  rtObj.rtU.Input3 = 1; 
  rtObj.step();
	cout << rtObj.rtY.Output<< endl;
	cout << rtObj.rtY.Output1<< endl;

  // Get model outputs here

  // Indicate task complete
  OverrunFlag = false;

  // Disable interrupts here
  // Restore FPU context here (if necessary)
  // Enable interrupts here
}

//
// The example "main" function illustrates what is required by your
// application code to initialize, execute, and terminate the generated code.
// Attaching rt_OneStep to a real-time clock is target specific.  This example
// illustrates how you do this relative to initializing the model.
//
int_T main(int_T argc, const char *argv[])
{
  // Unused arguments
  (void)(argc);
  (void)(argv);

  // Initialize model
  rtObj.initialize();

  // Attach rt_OneStep to a timer or interrupt service routine with
  //  period 0.01 seconds (the model's base sample time) here.  The
  //  call syntax for rt_OneStep is
  //
     rt_OneStep();

  printf("Warning: The simulation will run forever. "
         "Generated ERT main won't simulate model step behavior. "
         "To change this behavior select the 'MAT-file logging' option.\n");
  fflush((NULL));
  while (rtmGetErrorStatus(rtObj.getRTM()) == (NULL)) {
    //  Perform other application tasks here
  }

  // Disable rt_OneStep() here
  return 0;
}

//
// File trailer for generated code.
//
// [EOF]
//
