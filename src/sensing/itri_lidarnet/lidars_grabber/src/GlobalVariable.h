#ifndef GLOBALVARIABLE_H_
#define GLOBALVARIABLE_H_

#include "all_header.h"

class GlobalVariable
{
  public:

    static double UI_PARA[30];
    static double UI_PARA_BK[30];

    // Hino FineTune Trigger
    static bool Left_FineTune_Trigger;
    static bool Right_FineTune_Trigger;
    static bool Front_FineTune_Trigger;

    // B1 FineTune Trigger
    static bool FrontLeft_FineTune_Trigger;
    static bool FrontRight_FineTune_Trigger;
    static bool RearLeft_FineTune_Trigger;
    static bool RearRight_FineTune_Trigger;

    static size_t STITCHING_MODE_NUM;

};

#endif /* GLOBALVARIABLE_H_ */
