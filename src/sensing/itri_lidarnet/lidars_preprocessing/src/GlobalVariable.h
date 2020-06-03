
#ifndef GLOBALVARIABLE_H_
#define GLOBALVARIABLE_H_

#include "all_header.h"

class GlobalVariable
{
public:
  static double UI_PARA[20];
  static double UI_PARA_BK[20];

  static ros::Time ROS_TIMESTAMP;

  static string CONFIG_FILE_NAME;

  static int ERROR_CODE;

  static bool ENABLE_LABEL_TOOL;
};

#endif /* GLOBALVARIABLE_H_ */
