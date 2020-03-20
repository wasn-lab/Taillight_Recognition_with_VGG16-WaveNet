/*
 * GlobalVariable.h
 *
 *  Created on: 2017年6月14日
 *      Author: user
 */

#ifndef GLOBALVARIABLE_H_
#define GLOBALVARIABLE_H_

#include "all_header.h"

class GlobalVariable
{
public:
  static bool UI_ENABLE_OBJECTS;
  static int UI_LATERAL_RANGE;
  static double UI_UNIFORM_SAMPLING;
  static double UI_DBSCAN_EPS;
  static int UI_DBSCAN_MINPT;
  static int UI_DBSCAN_SELECT;
  static bool UI_ENABLE_TRAINING_TOOL;

  static bool UI_ENABLE_LANE;
  static float UI_LANE_INTENSITY_I;
  static float UI_LANE_DETECTION_WITH_OUTER;
  static float UI_LANE_DETECTION_WITH_INNER;
  static float UI_LANE_DETECTION_LENGTH;
  static float UI_LANE_IDEA_WIDTH;

  static bool UI_ENABLE_PARKING;
  static float UI_PARKING_INTENSITY_I;
  static float UI_PARKING_SPACE_DEGREE;
  static float UI_PARKING_SLOT_WIDTH;
  static float UI_PARKING_SLOT_LENGTH;

  static string UI_UDP_IP;
  static int UI_UDP_Port;

  static double UI_PARA[20];

  static double SENSOR_TO_GROUND;
  static ros::Time ROS_TIMESTAMP;

  static string CONFIG_FILE_NAME;
};

#endif /* GLOBALVARIABLE_H_ */
