/*
 * GlobalVariable.cpp
 *
 *  Created on: 2017年6月14日
 *      Author: user
 */

#include "GlobalVariable.h"

bool GlobalVariable::UI_ENABLE_OBJECTS = false;
int GlobalVariable::UI_LATERAL_RANGE = 10;
double GlobalVariable::UI_UNIFORM_SAMPLING = 0.1;
double GlobalVariable::UI_DBSCAN_EPS = 1;
int GlobalVariable::UI_DBSCAN_MINPT = 1;
int GlobalVariable::UI_DBSCAN_SELECT = 2;
bool GlobalVariable::UI_ENABLE_TRAINING_TOOL = false;

bool GlobalVariable::UI_ENABLE_LANE = false;
float GlobalVariable::UI_LANE_INTENSITY_I = 7;
float GlobalVariable::UI_LANE_DETECTION_WITH_OUTER = 3;
float GlobalVariable::UI_LANE_DETECTION_WITH_INNER = 0.5;
float GlobalVariable::UI_LANE_DETECTION_LENGTH = 8;
float GlobalVariable::UI_LANE_IDEA_WIDTH = 3.6;

bool GlobalVariable::UI_ENABLE_PARKING = false;
float GlobalVariable::UI_PARKING_INTENSITY_I = 7;
float GlobalVariable::UI_PARKING_SPACE_DEGREE = 90;
float GlobalVariable::UI_PARKING_SLOT_WIDTH = 2.5;
float GlobalVariable::UI_PARKING_SLOT_LENGTH = 6;

string GlobalVariable::UI_UDP_IP = "192.168.8.1";
int GlobalVariable::UI_UDP_Port = 8888;

bool GlobalVariable::UI_TESTING_BUTTOM = false;
double GlobalVariable::UI_PARA[20];

double GlobalVariable::SENSOR_TO_GROUND = 5;  // baby car 0.65 luxgen MVP7 0.723, ITRI bus 2.663
ros::Time GlobalVariable::ROS_TIMESTAMP;

string GlobalVariable::CONFIG_FILE_NAME = "config.ini";
