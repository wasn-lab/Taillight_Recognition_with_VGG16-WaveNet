/*
 *   File: localization_can_class.h
 *   Created on: April, 2018
 *   Author: Xu, Bo Chun & Wayne Yang
 *   Institute: ITRI ICL U300
 */

#ifndef LOCALIZATION_CAN_CLASS_H_
#define LOCALIZATION_CAN_CLASS_H_

#include <stdio.h>
#include <net/if.h> //struct ifreq
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <linux/can/error.h>
#include <iostream>
#include <fstream>


struct MsgSendToCan
{
        double x;
        double y;
        double heading;
        double fitness_score;
        double transform_prob;
        double ndt_reliability;
};

using namespace std;

class ClassLiDARPoseCan
{
private:

int s;
int required_mtu;
int can_counter = 0;
bool lidar_error = false;
public:

ClassLiDARPoseCan ();

~ClassLiDARPoseCan ();

void
initial ();

int
poseSendByCAN(const struct MsgSendToCan &input_msg);



};

#endif /*LOCALIZATION_CAN_CLASS_H_*/
