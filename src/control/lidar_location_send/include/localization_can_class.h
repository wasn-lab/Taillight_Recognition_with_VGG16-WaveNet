/*
 *   File: localization_can_class.h
 *   Created on: April, 2018
 *   Author: Xu, Bo Chun & Wayne Yang
 *   Institute: ITRI ICL U300
 */

#ifndef LOCALIZATION_CAN_CLASS_H_
#define LOCALIZATION_CAN_CLASS_H_

#include <cstdio>
#include <net/if.h> //struct ifreq
#include <cstdlib>
#include <unistd.h>
#include <cstring>
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
    double z;
    double ndt_reliability;
};

struct MsgSendToCan1
{
    double linear_accx;
    double linear_accy;
    double linear_accz;
    double angular_vx;
    double angular_vy;
    double angular_vz;
};
struct MsgSendToCan2
{
    double vehicle_target_y;
    int seg_id_near;
    int Target_seg_id;
    double Look_ahead_time;
};
struct MsgSendToCan3
{
    double front_vehicle_target_y;
    double rear_vehicle_target_y;
};

using namespace std;

class ClassLiDARPoseCan
{
private :

    int s;
    int required_mtu;
    int can_counter = 0;
    int can_counter1 = 0;
    bool lidar_error = false;
public :

    ClassLiDARPoseCan ();

    ~ClassLiDARPoseCan ();

    void
    initial ();

    int
    poseSendByCAN(const struct MsgSendToCan &input_msg);

    int
    imuSendByCAN(const struct MsgSendToCan1 &input_msg);

    int
    controlSendByCAN(const struct MsgSendToCan2 &input_msg);

    int
    controlSendByCAN_1(const struct MsgSendToCan3 &input_msg);

};

#endif /*LOCALIZATION_CAN_CLASS_H_*/
