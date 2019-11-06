/*
 *   File: lidar_location_send.cpp
 *   Created on: ep  2018
 *   Author: Bo Chun Xu
 *	 Institute: ITRI ICL U300
 */


#include <ros/ros.h>
#include "std_msgs/Header.h"

#include "localization_can_class.h"
// #include "lidar_location_send/LocalizationToVeh.h"
#include "msgs/LocalizationToVeh.h"
#include "lidar_location_send/MM_TP_msg.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <iostream>

#include <tf/tf.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Imu.h>

#define CAN_DLC 8;
static ClassLiDARPoseCan CLPC;
static ros::Publisher current_pose_2vm_pub;
static msgs::LocalizationToVeh predict_pose_msg;

static void localization_to_veh_callback(const msgs::LocalizationToVeh::ConstPtr& input)
{
        double fix_heading = input->heading;
        if (fix_heading > 2*M_PI)
        {
                fix_heading -= 2*M_PI;
        }
        std::cout << "call_back_pose_from IPC" << std::endl;
        std::cout << "x::" << input->x <<  "/" << "y::" << input->y <<  "/" << "heading::" << fix_heading <<  "/"
                  << "fitness_score::" << input->fitness_score <<  "/" << "z::" << input->z <<  "/"
                  << "ndt_reliability::" << input->ndt_reliability <<  "/"<< std::endl;
        MsgSendToCan msg_vcu {input->x, input->y, fix_heading,input->fitness_score, input->z, input->ndt_reliability};
        CLPC.poseSendByCAN(msg_vcu);

}

static void localization_to_veh_callback_2(const geometry_msgs::PoseStamped::ConstPtr& input)
{

	    // tf::Quaternion q(
     //    input->pose.orientation.x,
     //    input->pose.orientation.y,
     //    input->pose.orientation.z,
     //    input->pose.orientation.w);
	    // tf::Matrix3x3 m(q);
	    // double roll, pitch, yaw;
	    // m.getRPY(roll, pitch, yaw);
    
     //    double fix_heading = yaw;
     //    if (fix_heading > 2*M_PI)
     //    {
     //            fix_heading -= 2*M_PI;
     //    }
     //    if (fix_heading < 0)
     //    {
     //            fix_heading += 2*M_PI;
     //    }
     //    std::cout << "call_back_pose_from IPC" << std::endl;
     //    std::cout << "x::" << input->pose.position.x <<  "/" << "y::" << input->pose.position.y <<  "/" << "heading::" << fix_heading <<  "/"
     //              << "z::" << input->pose.position.z <<  "/" << std::endl;
     //    MsgSendToCan msg_vcu {input->pose.position.x, input->pose.position.y, fix_heading ,0, input->pose.position.z, 0};
     //    CLPC.poseSendByCAN(msg_vcu);

}

static void imu_data_callback(const sensor_msgs::Imu::ConstPtr& input)
{
		std::cout << "call_back_imu_from IPC" << std::endl;
		std::cout << "acc_x::" << input->linear_acceleration.x << "/" << "acc_y::" << input->linear_acceleration.y << "/" << "zcc_z::" << input->linear_acceleration.z << std::endl;
		std::cout << "angular_x::" << input->angular_velocity.x << "/" << "angular_y::" << input->angular_velocity.y << "/" << "angular_z::" << input->angular_velocity.z << std::endl;
		MsgSendToCan1 msg_vcu1 {input->linear_acceleration.x, input->linear_acceleration.y, input->linear_acceleration.z, input->angular_velocity.x, input->angular_velocity.y, input->angular_velocity.z};
		CLPC.imuSendByCAN(msg_vcu1);
}

static void mmtp_callback(const lidar_location_send::MM_TP_msg::ConstPtr& input)
{
        std::cout << "call_back_mmtp_from IPC" << std::endl;
        std::cout << "VTY::" << input->Vehicle_Target_y << "/" << "UKF_X::" << input->X_UKF_SLAM[0] << "/" << "UKF_Y::" << input->X_UKF_SLAM[1] << std::endl;
        MsgSendToCan2 msg_vcu2 {input->Vehicle_Target_y, input->seg_id_near, input->Target_seg_id, input->Look_ahead_time};
        CLPC.controlSendByCAN(msg_vcu2);
}

int main(int argc, char **argv)
{
        CLPC.initial();
        ros::init(argc, argv, "lidar_location_send");
        ros::NodeHandle nh;
        ros::Subscriber current_pose_2vm_sub = nh.subscribe("localization_to_veh", 1, localization_to_veh_callback);
        // ros::Subscriber current_pose_2vm_sub_2 = nh.subscribe("current_pose", 1, localization_to_veh_callback_2);
        ros::Subscriber IMU_acc_rpy_sub = nh.subscribe("imu_data", 1, imu_data_callback);
        ros::Subscriber mmtp_sub = nh.subscribe("mm_tp_topic", 1, mmtp_callback);

        ros::spin();

        return 0;
}
