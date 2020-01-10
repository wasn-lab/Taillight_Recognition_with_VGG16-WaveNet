#include <ros/ros.h>
#include <ros/package.h>
#include "std_msgs/String.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Header.h"
#include "mm_tp/MM_TP_msg.h"
#include "mm_tp/MM_TP_input_msg.h"
#include "msgs/DynamicPath.h"
#include "msgs/LocalizationToVeh.h"
#include "msgs/VehInfo.h"
#include <geometry_msgs/PoseStamped.h> //ros
#include <sensor_msgs/Imu.h>

#include "msgs/BoxPoint.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/PathPrediction.h"
#include "msgs/PointXY.h"
#include "msgs/PointXYZ.h"
#include "msgs/PointXYZV.h"
#include "msgs/TrackInfo.h" //BB

#include <stdlib.h>
#include <unistd.h>

#include <stddef.h>
#include <stdio.h>                     // This ert_main.c example uses printf/fflush
#include <iostream>
#include "MM_TP.h"                     // Model's header file
#include "rtwtypes.h"

static MM_DPP_1ModelClass rtObj;       // Instance of model class

#define RT_PI 3.14159265358979323846

ros::Publisher mmtp_pub;
ros::Publisher dynamicpath_pub;

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
	std::cout << "counter : " << rtObj.rtU.SLAM_counter << std::endl;
	std::cout << "Path_flag:" << rtObj.rtU.Path_flag << std::endl;

	// Step the model
	rtObj.step();

	// Get model outputs here
	std::cout << "SLAM_x:" << rtObj.rtU.SLAM_x << std::endl;
	std::cout << "X_UKF_SLAM_x:" << rtObj.rtY.X_UKF_SLAM[0] << std::endl;
	std::cout << "SLAM_y:" << rtObj.rtU.SLAM_y << std::endl;
	std::cout << "X_UKF_SLAM_y:" << rtObj.rtY.X_UKF_SLAM[1] << std::endl;
	std::cout << "SLAM_heading:" << rtObj.rtU.SLAM_heading << std::endl;
	std::cout << "X_UKF_SLAM_heading:" << rtObj.rtY.X_UKF_SLAM[2] << std::endl;
	std::cout << "Yaw_rate:" << rtObj.rtU.angular_vz << std::endl;
	std::cout << "Speed_mps:" << rtObj.rtU.Speed_mps << std::endl;
	std::cout << "rtY_VTY:" << rtObj.rtY.Vehicle_Target_y << std::endl;
	std::cout << "rtY_XP:" << rtObj.rtY.XP_final[0] << std::endl;
	std::cout << "rtY_YP:" << rtObj.rtY.YP_final[0] << std::endl;
	std::cout << "J_minind:" << rtObj.rtY.J_minind << std::endl;
	std::cout << "J_finalind:" << rtObj.rtY.J_finalind << std::endl;
	std::cout << "End_x:" << rtObj.rtY.End_x << std::endl;
	std::cout << "End_y:" << rtObj.rtY.End_y << std::endl;
	std::cout << "forward length:" << rtObj.rtY.forward_length << std::endl;

	// Publisher 1
	mm_tp::MM_TP_msg mmtpmsg;
	mmtpmsg.Vehicle_Target_x = rtObj.rtY.Vehicle_Target_x;
    mmtpmsg.Vehicle_Target_y = rtObj.rtY.Vehicle_Target_y;
    for (int i=0;i<6;i++)
    {
	      mmtpmsg.XP_final[i] = rtObj.rtY.XP_final[i];
	      mmtpmsg.YP_final[i] = rtObj.rtY.YP_final[i];
	      mmtpmsg.XP_final_1[i] = rtObj.rtY.XP_final_1[i];
	      mmtpmsg.YP_final_1[i] = rtObj.rtY.YP_final_1[i];
    }
    for (int i=0;i<5;i++)
    {
    	mmtpmsg.X_UKF_SLAM[i] = rtObj.rtY.X_UKF_SLAM[i];
    }
    mmtpmsg.J_minind = rtObj.rtY.J_minind;
    mmtpmsg.J_finalind = rtObj.rtY.J_finalind;
	mmtpmsg.End_x = rtObj.rtY.End_x;
	mmtpmsg.End_y = rtObj.rtY.End_y;
	mmtpmsg.forward_length = rtObj.rtY.forward_length;
	mmtpmsg.seg_id_near = rtObj.rtY.seg_id_near;
  	mmtpmsg.Target_seg_id = rtObj.rtY.Target_seg_id;
  	mmtpmsg.Look_ahead_time = rtObj.rtY.Look_ahead_time; 
      
    mmtp_pub.publish(mmtpmsg);
    // std::cout << "rtY_XP_pub:" << mmtpmsg.XP_final[0] << std::endl;
    // std::cout << "rtY_YP_pub:" << mmtpmsg.YP_final[0] << std::endl;

    // Publisher 2
    msgs::DynamicPath dpmsg;
    dpmsg.XP1_0 = rtObj.rtY.XP_final[0];
    dpmsg.XP1_1 = rtObj.rtY.XP_final[1];
    dpmsg.XP1_2 = rtObj.rtY.XP_final[2];
    dpmsg.XP1_3 = rtObj.rtY.XP_final[3];
    dpmsg.XP1_4 = rtObj.rtY.XP_final[4];
    dpmsg.XP1_5 = rtObj.rtY.XP_final[5];

    dpmsg.YP1_0 = rtObj.rtY.YP_final[0];
    dpmsg.YP1_1 = rtObj.rtY.YP_final[1];
    dpmsg.YP1_2 = rtObj.rtY.YP_final[2];
    dpmsg.YP1_3 = rtObj.rtY.YP_final[3];
    dpmsg.YP1_4 = rtObj.rtY.YP_final[4];
    dpmsg.YP1_5 = rtObj.rtY.YP_final[5];

    dpmsg.XP2_0 = rtObj.rtY.XP_final_1[0];
    dpmsg.XP2_1 = rtObj.rtY.XP_final_1[1];
    dpmsg.XP2_2 = rtObj.rtY.XP_final_1[2];
    dpmsg.XP2_3 = rtObj.rtY.XP_final_1[3];
    dpmsg.XP2_4 = rtObj.rtY.XP_final_1[4];
    dpmsg.XP2_5 = rtObj.rtY.XP_final_1[5];

    dpmsg.YP2_0 = rtObj.rtY.YP_final_1[0];
    dpmsg.YP2_1 = rtObj.rtY.YP_final_1[1];
    dpmsg.YP2_2 = rtObj.rtY.YP_final_1[2];
    dpmsg.YP2_3 = rtObj.rtY.YP_final_1[3];
    dpmsg.YP2_4 = rtObj.rtY.YP_final_1[4];
    dpmsg.YP2_5 = rtObj.rtY.YP_final_1[5];

    dynamicpath_pub.publish(dpmsg);
    // std::cout << "rtY_XP10_pub:" << dpmsg.XP1_0 << std::endl;
    // std::cout << "rtY_YP10_pub:" << dpmsg.YP1_0 << std::endl;

	// Indicate task complete
	OverrunFlag = false;

	// Disable interrupts here
	// Restore FPU context here (if necessary)
	// Enable interrupts here
}

void mmtppubtopicCallback(const mm_tp::MM_TP_input_msg::ConstPtr& mmtpinputmsg)
{
  // rtObj.rtU.SLAM_x = mmtpinputmsg->SLAM_x;
  // rtObj.rtU.SLAM_y = mmtpinputmsg->SLAM_y;
  // rtObj.rtU.SLAM_heading = mmtpinputmsg->SLAM_heading;
  // rtObj.rtU.angular_vz = mmtpinputmsg->angular_vz;
  // rtObj.rtU.Speed_mps = mmtpinputmsg->Speed_mps;
  // rtObj.rtU.SLAM_counter = mmtpinputmsg->SLAM_counter;
  // rtObj.rtU.SLAM_fs = mmtpinputmsg->SLAM_fs;
  // rtObj.rtU.SLAM_fault = mmtpinputmsg->SLAM_fault;
  // rtObj.rtU.U_turn = mmtpinputmsg->U_turn;
  // rtObj.rtU.BB_num = mmtpinputmsg->BB_num;
  // rtObj.rtU.Look_ahead_time = mmtpinputmsg->Look_ahead_time;
}

void LocalizationToVehCallback(const msgs::LocalizationToVeh::ConstPtr& LTVmsg)
{
	rtObj.rtU.SLAM_x = LTVmsg->x;
	rtObj.rtU.SLAM_y = LTVmsg->y;
	rtObj.rtU.SLAM_heading = LTVmsg->heading;
	rtObj.rtU.SLAM_fs = LTVmsg->fitness_score;
	rtObj.rtU.SLAM_counter++;
	if (rtObj.rtU.SLAM_counter > 255)
		rtObj.rtU.SLAM_counter = 0;
  
}

void VehinfoCallback(const msgs::VehInfo::ConstPtr& VImsg)
{
  	rtObj.rtU.Speed_mps = VImsg->ego_speed;
}

void currentposeCallback(const geometry_msgs::PoseStamped::ConstPtr& PSmsg)
{
	// rtObj.rtU.SLAM_x = PSmsg->pose.position.x;
	// rtObj.rtU.SLAM_y = PSmsg->pose.position.y;
	// rtObj.rtU.SLAM_counter++;
	// if (rtObj.rtU.SLAM_counter > 255)
 //    	rtObj.rtU.SLAM_counter = 0;
}

void imu_data_callback(const sensor_msgs::Imu::ConstPtr& imumsg)
{
	rtObj.rtU.angular_vz = (imumsg->angular_velocity.z)*(-RT_PI/180);
}

void BB_data_callback(const msgs::DetectedObjectArray::ConstPtr& BBmsg)
{
	int size = BBmsg->objects.size();
	std::cout << "Size = " << size  << std::endl;

	rtObj.rtU.BB_num = size;
			
	for (int i =0 ; i<size ; i++)
	{
		std::cout << " Object_" << i << ": " << std::endl;
		rtObj.rtU.BB_all_XY[2*i] = (BBmsg->objects[i].bPoint.p0.x);
		rtObj.rtU.BB_all_XY[2*i+1] = (BBmsg->objects[i].bPoint.p0.y);
		rtObj.rtU.BB_all_XY[2*i+100] = (BBmsg->objects[i].bPoint.p3.x);
		rtObj.rtU.BB_all_XY[2*i+101] = (BBmsg->objects[i].bPoint.p3.y);
		rtObj.rtU.BB_all_XY[2*i+200] = (BBmsg->objects[i].bPoint.p4.x);
		rtObj.rtU.BB_all_XY[2*i+201] = (BBmsg->objects[i].bPoint.p4.y);
		rtObj.rtU.BB_all_XY[2*i+300] = (BBmsg->objects[i].bPoint.p7.x);
		rtObj.rtU.BB_all_XY[2*i+301] = (BBmsg->objects[i].bPoint.p7.y);
		std::cout << "p0: (" << BBmsg->objects[i].bPoint.p0.x << "," << BBmsg->objects[i].bPoint.p0.y << ")" << std::endl;
		std::cout << "p3: (" << BBmsg->objects[i].bPoint.p3.x << "," << BBmsg->objects[i].bPoint.p3.y << ")" << std::endl;
		std::cout << "p4: (" << BBmsg->objects[i].bPoint.p4.x << "," << BBmsg->objects[i].bPoint.p4.y << ")" << std::endl;
		std::cout << "p7: (" << BBmsg->objects[i].bPoint.p7.x << "," << BBmsg->objects[i].bPoint.p7.y << ")" << std::endl;			
		std::cout << "Relative speed: (" << BBmsg->objects[i].track.relative_velocity.x << "," << BBmsg->objects[i].track.relative_velocity.y << ")" << std::endl;					
	}
}

int main(int argc, char **argv)
{
	// Initialize model
	rtObj.initialize();

	//ros
	ros::init(argc, argv, "mm_tp");
	ros::NodeHandle nh;

	int Path_flag = 0;
  	if (ros::param::get(ros::this_node::getName()+"/Path_flag", Path_flag));
  	float look_ahead_time_straight = 1.6;
  	if (ros::param::get(ros::this_node::getName()+"/look_ahead_time_straight", look_ahead_time_straight));
  	float look_ahead_time_turn = 1.6;
  	if (ros::param::get(ros::this_node::getName()+"/look_ahead_time_turn", look_ahead_time_turn));
  	int ID_1 = 61;
  	if (ros::param::get(ros::this_node::getName()+"/ID_1", ID_1));
  	int ID_2 = 79;
  	if (ros::param::get(ros::this_node::getName()+"/ID_2", ID_2));
  	int ID_3 = 83;
  	if (ros::param::get(ros::this_node::getName()+"/ID_3", ID_3));
  	int ID_4 = 104;
  	if (ros::param::get(ros::this_node::getName()+"/ID_4", ID_4));

	// subscriber
	// Test
  	// ros::Subscriber mmtp_sub = nh.subscribe("mmtppubtopic", 1, mmtppubtopicCallback);
  	// Main
  	// ros::Subscriber lidarxyz_sub = nh.subscribe("current_pose", 1, currentposeCallback);
  	ros::Subscriber LTV_sub = nh.subscribe("localization_to_veh", 1, LocalizationToVehCallback);
  	ros::Subscriber VI_sub = nh.subscribe("veh_info", 1, VehinfoCallback);
  	ros::Subscriber IMU_acc_rpy_sub = nh.subscribe("imu_data", 1, imu_data_callback);
  	ros::Subscriber RelBB_sub = nh.subscribe("PathPredictionOutput/lidar", 1, BB_data_callback);

  	// publisher
  	mmtp_pub = nh.advertise<mm_tp::MM_TP_msg>("mm_tp_topic", 1);
  	dynamicpath_pub = nh.advertise<msgs::DynamicPath>("dynamic_path_para", 1);

  	ros::Rate loop_rate(100);
  	while (ros::ok())
  	{
		// undesign input
		rtObj.rtU.SLAM_fault = 0;
		rtObj.rtU.U_turn = 0;
		rtObj.rtU.Look_ahead_time_straight = look_ahead_time_straight;
		rtObj.rtU.Look_ahead_time_turn = look_ahead_time_turn;
		rtObj.rtU.Path_flag = Path_flag;
		rtObj.rtU.ID_turn[0] = ID_1;
		rtObj.rtU.ID_turn[1] = ID_2;
		rtObj.rtU.ID_turn[2] = ID_3;
		rtObj.rtU.ID_turn[3] = ID_4;
		std::cout << "Look_ahead_time:" << rtObj.rtY.Look_ahead_time << std::endl;
    	rt_OneStep();
	    ros::spinOnce();
	    loop_rate.sleep();
	}
	// ros::spin();
  

  // printf("Warning: The simulation will run forever. "
  //        "Generated ERT main won't simulate model step behavior. "
  //        "To change this behavior select the 'MAT-file logging' option.\n");
  // fflush((NULL));
  // while (rtmGetErrorStatus(rtObj.getRTM()) == (NULL)) {
  //   //  Perform other application tasks here
  // }

  // Disable rt_OneStep() here
  	return 0;
}