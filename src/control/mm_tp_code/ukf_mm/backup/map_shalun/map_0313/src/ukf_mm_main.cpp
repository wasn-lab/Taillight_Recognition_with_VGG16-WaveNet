#include <ros/ros.h>
#include <ros/package.h>
#include "std_msgs/String.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Header.h"
#include "ukf_mm/UKF_MM_msg.h"
#include "msgs/LocalizationToVeh.h"
#include "msgs/VehInfo.h"
#include <geometry_msgs/PoseStamped.h> //ros
#include <geometry_msgs/PolygonStamped.h>
#include <sensor_msgs/Imu.h>

// #include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>  //tf

#include <stdlib.h>
#include <unistd.h>

#include <stddef.h>
#include <stdio.h>                     // This ert_main.c example uses printf/fflush
#include <iostream>
#include <math.h>
#include "UKF_MM.h"                     // Model's header file
#include "rtwtypes.h"

static UKF_MMModelClass rtObj;       // Instance of model class

#define RT_PI 3.14159265358979323846

ros::Publisher ukfmm_pub;
ros::Publisher Localization_UKF_pub;

float look_ahead_time_straight, look_ahead_time_turn;
int ID_1, ID_2, ID_3, ID_4;

struct pose
{
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
};

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
	// undesign input
	rtObj.rtU.SLAM_fault = 0;
	rtObj.rtU.U_turn = 0;
	rtObj.rtU.Look_ahead_time_straight = look_ahead_time_straight;
	rtObj.rtU.Look_ahead_time_turn = look_ahead_time_turn;	
	rtObj.rtU.ID_turn[0] = ID_1;
	rtObj.rtU.ID_turn[1] = ID_2;
	rtObj.rtU.ID_turn[2] = ID_3;
	rtObj.rtU.ID_turn[3] = ID_4;

	// Step the model
	rtObj.step();
	
	// Publisher 1
	ukf_mm::UKF_MM_msg ukfmmmsg;
	for (int i=0;i<5;i++)
	{
		ukfmmmsg.X_UKF_SLAM[i] = rtObj.rtY.X_UKF_SLAM[i];
	}
	ukfmmmsg.seg_id_near = rtObj.rtY.seg_id_near;
	ukfmmmsg.Target_seg_id = rtObj.rtY.Target_seg_id;
	ukfmmmsg.Look_ahead_time = rtObj.rtY.Look_ahead_time;
    
	ukfmm_pub.publish(ukfmmmsg);

	// Publisher 3
	msgs::LocalizationToVeh ltvmsg;
	ltvmsg.header.frame_id = "map";
	ltvmsg.header.stamp = ros::Time::now();
	ltvmsg.x = rtObj.rtY.X_UKF_SLAM[0];
	ltvmsg.y = rtObj.rtY.X_UKF_SLAM[1];
	ltvmsg.heading = rtObj.rtY.X_UKF_SLAM[2];
	Localization_UKF_pub.publish(ltvmsg);

	// Indicate task complete
	OverrunFlag = false;

	// Disable interrupts here
	// Restore FPU context here (if necessary)
	// Enable interrupts here
}

void LocalizationToVehCallback(const msgs::LocalizationToVeh::ConstPtr& LTVmsg)
{
	rtObj.rtU.SLAM_x = LTVmsg->x;
	rtObj.rtU.SLAM_y = LTVmsg->y;
	rtObj.rtU.SLAM_heading = LTVmsg->heading;
	rtObj.rtU.SLAM_fs = LTVmsg->fitness_score;
	rtObj.rtU.SLAM_counter++;
  if (rtObj.rtU.SLAM_counter > 255)
  {
    rtObj.rtU.SLAM_counter = 0;
  }
}

void VehinfoCallback(const msgs::VehInfo::ConstPtr& VImsg)
{
  	rtObj.rtU.Speed_mps = VImsg->ego_speed;
}

void currentposeCallback(const geometry_msgs::PoseStamped::ConstPtr& PSmsg)
{	
	tf::Quaternion lidar_q(PSmsg->pose.orientation.x, PSmsg->pose.orientation.y, PSmsg->pose.orientation.z,PSmsg->pose.orientation.w);
	tf::Matrix3x3 lidar_m(lidar_q);

	pose current_pose;
	current_pose.x = PSmsg->pose.position.x;
	current_pose.y = PSmsg->pose.position.y;
	current_pose.z = PSmsg->pose.position.z;
	lidar_m.getRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);

	rtObj.rtU.SLAM_x = current_pose.x;
	rtObj.rtU.SLAM_y = current_pose.y;
  if (current_pose.yaw < 0)
  {
    rtObj.rtU.SLAM_heading = current_pose.yaw + 2*RT_PI;
  }
  else if (current_pose.yaw >= 2 * RT_PI)
  {
    rtObj.rtU.SLAM_heading = current_pose.yaw - 2*RT_PI;
  }
  else
  {
    rtObj.rtU.SLAM_heading = current_pose.yaw;
  }
  rtObj.rtU.SLAM_fs = 0;
	rtObj.rtU.SLAM_counter++;
  if (rtObj.rtU.SLAM_counter > 255)
  {
    rtObj.rtU.SLAM_counter = 0;
  }
}

void imu_data_callback(const sensor_msgs::Imu::ConstPtr& imumsg)
{
	rtObj.rtU.angular_vz = imumsg->angular_velocity.z;
}

int main(int argc, char **argv)
{
	// Initialize model
	rtObj.initialize();

	//ros
	ros::init(argc, argv, "ukf_mm");
	ros::NodeHandle nh;

	look_ahead_time_straight = 1.6;
	ros::param::get(ros::this_node::getName()+"/look_ahead_time_straight", look_ahead_time_straight);
	look_ahead_time_turn = 1.1;
	ros::param::get(ros::this_node::getName()+"/look_ahead_time_turn", look_ahead_time_turn);
	ID_1 = 61;
	ros::param::get(ros::this_node::getName()+"/ID_1", ID_1);
	ID_2 = 79;
	ros::param::get(ros::this_node::getName()+"/ID_2", ID_2);
	ID_3 = 83;
	ros::param::get(ros::this_node::getName()+"/ID_3", ID_3);
	ID_4 = 104;
	ros::param::get(ros::this_node::getName()+"/ID_4", ID_4);
	// subscriber
	// Main
	ros::Subscriber lidarxyz_sub = nh.subscribe("current_pose", 1, currentposeCallback);
	// ros::Subscriber LTV_sub = nh.subscribe("localization_to_veh", 1, LocalizationToVehCallback);
	ros::Subscriber VI_sub = nh.subscribe("veh_info", 1, VehinfoCallback);
	ros::Subscriber IMU_acc_rpy_sub = nh.subscribe("imu_data", 1, imu_data_callback);

	// publisher
	// Main
	ukfmm_pub = nh.advertise<ukf_mm::UKF_MM_msg>("ukf_mm_topic", 1);
	// rviz
	Localization_UKF_pub = nh.advertise<msgs::LocalizationToVeh>("localization_ukf", 1);

	ros::Rate loop_rate(100);
	while (ros::ok())
	{	
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