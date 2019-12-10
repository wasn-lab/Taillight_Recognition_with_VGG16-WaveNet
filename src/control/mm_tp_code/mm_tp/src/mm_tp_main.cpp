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
#include "msgs/MMTPInfo.h"
#include <geometry_msgs/PoseStamped.h> //ros
#include <geometry_msgs/PolygonStamped.h>
#include <sensor_msgs/Imu.h>

// #include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>  //tf

#include "msgs/BoxPoint.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/PathPrediction.h"
#include "msgs/PointXY.h"
#include "msgs/PointXYZ.h"
#include "msgs/PointXYZV.h"
#include "msgs/TrackInfo.h" //BB

#include <nav_msgs/OccupancyGrid.h> // costmap
#include <nav_msgs/Path.h>
#include <jsk_recognition_msgs/PolygonArray.h>

#include <stdlib.h>
#include <unistd.h>

#include <stddef.h>
#include <stdio.h>                     // This ert_main.c example uses printf/fflush
#include <iostream>
#include <math.h>
#include "MM_TP.h"                     // Model's header file
#include "rtwtypes.h"

static MM_DPP_1ModelClass rtObj;       // Instance of model class

#define RT_PI 3.14159265358979323846

ros::Publisher mmtp_pub;
ros::Publisher dynamicpath_pub;
ros::Publisher NavPath_Pub;
ros::Publisher NavPath_1_Pub;
ros::Publisher polygon_pub;
ros::Publisher polygon_array_pub;
ros::Publisher FS_occ_pub;
ros::Publisher Localization_UKF_pub;
ros::Publisher mmtpinfo_pub;

int Path_flag, Freespace_mode, VirBB_mode;
float look_ahead_time_straight, look_ahead_time_turn;
int ID_1, ID_2, ID_3, ID_4;
float safe_range, takeover_mag, forward_length_2, J_minvalue_diff_min, J_minvalue_index, OB_enlarge, min_takeoverlength, Delay_length;
float w_l, w_k, w_k_1, w_obs, w_c, w_lane, w_fs, w_off_, w_off_avoid;

struct pose
{
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
};

pose path_pose;

void publishNavPath(double XP_final[], double YP_final[], double XP_final_1[], double YP_final_1[], int index)
{
	nav_msgs::Path Dpath;
	geometry_msgs::PoseStamped Dpose;

	int i,j;

	Dpath.header.frame_id = "map";
	Dpose.header.frame_id = "map";
	for (i=0;i<11;i++)
	{
		float u = i/10.0;
		Dpose.header.seq = i;
		Dpose.pose.position.x = XP_final[0] + XP_final[1]*pow(u,1) + XP_final[2]*pow(u,2) + XP_final[3]*pow(u,3) + XP_final[4]*pow(u,4) + XP_final[5]*pow(u,5);
		Dpose.pose.position.y = YP_final[0] + YP_final[1]*pow(u,1) + YP_final[2]*pow(u,2) + YP_final[3]*pow(u,3) + YP_final[4]*pow(u,4) + YP_final[5]*pow(u,5);
		Dpose.pose.position.z = path_pose.z - 3;
		Dpose.pose.orientation.w = 1.0;
		Dpath.poses.push_back(Dpose);
	}
	for (j=0;j<11;j++)
	{
		float u = j/10.0;
		Dpose.header.seq = i+j;
		Dpose.pose.position.x = XP_final_1[0] + XP_final_1[1]*pow(u,1) + XP_final_1[2]*pow(u,2) + XP_final_1[3]*pow(u,3) + XP_final_1[4]*pow(u,4) + XP_final_1[5]*pow(u,5);
		Dpose.pose.position.y = YP_final_1[0] + YP_final_1[1]*pow(u,1) + YP_final_1[2]*pow(u,2) + YP_final_1[3]*pow(u,3) + YP_final_1[4]*pow(u,4) + YP_final_1[5]*pow(u,5);
		Dpose.pose.position.z = path_pose.z - 3;
		Dpose.pose.orientation.w = 1.0;
		Dpath.poses.push_back(Dpose);
	}
	NavPath_Pub.publish(Dpath);
}

void publishNavPath_1(double XP_final_1[], double YP_final_1[], int index)
{
	nav_msgs::Path Dpath;
	geometry_msgs::PoseStamped Dpose;

	int j;

	Dpath.header.frame_id = "map";
	Dpose.header.frame_id = "map";
	for (j=index-1;j<11;j++)
	{
		float u = j/10.0;
		Dpose.header.seq = j;
		Dpose.pose.position.x = XP_final_1[0] + XP_final_1[1]*pow(u,1) + XP_final_1[2]*pow(u,2) + XP_final_1[3]*pow(u,3) + XP_final_1[4]*pow(u,4) + XP_final_1[5]*pow(u,5);
		Dpose.pose.position.y = YP_final_1[0] + YP_final_1[1]*pow(u,1) + YP_final_1[2]*pow(u,2) + YP_final_1[3]*pow(u,3) + YP_final_1[4]*pow(u,4) + YP_final_1[5]*pow(u,5);
		Dpose.pose.position.z = path_pose.z - 3;
		Dpose.pose.orientation.w = 1.0;
		Dpath.poses.push_back(Dpose);
	}
	NavPath_1_Pub.publish(Dpath);
}

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
	rtObj.rtU.Path_flag = Path_flag;	
	rtObj.rtU.ID_turn[0] = ID_1;
	rtObj.rtU.ID_turn[1] = ID_2;
	rtObj.rtU.ID_turn[2] = ID_3;
	rtObj.rtU.ID_turn[3] = ID_4;
	rtObj.rtU.safe_range = safe_range;
	rtObj.rtU.W_1[0] = w_l;
	rtObj.rtU.W_1[1] = w_k;
	rtObj.rtU.W_1[2] = w_k_1;
	rtObj.rtU.W_1[3] = w_obs;
	rtObj.rtU.W_1[4] = w_c;
	rtObj.rtU.W_1[5] = w_lane;
	rtObj.rtU.W_2[0] = w_obs;
	rtObj.rtU.W_2[1] = w_c;
	rtObj.rtU.W_2[2] = w_lane;
	rtObj.rtU.w_fs = w_fs;
	rtObj.rtU.Freespace_mode = Freespace_mode;
	rtObj.rtU.takeover_mag = takeover_mag;
	rtObj.rtU.forward_length_2 = forward_length_2;
	rtObj.rtU.J_minvalue_diff_min = J_minvalue_diff_min;
	rtObj.rtU.J_minvalue_index = J_minvalue_index;
	rtObj.rtU.VirBB_mode = VirBB_mode;
	rtObj.rtU.w_off_ = w_off_;
	rtObj.rtU.w_off_avoid = w_off_avoid;
	rtObj.rtU.OB_enlarge = OB_enlarge;
	rtObj.rtU.min_takeoverlength = min_takeoverlength;
	rtObj.rtU.Delay_length = Delay_length;

	// Set model inputs here
	std::cout << "-----------------------------------------"  << std::endl;
	// std::cout << "counter : " << rtObj.rtU.SLAM_counter << std::endl;
	std::cout << "Path_flag:" << rtObj.rtU.Path_flag << std::endl;
	std::cout << "Freespace_mode:" << rtObj.rtU.Freespace_mode << std::endl;
	std::cout << "VirBB_mode:" << rtObj.rtU.VirBB_mode << std::endl;
	if (Freespace_mode == 1 || Freespace_mode == 2)
		std::cout << "safe_range:" << rtObj.rtU.safe_range << std::endl;

	// int height = 150;
	// int width = 250;
	// std::cout << "-----------------------------------------"  << std::endl;
	// std::cout << "height = " << height << std::endl; //150
	// std::cout << "width = " << width << std::endl; //250
	// int tmp_freespace = 0;
	// for (int k=-11*15;k<1;k++)
	// 	{
	// 		for (int m=-5;m<6;m++)
	// 		{
	// 			int ooo = rtObj.rtU.Freespace[(width-1-(174+k)) + (height-1-(74+m))*width];
	// 			// std::cout << "Freespace :(" << m << "," << k << ")=" << ooo << std::endl;
	// 			if (ooo > 0)
	// 				tmp_freespace++;
	// 		}
	// 	}
	// std::cout << "tmp_freespace = " << tmp_freespace << std::endl;

	// Step the model
	rtObj.step();

	// Get model outputs here
	// std::cout << "MM_TP running!" << std::endl;
	// std::cout << "SLAM_x:" << rtObj.rtU.SLAM_x << std::endl;
	// std::cout << "X_UKF_SLAM_x:" << rtObj.rtY.X_UKF_SLAM[0] << std::endl;
	// std::cout << "SLAM_y:" << rtObj.rtU.SLAM_y << std::endl;
	// std::cout << "X_UKF_SLAM_y:" << rtObj.rtY.X_UKF_SLAM[1] << std::endl;
	// std::cout << "SLAM_heading:" << rtObj.rtU.SLAM_heading << std::endl;
	// std::cout << "X_UKF_SLAM_heading:" << rtObj.rtY.X_UKF_SLAM[2] << std::endl;
	// std::cout << "Yaw_rate:" << rtObj.rtU.angular_vz << std::endl;
	// std::cout << "Speed_mps:" << rtObj.rtU.Speed_mps << std::endl;
	// std::cout << "rtY_VTY:" << rtObj.rtY.Vehicle_Target_y << std::endl;
	// std::cout << "rtY_XP:" << rtObj.rtY.XP_final[0] << std::endl;
	// std::cout << "rtY_YP:" << rtObj.rtY.YP_final[0] << std::endl;
	// std::cout << "J_minind:" << rtObj.rtY.J_minind << std::endl;
	// std::cout << "J_finalind:" << rtObj.rtY.J_finalind << std::endl;
	// std::cout << "End_x:" << rtObj.rtY.End_x << std::endl;
	// std::cout << "End_y:" << rtObj.rtY.End_y << std::endl;
	// std::cout << "forward length:" << rtObj.rtY.forward_length << std::endl;
	// std::cout << "Look_ahead_time:" << rtObj.rtY.Look_ahead_time << std::endl;
	// for (int i=0;i<13;i++)
	// {
		// std::cout << "U_c[" << i << "]:" << rtObj.rtY.U_c[i] << std::endl;
		// std::cout << "U_c_1[" << i << "]:" << rtObj.rtY.U_c_1[i] << std::endl;
		// std::cout << "safety_level_all[" << i << "]:" << rtObj.rtY.safety_level_all[i] << std::endl;
		// std::cout << "safety_level_all_1[" << i << "]:" << rtObj.rtY.safety_level_all_1[i] << std::endl;
	// }

	// int J_finalindex = rtObj.rtY.J_finalind-1;
	std::cout << "U_c[" << 5 << "]:" << rtObj.rtY.U_c[4] << std::endl;
	std::cout << "U_c_1[" << 5 << "]:" << rtObj.rtY.U_c_1[4] << std::endl;
	std::cout << "safety_level_all[" << "5" << "]:" << rtObj.rtY.safety_level_all[4] << std::endl;
	std::cout << "safety_level_all_1[" << "5" << "]:" << rtObj.rtY.safety_level_all_1[4] << std::endl;
	// std::cout << "J_fsc[" << J_finalindex << "]:" << rtObj.rtY.J_fsc[J_finalindex] << std::endl;
	std::cout << "avoidance_mode:" << rtObj.rtY.avoidance_mode << std::endl;
	
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
	for (int i=0;i<13;i++)
	{
		mmtpmsg.J_fsc[i] = rtObj.rtY.J_fsc[i];
		mmtpmsg.J[i] = rtObj.rtY.J[i];
	}
	mmtpmsg.forward_length_free = rtObj.rtY.forward_length_free;
	mmtpmsg.takeover_length = rtObj.rtY.takeover_length;
	mmtpmsg.avoidance_mode = rtObj.rtY.avoidance_mode;
    
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

	// dynamicpath_pub.publish(dpmsg);

	// Publisher 3
	msgs::LocalizationToVeh ltvmsg;
	ltvmsg.header.frame_id = "map";
	ltvmsg.header.stamp = ros::Time::now();
	ltvmsg.x = rtObj.rtY.X_UKF_SLAM[0];
	ltvmsg.y = rtObj.rtY.X_UKF_SLAM[1];
	ltvmsg.heading = rtObj.rtY.X_UKF_SLAM[2];
	Localization_UKF_pub.publish(ltvmsg);

	publishNavPath(rtObj.rtY.XP_final,rtObj.rtY.YP_final,rtObj.rtY.XP_final_1,rtObj.rtY.YP_final_1,std::floor(rtObj.rtY.takeoverlength_ind));
	publishNavPath_1(rtObj.rtY.XP_final_1,rtObj.rtY.YP_final_1,std::floor(rtObj.rtY.takeoverlength_ind));

	// Publisher 4
	msgs::MMTPInfo mmtpinfomsg;
	mmtpinfomsg.Deadend_flag = rtObj.rtY.Deadend_flag;
	mmtpinfo_pub.publish(mmtpinfomsg);

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
		rtObj.rtU.SLAM_heading = current_pose.yaw + 2*RT_PI;
	else if (current_pose.yaw >= 2*RT_PI)
		rtObj.rtU.SLAM_heading = current_pose.yaw - 2*RT_PI;
	else
		rtObj.rtU.SLAM_heading = current_pose.yaw;
	rtObj.rtU.SLAM_fs = 0;
	rtObj.rtU.SLAM_counter++;
	if (rtObj.rtU.SLAM_counter > 255)
    	rtObj.rtU.SLAM_counter = 0;

    path_pose.z = PSmsg->pose.position.z;
}

void imu_data_callback(const sensor_msgs::Imu::ConstPtr& imumsg)
{
	rtObj.rtU.angular_vz = imumsg->angular_velocity.z;
}

void BB_data_callback(const msgs::DetectedObjectArray::ConstPtr& BBmsg)
{
	jsk_recognition_msgs::PolygonArray polygon_array;
	polygon_array.labels.push_back(3);
	// geometry_msgs::PolygonStamped polygon;
	if (VirBB_mode == 1)
	{
		// polygon.header.frame_id = "map";
		polygon_array.header.frame_id = "map";
	}
	else
	{
		// polygon.header.frame_id = "lidar";
		polygon_array.header.frame_id = "lidar";
	}
	geometry_msgs::Point32 point;

	int size = BBmsg->objects.size();
	std::cout << "Size = " << size  << std::endl;
	rtObj.rtU.BB_num = size;
			
	for (int i =0 ; i<size ; i++)
	{
		geometry_msgs::PolygonStamped polygon;
		if (VirBB_mode == 1)
		{
			polygon.header.frame_id = "map";
		}
		else
		{
			polygon.header.frame_id = "lidar";
		}
		// std::cout << " Object_" << i << ": " << std::endl;
		rtObj.rtU.BB_all_XY[2*i] = (BBmsg->objects[i].bPoint.p0.x);
		rtObj.rtU.BB_all_XY[2*i+1] = (BBmsg->objects[i].bPoint.p0.y);
		rtObj.rtU.BB_all_XY[2*i+100] = (BBmsg->objects[i].bPoint.p3.x);
		rtObj.rtU.BB_all_XY[2*i+101] = (BBmsg->objects[i].bPoint.p3.y);
		rtObj.rtU.BB_all_XY[2*i+200] = (BBmsg->objects[i].bPoint.p7.x);
		rtObj.rtU.BB_all_XY[2*i+201] = (BBmsg->objects[i].bPoint.p7.y);
		rtObj.rtU.BB_all_XY[2*i+300] = (BBmsg->objects[i].bPoint.p4.x);
		rtObj.rtU.BB_all_XY[2*i+301] = (BBmsg->objects[i].bPoint.p4.y);
		// std::cout << "p0: (" << BBmsg->objects[i].bPoint.p0.x << "," << BBmsg->objects[i].bPoint.p0.y << ")" << std::endl;
		// std::cout << "p3: (" << BBmsg->objects[i].bPoint.p3.x << "," << BBmsg->objects[i].bPoint.p3.y << ")" << std::endl;
		// std::cout << "p4: (" << BBmsg->objects[i].bPoint.p4.x << "," << BBmsg->objects[i].bPoint.p4.y << ")" << std::endl;
		// std::cout << "p7: (" << BBmsg->objects[i].bPoint.p7.x << "," << BBmsg->objects[i].bPoint.p7.y << ")" << std::endl;			
		// std::cout << "Relative speed: (" << BBmsg->objects[i].track.relative_velocity.x << "," << BBmsg->objects[i].track.relative_velocity.y << ")" << std::endl;

		// point.x = rtObj.rtU.BB_all_XY[2*i];
		// point.y = rtObj.rtU.BB_all_XY[2*i+1];
		// polygon.polygon.points.push_back(point);
		// point.x = rtObj.rtU.BB_all_XY[2*i+100];
		// point.y = rtObj.rtU.BB_all_XY[2*i+101];
		// polygon.polygon.points.push_back(point);
		// point.x = rtObj.rtU.BB_all_XY[2*i+200];
		// point.y = rtObj.rtU.BB_all_XY[2*i+201];
		// polygon.polygon.points.push_back(point);
		// point.x = rtObj.rtU.BB_all_XY[2*i+300];
		// point.y = rtObj.rtU.BB_all_XY[2*i+301];
		// polygon.polygon.points.push_back(point);

		// enlarge
		double p0x = rtObj.rtU.BB_all_XY[2*i];
		double p0y = rtObj.rtU.BB_all_XY[2*i+1];
		double p3x = rtObj.rtU.BB_all_XY[2*i+100];
		double p3y = rtObj.rtU.BB_all_XY[2*i+101];
		double p7x = rtObj.rtU.BB_all_XY[2*i+200];
		double p7y = rtObj.rtU.BB_all_XY[2*i+201];
		double p4x = rtObj.rtU.BB_all_XY[2*i+300];
		double p4y = rtObj.rtU.BB_all_XY[2*i+301];

		double OB_enlarge_frontbehind = OB_enlarge;
		double OB_enlargescale = OB_enlarge/std::sqrt((p0x-p3x)*(p0x-p3x) + (p0y-p3y)*(p0y-p3y));
		double OB_enlargescale_frontbehind = OB_enlarge_frontbehind/std::sqrt((p0x-p4x)*(p0x-p4x) + (p0y-p4y)*(p0y-p4y));
		point.x = p0x + (p0x-p3x)*OB_enlargescale + (p0x-p4x)*OB_enlargescale_frontbehind;
		point.y = p0y + (p0y-p3y)*OB_enlargescale + (p0y-p4y)*OB_enlargescale_frontbehind;
		polygon.polygon.points.push_back(point);
		point.x = p3x + (p3x-p0x)*OB_enlargescale + (p3x-p7x)*OB_enlargescale_frontbehind;
		point.y = p3y + (p3y-p0y)*OB_enlargescale + (p3y-p7y)*OB_enlargescale_frontbehind;
		polygon.polygon.points.push_back(point);
		point.x = p7x + (p7x-p4x)*OB_enlargescale + (p7x-p3x)*OB_enlargescale_frontbehind;
		point.y = p7y + (p7y-p4y)*OB_enlargescale + (p7y-p3y)*OB_enlargescale_frontbehind;
		polygon.polygon.points.push_back(point);
		point.x = p4x + (p4x-p7x)*OB_enlargescale + (p4x-p0x)*OB_enlargescale_frontbehind;
		point.y = p4y + (p4y-p7y)*OB_enlargescale + (p4y-p0y)*OB_enlargescale_frontbehind;
		polygon.polygon.points.push_back(point);

		polygon_pub.publish(polygon);
		polygon_array.polygons.push_back(polygon);					
	}
	// polygon_pub.publish(polygon);
	polygon_array_pub.publish(polygon_array);
}

void FS_data_callback(const nav_msgs::OccupancyGrid& FSmsg)
{
	nav_msgs::OccupancyGrid costmap_ = FSmsg;
	nav_msgs::OccupancyGrid Freespace_ = FSmsg;
	
	// for (int i=0;i<250;i++)
	// {
	// 	for (int j=0;j<150;j++)
	// 		rtObj.rtU.Freespace[i+j*250];
	// }

	int height = costmap_.info.height;
	int width = costmap_.info.width;

	// cost initialization
	for (int i = 0; i < height; i++) //for (int i = height/2 - 10; i < height/2 + 11; i++) //
	{
		for (int j = 0; j < width; j++)
		{
			// Index of subscribing OccupancyGrid message
			int og_index = i * width + j;
			int cost = costmap_.data[og_index];
			int fs_index = (width-j-1) + (height-i-1) * width;
			double FScost = 0;

			// obstacle or unknown area
			if (cost > 0)
			{
				FScost = 1;
			}
			rtObj.rtU.Freespace[fs_index] = FScost;
			// std::cout << "123132132132132131321 : " << rtObj.rtU.Freespace[fs_index] << std::endl;
		}
	}

	// FS publish
	Freespace_.header.frame_id = "base_link";
	int FS_size = height * width;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int fs_index = (width-j-1) + (height-i-1) * width;
			int fso_index = i * width + j;
			// if (i > height/2 || j > width/2)
			// 	Freespace_.data[fso_index] = 0;
			// else
				Freespace_.data[fso_index] = rtObj.rtU.Freespace[fs_index] * 100;
		}
	}
	FS_occ_pub.publish(Freespace_);

	// std::cout << "-----------------------------------------"  << std::endl;
	// std::cout << "height = " << height << std::endl; //150
	// std::cout << "width = " << width << std::endl; //250
	// int tmp_costmap = 0;
	// int tmp_freespace = 0;
	// for (int k=-11*15;k<10;k++)
	// 	{
	// 		for (int m=-10;m<20;m++)
	// 		{
	// 			int ttt = costmap_.data[(height/2-1+m)*width + (width/2-1+k)];
	// 			int ooo = rtObj.rtU.Freespace[(width-1-(width/2-1+k)) + (height-1-(height/2-1+m))*width];
	// 			// std::cout << "costmap :(" << k << "," << m << ")=" << ttt << std::endl;
	// 			// std::cout << "Freespace :(" << m << "," << k << ")=" << ooo << std::endl;
	// 			if (ttt > 0)
	// 				tmp_costmap++;
	// 			if (ooo > 0)
	// 				tmp_freespace++;
	// 		}
	// 	}
	// std::cout << "---------------------------------tmp_costmap = " << tmp_costmap << std::endl;
	// std::cout << "tmp_freespace = " << tmp_freespace << std::endl;
}

int main(int argc, char **argv)
{
	// Initialize model
	rtObj.initialize();

	//ros
	ros::init(argc, argv, "mm_tp");
	ros::NodeHandle nh;

	Path_flag = 0;
	ros::param::get(ros::this_node::getName()+"/Path_flag", Path_flag);
	Freespace_mode = 0;
	ros::param::get(ros::this_node::getName()+"/Freespace_mode", Freespace_mode);
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
	safe_range = 0.6;
	ros::param::get(ros::this_node::getName()+"/safe_range", safe_range);
	w_l = 0;
	ros::param::get(ros::this_node::getName()+"/w_l", w_l);
	w_k = 0;
	ros::param::get(ros::this_node::getName()+"/w_k", w_k);
	w_k_1 = 0;
	ros::param::get(ros::this_node::getName()+"/w_k_1", w_k_1);
	w_obs = 30;
	ros::param::get(ros::this_node::getName()+"/w_obs", w_obs);
	w_c = 5;
	ros::param::get(ros::this_node::getName()+"/w_c", w_c);
	w_lane = 100;
	ros::param::get(ros::this_node::getName()+"/w_lane", w_lane);
	w_fs = 30;
	ros::param::get(ros::this_node::getName()+"/w_fs", w_fs);
	takeover_mag = 2;
	ros::param::get(ros::this_node::getName()+"/takeover_mag", takeover_mag);
	forward_length_2 = 20;
	ros::param::get(ros::this_node::getName()+"/forward_length_2", forward_length_2);
	J_minvalue_diff_min = 0;
	ros::param::get(ros::this_node::getName()+"/J_minvalue_diff_min", J_minvalue_diff_min);
	J_minvalue_index = 0;
	ros::param::get(ros::this_node::getName()+"/J_minvalue_index", J_minvalue_index);
	VirBB_mode = 0;
	ros::param::get(ros::this_node::getName()+"/VirBB_mode", VirBB_mode);
	w_off_ = 10;
	ros::param::get(ros::this_node::getName()+"/w_off_", w_off_);
	w_off_avoid = 5;
	ros::param::get(ros::this_node::getName()+"/w_off_avoid", w_off_avoid);
	OB_enlarge = 0.3;
	ros::param::get(ros::this_node::getName()+"/OB_enlarge", OB_enlarge);
	min_takeoverlength = 15;
	ros::param::get(ros::this_node::getName()+"/min_takeoverlength", min_takeoverlength);
	Delay_length = 7/3;
	ros::param::get(ros::this_node::getName()+"/Delay_length", Delay_length);
	
	// subscriber
	// Test
	// ros::Subscriber mmtp_sub = nh.subscribe("mmtppubtopic", 1, mmtppubtopicCallback);
	ros::Subscriber virBB_sub = nh.subscribe("abs_virBB_array", 1, BB_data_callback);
	// Main
	ros::Subscriber lidarxyz_sub = nh.subscribe("current_pose", 1, currentposeCallback);
	// ros::Subscriber LTV_sub = nh.subscribe("localization_to_veh", 1, LocalizationToVehCallback);
	ros::Subscriber VI_sub = nh.subscribe("veh_info", 1, VehinfoCallback);
	ros::Subscriber IMU_acc_rpy_sub = nh.subscribe("imu_data", 1, imu_data_callback);
	// ros::Subscriber RelBB_sub = nh.subscribe("PathPredictionOutput/lidar", 1, BB_data_callback);
	ros::Subscriber FS_sub = nh.subscribe("occupancy_grid", 1, FS_data_callback);

	// publisher
	// Main
	mmtp_pub = nh.advertise<mm_tp::MM_TP_msg>("mm_tp_topic", 1);
	// dynamicpath_pub = nh.advertise<msgs::DynamicPath>("dynamic_path_para", 1);
	mmtpinfo_pub = nh.advertise<msgs::MMTPInfo>("mm_tp_info", 1);
	// rviz
	NavPath_Pub = nh.advertise<nav_msgs::Path>("nav_path", 1);
	NavPath_1_Pub = nh.advertise<nav_msgs::Path>("nav_path_1", 1);
	polygon_pub = nh.advertise<geometry_msgs::PolygonStamped>("obs_polygon", 1);
	polygon_array_pub = nh.advertise<jsk_recognition_msgs::PolygonArray>("obs_polygon_array", 1);
	FS_occ_pub = nh.advertise<nav_msgs::OccupancyGrid>("fs_occ_grid", 1);
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