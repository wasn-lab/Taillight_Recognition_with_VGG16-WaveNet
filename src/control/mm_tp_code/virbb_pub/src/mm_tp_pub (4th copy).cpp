#include <iostream>
#include <iomanip>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float64.h"
#include "mm_tp_pub/MM_TP_input_msg.h"
#include "mm_tp_pub/MM_TP_object_msg.h"
#include "mm_tp_pub/MM_TP_AEB_obj_trigger.h"
#include <sstream>

#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <math.h>
#include <geometry_msgs/PolygonStamped.h>
#include "std_msgs/Header.h"
#include <vector>

#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

#define RT_PI 3.14159265358979323846

ros::Publisher mmtp_pub;
ros::Publisher polygon_pub;
ros::Publisher polygon0_pub;
ros::Publisher marker_pub;
ros::Publisher marker_array_pub;

bool AEBObjTrigger;
bool AEBObjTrigger_last = false;

struct object
{
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
    double sx;
    double sy;
    double sz;
};

struct pose
{
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
};

pose current_pose;
pose AEB_obj_pose;

void RelchatterCallback(const geometry_msgs::PoseStamped::ConstPtr& PSmsg)
{
	// tf::Quaternion lidar_q(PSmsg->pose.orientation.x, PSmsg->pose.orientation.y, PSmsg->pose.orientation.z,PSmsg->pose.orientation.w);
	// tf::Matrix3x3 lidar_m(lidar_q);

	// pose current_pose;
	// current_pose.x = PSmsg->pose.position.x;
	// current_pose.y = PSmsg->pose.position.y;
	// current_pose.z = PSmsg->pose.position.z;
	// lidar_m.getRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);

	// double rotate_ang = -current_pose.yaw;
	// double cos_rot = std::cos(rotate_ang);
	// double sin_rot = std::sin(rotate_ang);

	// double P0x = -4.0299;
	// double P0y = -68.0784;
	// double P3x = -2.5986;
	// double P3y = -68.6022;
	// double P4x = -1.6241;
	// double P4y = -63.6022;
	// double P7x = -3.1514;
	// double P7y = -63.5003;

	// geometry_msgs::PolygonStamped polygon0;
	// polygon0.header.frame_id = "map";
	// geometry_msgs::Point32 point0;
	// point0.x = P0x;
	// point0.y = P0y;
	// polygon0.polygon.points.push_back(point0);
	// point0.x = P3x;
	// point0.y = P3y;
	// polygon0.polygon.points.push_back(point0);
	// point0.x = P4x;
	// point0.y = P4y;
	// polygon0.polygon.points.push_back(point0);
	// point0.x = P7x;
	// point0.y = P7y;
	// polygon0.polygon.points.push_back(point0);
	// polygon0_pub.publish(polygon0);


	// int size = 1;
	// std::cout << "Size = " << size  << std::endl;
	// mm_tp_pub::MM_TP_input_msg mmtpmsg;
	// mmtpmsg.BB_num = size;

	// mm_tp_pub::MM_TP_object_msg mmtpobjmsg;
	// // mmtpobjmsg.header.frame_id = "map";
	// mmtpobjmsg.object_p0x = (P0x - current_pose.x) * cos_rot - (P0y - current_pose.y) * sin_rot;
	// mmtpobjmsg.object_p0y = (P0x - current_pose.x) * sin_rot + (P0y - current_pose.y) * cos_rot;
	// mmtpobjmsg.object_p3x = (P3x - current_pose.x) * cos_rot - (P3y - current_pose.y) * sin_rot;
	// mmtpobjmsg.object_p3y = (P3x - current_pose.x) * sin_rot + (P3y - current_pose.y) * cos_rot;
	// mmtpobjmsg.object_p4x = (P4x - current_pose.x) * cos_rot - (P4y - current_pose.y) * sin_rot;
	// mmtpobjmsg.object_p4y = (P4x - current_pose.x) * sin_rot + (P4y - current_pose.y) * cos_rot;
	// mmtpobjmsg.object_p7x = (P7x - current_pose.x) * cos_rot - (P7y - current_pose.y) * sin_rot;
	// mmtpobjmsg.object_p7y = (P7x - current_pose.x) * sin_rot + (P7y - current_pose.y) * cos_rot;

	// std::cout << "p0: (" << mmtpobjmsg.object_p0x << "," << mmtpobjmsg.object_p0y << ")" << std::endl;
	// std::cout << "p3: (" << mmtpobjmsg.object_p3x << "," << mmtpobjmsg.object_p3y << ")" << std::endl;
	// std::cout << "p4: (" << mmtpobjmsg.object_p4x << "," << mmtpobjmsg.object_p4y << ")" << std::endl;
	// std::cout << "p7: (" << mmtpobjmsg.object_p7x << "," << mmtpobjmsg.object_p7y << ")" << std::endl;	

	// // mmtpmsg.header.frame_id = "map";
	// mmtpmsg.BB_all_XY.push_back(mmtpobjmsg);
	// mmtp_pub.publish(mmtpmsg);

	// geometry_msgs::PolygonStamped polygon;
	// polygon.header.frame_id = "lidar";
	// geometry_msgs::Point32 point;
	// point.x = mmtpobjmsg.object_p0x;
	// point.y = mmtpobjmsg.object_p0y;
	// polygon.polygon.points.push_back(point);
	// point.x = mmtpobjmsg.object_p3x;
	// point.y = mmtpobjmsg.object_p3y;
	// polygon.polygon.points.push_back(point);
	// point.x = mmtpobjmsg.object_p4x;
	// point.y = mmtpobjmsg.object_p4y;
	// polygon.polygon.points.push_back(point);
	// point.x = mmtpobjmsg.object_p7x;
	// point.y = mmtpobjmsg.object_p7y;
	// polygon.polygon.points.push_back(point);
	// polygon_pub.publish(polygon);
}

void abs_obs_polygon_static()
{
	// ABS
	double P0x = -4.0299;
	double P0y = -68.0784;
	double P3x = -2.5986;
	double P3y = -68.6022;
	double P4x = -1.6241;
	double P4y = -63.6022;
	double P7x = -3.1514;
	double P7y = -63.5003;

	geometry_msgs::PolygonStamped polygon0;
	polygon0.header.frame_id = "map";
	geometry_msgs::Point32 point0;
	point0.x = P0x;
	point0.y = P0y;
	polygon0.polygon.points.push_back(point0);
	point0.x = P3x;
	point0.y = P3y;
	polygon0.polygon.points.push_back(point0);
	point0.x = P4x;
	point0.y = P4y;
	polygon0.polygon.points.push_back(point0);
	point0.x = P7x;
	point0.y = P7y;
	polygon0.polygon.points.push_back(point0);
	polygon0_pub.publish(polygon0);

	int size = 1;
	std::cout << "Size = " << size  << std::endl;
	mm_tp_pub::MM_TP_input_msg mmtpmsg;
	mmtpmsg.BB_num = size;

	mm_tp_pub::MM_TP_object_msg mmtpobjmsg;
	// mmtpobjmsg.header.frame_id = "map";
	mmtpobjmsg.object_p0x = P0x;
	mmtpobjmsg.object_p0y = P0y;
	mmtpobjmsg.object_p3x = P3x;
	mmtpobjmsg.object_p3y = P3y;
	mmtpobjmsg.object_p4x = P4x;
	mmtpobjmsg.object_p4y = P4y;
	mmtpobjmsg.object_p7x = P7x;
	mmtpobjmsg.object_p7y = P7y;

	std::cout << "p0: (" << mmtpobjmsg.object_p0x << "," << mmtpobjmsg.object_p0y << ")" << std::endl;
	std::cout << "p3: (" << mmtpobjmsg.object_p3x << "," << mmtpobjmsg.object_p3y << ")" << std::endl;
	std::cout << "p4: (" << mmtpobjmsg.object_p4x << "," << mmtpobjmsg.object_p4y << ")" << std::endl;
	std::cout << "p7: (" << mmtpobjmsg.object_p7x << "," << mmtpobjmsg.object_p7y << ")" << std::endl;	

	// mmtpmsg.header.frame_id = "map";
	mmtpmsg.BB_all_XY.push_back(mmtpobjmsg);
	mmtp_pub.publish(mmtpmsg);
}

void emitRow(const std::string type_name, uint32_t type, double x_pos, double y_pos, double z_pos, double x_ori, double y_ori, double z_ori, double w_ori, float r, float g, float b,
    ros::Duration lifetime, ros::Publisher& pub, visualization_msgs::Marker &marker, bool frame_locked = false, std::string frame_id = std::string("map"),
    float sx = 1.0, float sy = 1.0, float sz = 1.0, int id = 0)
{
    // visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = ros::Time::now();
    marker.ns = "marker_" + type_name;
    marker.id = id;
    marker.type = type;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = x_pos;
    marker.pose.position.y = y_pos;
    marker.pose.position.z = z_pos;
    marker.pose.orientation.x = x_ori;
    marker.pose.orientation.y = y_ori;
    marker.pose.orientation.z = z_ori;
    marker.pose.orientation.w = w_ori;
    marker.scale.x = sx;
    marker.scale.y = sy;
    marker.scale.z = sz;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 1.0;

    marker.lifetime = lifetime;
    marker.frame_locked = frame_locked;
    marker.mesh_resource = "package://pr2_description/meshes/base_v0/base.dae";
    pub.publish(marker);
}

void abs_obs_marker_static(object Obs_object, int &size, int id, mm_tp_pub::MM_TP_object_msg &mmtpobjmsg, visualization_msgs::Marker &marker)
{
	tf::Quaternion Obs_q;
	Obs_q.setRPY( Obs_object.roll, Obs_object.pitch, Obs_object.yaw);  // Create this quaternion from roll/pitch/yaw (in radians)
	emitRow("CUBE", visualization_msgs::Marker::CUBE, Obs_object.x, Obs_object.y, Obs_object.z, Obs_q.x(), Obs_q.y(), Obs_q.z(), Obs_q.w(), 0.0, 0.7, 0.3, ros::Duration(), marker_pub, marker,false,"map", Obs_object.sx, Obs_object.sy, Obs_object.sz , id);

	size = 1;
	// std::cout << "Size = " << size  << std::endl;
	// mm_tp_pub::MM_TP_input_msg mmtpmsg;
	// mmtpmsg.BB_num = size;

	// mm_tp_pub::MM_TP_object_msg mmtpobjmsg;
	// mmtpobjmsg.header.frame_id = "map";
	double CY = std::cos(Obs_object.yaw);
	double SY = std::sin(Obs_object.yaw);
	mmtpobjmsg.object_p0x = Obs_object.x + CY*Obs_object.sx/2 - SY*Obs_object.sy/2;
	mmtpobjmsg.object_p0y = Obs_object.y + SY*Obs_object.sx/2 + CY*Obs_object.sy/2;
	mmtpobjmsg.object_p3x = Obs_object.x + CY*(-Obs_object.sx/2) - SY*Obs_object.sy/2;
	mmtpobjmsg.object_p3y = Obs_object.y + SY*(-Obs_object.sx/2) + CY*Obs_object.sy/2;
	mmtpobjmsg.object_p4x = Obs_object.x + CY*(-Obs_object.sx/2) - SY*(-Obs_object.sy/2);
	mmtpobjmsg.object_p4y = Obs_object.y + SY*(-Obs_object.sx/2) + CY*(-Obs_object.sy/2);
	mmtpobjmsg.object_p7x = Obs_object.x + CY*Obs_object.sx/2 - SY*(-Obs_object.sy/2);
	mmtpobjmsg.object_p7y = Obs_object.y + SY*Obs_object.sx/2 + CY*(-Obs_object.sy/2);

	std::cout << "p0: (" << mmtpobjmsg.object_p0x << "," << mmtpobjmsg.object_p0y << ")" << std::endl;
	std::cout << "p3: (" << mmtpobjmsg.object_p3x << "," << mmtpobjmsg.object_p3y << ")" << std::endl;
	std::cout << "p4: (" << mmtpobjmsg.object_p4x << "," << mmtpobjmsg.object_p4y << ")" << std::endl;
	std::cout << "p7: (" << mmtpobjmsg.object_p7x << "," << mmtpobjmsg.object_p7y << ")" << std::endl;	

	// mmtpmsg.BB_all_XY.push_back(mmtpobjmsg);
	// mmtp_pub.publish(mmtpmsg);
}

void abs_obs_marker_dynamic(object Obsdy_object, int index, double velocity, int hz, int &size, int id, mm_tp_pub::MM_TP_object_msg &mmtpobjmsg, visualization_msgs::Marker &marker)
{
	tf::Quaternion Obsdy_q;
	Obsdy_q.setRPY( Obsdy_object.roll, Obsdy_object.pitch, Obsdy_object.yaw);  // Create this quaternion from roll/pitch/yaw (in radians)
	Obsdy_object.x += std::cos(Obsdy_object.yaw + RT_PI/2.0 )*index*velocity/hz;
	Obsdy_object.y += std::sin(Obsdy_object.yaw + RT_PI/2.0 )*index*velocity/hz;
	emitRow("CUBE", visualization_msgs::Marker::CUBE, Obsdy_object.x, Obsdy_object.y, Obsdy_object.z, Obsdy_q.x(), Obsdy_q.y(), Obsdy_q.z(), Obsdy_q.w(), 0.7, 0.0, 0.3, ros::Duration(), marker_pub, marker,false,"map", Obsdy_object.sx, Obsdy_object.sy, Obsdy_object.sz, id);

	size = 1;
	// std::cout << "Size = " << size  << std::endl;
	// mm_tp_pub::MM_TP_input_msg mmtpmsg;
	// mmtpmsg.BB_num = size;

	// mm_tp_pub::MM_TP_object_msg mmtpobjmsg;
	// mmtpobjmsg.header.frame_id = "map";
	double CY = std::cos(Obsdy_object.yaw);
	double SY = std::sin(Obsdy_object.yaw);
	mmtpobjmsg.object_p0x = Obsdy_object.x + CY*Obsdy_object.sx/2 - SY*Obsdy_object.sy/2;
	mmtpobjmsg.object_p0y = Obsdy_object.y + SY*Obsdy_object.sx/2 + CY*Obsdy_object.sy/2;
	mmtpobjmsg.object_p3x = Obsdy_object.x + CY*(-Obsdy_object.sx/2) - SY*Obsdy_object.sy/2;
	mmtpobjmsg.object_p3y = Obsdy_object.y + SY*(-Obsdy_object.sx/2) + CY*Obsdy_object.sy/2;
	mmtpobjmsg.object_p4x = Obsdy_object.x + CY*(-Obsdy_object.sx/2) - SY*(-Obsdy_object.sy/2);
	mmtpobjmsg.object_p4y = Obsdy_object.y + SY*(-Obsdy_object.sx/2) + CY*(-Obsdy_object.sy/2);
	mmtpobjmsg.object_p7x = Obsdy_object.x + CY*Obsdy_object.sx/2 - SY*(-Obsdy_object.sy/2);
	mmtpobjmsg.object_p7y = Obsdy_object.y + SY*Obsdy_object.sx/2 + CY*(-Obsdy_object.sy/2);

	std::cout << "p0: (" << mmtpobjmsg.object_p0x << "," << mmtpobjmsg.object_p0y << ")" << std::endl;
	std::cout << "p3: (" << mmtpobjmsg.object_p3x << "," << mmtpobjmsg.object_p3y << ")" << std::endl;
	std::cout << "p4: (" << mmtpobjmsg.object_p4x << "," << mmtpobjmsg.object_p4y << ")" << std::endl;
	std::cout << "p7: (" << mmtpobjmsg.object_p7x << "," << mmtpobjmsg.object_p7y << ")" << std::endl;	

	// mmtpmsg.BB_all_XY.push_back(mmtpobjmsg);
	// mmtp_pub.publish(mmtpmsg);
}

void CurrentPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& CPmsg)
{
	tf::Quaternion lidar_q(CPmsg->pose.orientation.x, CPmsg->pose.orientation.y, CPmsg->pose.orientation.z, CPmsg->pose.orientation.w);
	tf::Matrix3x3 lidar_m(lidar_q);

	// pose current_pose;
	current_pose.x = CPmsg->pose.position.x;
	current_pose.y = CPmsg->pose.position.y;
	current_pose.z = CPmsg->pose.position.z;
	lidar_m.getRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);
}

void AEBObjTriggerCallback(mm_tp_pub::MM_TP_AEB_obj_trigger AOTmsg)
{
	AEBObjTrigger = AOTmsg.trigger;
}
void aebobjgen(double dist)
{
	AEB_obj_pose = current_pose;
	AEB_obj_pose.x = current_pose.x + std::cos(current_pose.yaw)*dist;
	AEB_obj_pose.y = current_pose.y + std::sin(current_pose.yaw)*dist;
	AEB_obj_pose.yaw = current_pose.yaw - RT_PI/2.0;
}
void aebobj2msg(object Obsaeb_object, bool &pushback_flag, int &size, int id, mm_tp_pub::MM_TP_object_msg &mmtpobjmsg, visualization_msgs::Marker &marker)
{
	tf::Quaternion Obsdy_q;
	Obsdy_q.setRPY(AEB_obj_pose.roll, AEB_obj_pose.pitch, AEB_obj_pose.yaw);  // Create this quaternion from roll/pitch/yaw (in radians)
	emitRow("CUBE", visualization_msgs::Marker::CUBE, AEB_obj_pose.x, AEB_obj_pose.y, AEB_obj_pose.z, Obsdy_q.x(), Obsdy_q.y(), Obsdy_q.z(), Obsdy_q.w(), 0.0, 0.0, 1.0, ros::Duration(0.2), marker_pub, marker,false,"map", Obsaeb_object.sx, Obsaeb_object.sy, Obsaeb_object.sz, id);

	size = 1;
	// std::cout << "Size = " << size  << std::endl;
	// mm_tp_pub::MM_TP_input_msg mmtpmsg;
	// mmtpmsg.BB_num = size;

	// mm_tp_pub::MM_TP_object_msg mmtpobjmsg;
	// mmtpobjmsg.header.frame_id = "map";
	double CY = std::cos(AEB_obj_pose.yaw);
	double SY = std::sin(AEB_obj_pose.yaw);
	std::cout << "CY " << CY << std::endl;
	mmtpobjmsg.object_p0x = AEB_obj_pose.x + CY*Obsaeb_object.sx/2 - SY*Obsaeb_object.sy/2;
	mmtpobjmsg.object_p0y = AEB_obj_pose.y + SY*Obsaeb_object.sx/2 + CY*Obsaeb_object.sy/2;
	mmtpobjmsg.object_p3x = AEB_obj_pose.x + CY*(-Obsaeb_object.sx/2) - SY*Obsaeb_object.sy/2;
	mmtpobjmsg.object_p3y = AEB_obj_pose.y + SY*(-Obsaeb_object.sx/2) + CY*Obsaeb_object.sy/2;
	mmtpobjmsg.object_p4x = AEB_obj_pose.x + CY*(-Obsaeb_object.sx/2) - SY*(-Obsaeb_object.sy/2);
	mmtpobjmsg.object_p4y = AEB_obj_pose.y + SY*(-Obsaeb_object.sx/2) + CY*(-Obsaeb_object.sy/2);
	mmtpobjmsg.object_p7x = AEB_obj_pose.x + CY*Obsaeb_object.sx/2 - SY*(-Obsaeb_object.sy/2);
	mmtpobjmsg.object_p7y = AEB_obj_pose.y + SY*Obsaeb_object.sx/2 + CY*(-Obsaeb_object.sy/2);

	std::cout << "p0: (" << mmtpobjmsg.object_p0x << "," << mmtpobjmsg.object_p0y << ")" << std::endl;
	std::cout << "p3: (" << mmtpobjmsg.object_p3x << "," << mmtpobjmsg.object_p3y << ")" << std::endl;
	std::cout << "p4: (" << mmtpobjmsg.object_p4x << "," << mmtpobjmsg.object_p4y << ")" << std::endl;
	std::cout << "p7: (" << mmtpobjmsg.object_p7x << "," << mmtpobjmsg.object_p7y << ")" << std::endl;	
}
void abs_obs_marker_AEB(object Obsaeb_object, double dist, bool &pushback_flag, int &size, int id, mm_tp_pub::MM_TP_object_msg &mmtpobjmsg, visualization_msgs::Marker &marker)
{
	if (AEBObjTrigger == true && AEBObjTrigger_last == false)
	{
		aebobjgen(dist);
		aebobj2msg(Obsaeb_object, pushback_flag, size, id, mmtpobjmsg, marker);
		pushback_flag = true;
	}
	else if (AEBObjTrigger == true && AEBObjTrigger_last == true)
	{
		aebobj2msg(Obsaeb_object, pushback_flag, size, id, mmtpobjmsg, marker);
		pushback_flag = true;
	}
	else
		pushback_flag = false;

	AEBObjTrigger_last = AEBObjTrigger;
}

void abs_obs_marker(int index, int hz,double velocity, double dist)
{
	// Initial
	int size_s = 0;
	int size_d = 0;
	int size_a = 0;
	int id = 0;
	mm_tp_pub::MM_TP_input_msg mmtpmsg;
	mm_tp_pub::MM_TP_object_msg mmtpobjmsg;
	visualization_msgs::MarkerArray MAmsg;
	visualization_msgs::Marker marker;

	// Static
	// id ++;
	// object Obs1_object;
	// Obs1_object.x = -2.851;
	// Obs1_object.y = -65.85855;
	// Obs1_object.z = 1.0;
	// Obs1_object.roll = 0;
	// Obs1_object.pitch = 0;
	// Obs1_object.yaw = -0.1;
	// Obs1_object.sx = 1.8;
	// Obs1_object.sy = 4.6;
	// Obs1_object.sz = 1.5;
	// abs_obs_marker_static(Obs1_object, size_s, id, mmtpobjmsg, marker);
	// mmtpmsg.BB_all_XY.push_back(mmtpobjmsg);
	// MAmsg.markers.push_back(marker);

	// Dynamic
	id ++;
	object Obsdy1_object;
	Obsdy1_object.x = -2.851;
	Obsdy1_object.y = -65.85855;
	Obsdy1_object.z = 1.0;
	Obsdy1_object.roll = 0;
	Obsdy1_object.pitch = 0;
	Obsdy1_object.yaw = -0.1;
	Obsdy1_object.sx = 1.8;
	Obsdy1_object.sy = 4.6;
	Obsdy1_object.sz = 1.5;
	double velocity1 = velocity; //(m/s)
	abs_obs_marker_dynamic(Obsdy1_object, index, velocity1, hz, size_d, id, mmtpobjmsg, marker);
	mmtpmsg.BB_all_XY.push_back(mmtpobjmsg);
	MAmsg.markers.push_back(marker);

	// AEB obj
	// id ++;
	// object Obsaeb1_object;
	// Obsaeb1_object.sx = 1.8;
	// Obsaeb1_object.sy = 4.6;
	// Obsaeb1_object.sz = 1.5;
	// bool pushback_flag = false;
	// abs_obs_marker_AEB(Obsaeb1_object, dist, pushback_flag, size_a, id, mmtpobjmsg, marker);
	// if (pushback_flag == true)
	// {
	// 	mmtpmsg.BB_all_XY.push_back(mmtpobjmsg);
	// 	MAmsg.markers.push_back(marker);
	// }
	
	// Publish
	mmtpmsg.BB_num = size_s + size_d + size_a;
	std::cout << "Size = " << mmtpmsg.BB_num  << std::endl;
	mmtp_pub.publish(mmtpmsg);
	marker_array_pub.publish(MAmsg);
}

int main( int argc, char **argv )
{
	int index = 0;
	int hz = 100;
	int time_loop = 10; //(s)
	// ros initial
	ros::init(argc, argv, "mm_tp_pub");
	ros::NodeHandle nh;

	// Ini
	double dist = 0;
	if (ros::param::get(ros::this_node::getName()+"/dist", dist));
	double velocity = 0;
	if (ros::param::get(ros::this_node::getName()+"/velocity", velocity));

	// ros::Subscriber object_subscriber = nh.subscribe("current_pose", 1, RelchatterCallback);
	ros::Subscriber current_pose_sub = nh.subscribe("current_pose", 1, CurrentPoseCallback);
	ros::Subscriber AEB_Obj_Trigger_sub = nh.subscribe("aeb_obj_trigger", 1, AEBObjTriggerCallback);

	mmtp_pub = nh.advertise<mm_tp_pub::MM_TP_input_msg>("mmtppubtopic",1);
	// polygon_pub = nh.advertise<geometry_msgs::PolygonStamped>("polygon_topic",1);
	polygon0_pub = nh.advertise<geometry_msgs::PolygonStamped>("polygon0_topic",1);
	marker_pub = nh.advertise<visualization_msgs::Marker>("marker_topic",1);
	marker_array_pub = nh.advertise<visualization_msgs::MarkerArray>("marker_array_topic",1);

	ros::Rate loop_rate(hz);
	while (ros::ok())
	{
		// abs_obs_polygon_static();
		abs_obs_marker(index,hz,velocity,dist);
		ros::spinOnce();
		loop_rate.sleep();

		index ++;
		if (index > time_loop*hz)
			index = 0;
	}
	// ros::spin();
	return 0 ;
} // main