#include <iostream>
#include <iomanip>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float64.h"
#include "virbb_pub/MM_TP_input_msg.h"
#include "virbb_pub/virbb_pub_obj_trigger.h"
#include <sstream>
#include <ros/package.h>
#include <string.h>
#include <fstream>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <tf/tf.h>
#include <math.h>
#include <geometry_msgs/PolygonStamped.h>
#include "std_msgs/Header.h"
#include <vector>

#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

#include "msgs/BoxPoint.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/PointXYZ.h"
#include "msgs/PointXY.h"  
#include "msgs/TrackInfo.h" 
#include "msgs/PathPrediction.h"//BB

#include "msgs/LocalizationToVeh.h"

#define RT_PI 3.14159265358979323846

// ros::Publisher mmtp_pub;
ros::Publisher polygon_pub;
ros::Publisher polygon0_pub;
ros::Publisher abs_marker_pub;
ros::Publisher abs_marker_array_pub;
ros::Publisher abs_BB_array_pub;
ros::Publisher rel_BB_array_pub;
ros::Publisher abs_PPmarker_pub;
ros::Publisher abs_PPmarker_array_pub;

bool AEBObjTrigger;
bool AEBObjTrigger_last = false;
bool StaticObjTrigger = false;
bool DynamicObjTrigger = false;
bool rvizPubPointObjTrigger = false;
std::string rvizPublisherPoint_frame_id;

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

object Obs1_object;
object Obsdy1_object;
object Obsaeb1_object;

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
pose Localization_ukf_pose;
pose rvizPublisherPoint;

double read_tmp_s[9] = {};
double read_tmp_d[11] = {};
double read_tmp_a[4] = {};
double read_tmp_r[7] = {};
int dynamicobj_index;
int rvizppobj_index = 0;

double ppdt;
double ppft;

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


// void abs_obs_polygon_static()
// {
// 	// ABS
// 	double P0x = -4.0299;
// 	double P0y = -68.0784;
// 	double P3x = -2.5986;
// 	double P3y = -68.6022;
// 	double P7x = -1.6241;
// 	double P7y = -63.6022;
// 	double P4x = -3.1514;
// 	double P4y = -63.5003;

// 	geometry_msgs::PolygonStamped polygon0;
// 	polygon0.header.frame_id = "map";
// 	geometry_msgs::Point32 point0;
// 	point0.x = P0x;
// 	point0.y = P0y;
// 	polygon0.polygon.points.push_back(point0);
// 	point0.x = P3x;
// 	point0.y = P3y;
// 	polygon0.polygon.points.push_back(point0);
// 	point0.x = P7x;
// 	point0.y = P7y;
// 	polygon0.polygon.points.push_back(point0);
// 	point0.x = P4x;
// 	point0.y = P4y;
// 	polygon0.polygon.points.push_back(point0);
// 	polygon0_pub.publish(polygon0);
// }

void LocalizationUKFCallback(const msgs::LocalizationToVeh::ConstPtr& LUmsg)
{
	Localization_ukf_pose.x = LUmsg->x;
	Localization_ukf_pose.y = LUmsg->y;
	Localization_ukf_pose.yaw = LUmsg->heading;
}

void rvizPublisherPointCallback(const geometry_msgs::PointStamped::ConstPtr& rvizPmsg)
{
	rvizPublisherPoint_frame_id = rvizPmsg->header.frame_id;
	if (rvizPublisherPoint_frame_id == "map")
	{
		rvizPublisherPoint.x = rvizPmsg->point.x;
		rvizPublisherPoint.y = rvizPmsg->point.y;
		rvizPublisherPoint.z = rvizPmsg->point.z;	
	}
	else
	{
		double cy = std::cos(current_pose.yaw);
		double sy = std::sin(current_pose.yaw);
		rvizPublisherPoint.x = rvizPmsg->point.x*cy - rvizPmsg->point.y*sy + current_pose.x;
		rvizPublisherPoint.y = rvizPmsg->point.x*sy + rvizPmsg->point.y*cy + current_pose.y;
		rvizPublisherPoint.z = rvizPmsg->point.z + current_pose.z;
	}
	rvizPublisherPoint.roll = current_pose.roll;
	rvizPublisherPoint.pitch = current_pose.pitch;
	rvizPublisherPoint.yaw = current_pose.yaw;
	rvizPubPointObjTrigger = true;
	rvizppobj_index = 0;
}

void abs2rel(msgs::DetectedObject &abs_BBmsg, msgs::DetectedObject &rel_BBmsg)
{
	double rotate_ang = -Localization_ukf_pose.yaw;
	double cos_rot = std::cos(rotate_ang);
	double sin_rot = std::sin(rotate_ang);

	rel_BBmsg.header.frame_id = "lidar";
    rel_BBmsg.header.stamp = ros::Time::now();
	rel_BBmsg.bPoint.p0.x = (abs_BBmsg.bPoint.p0.x - Localization_ukf_pose.x) * cos_rot - (abs_BBmsg.bPoint.p0.y - Localization_ukf_pose.y) * sin_rot;
	rel_BBmsg.bPoint.p0.y = (abs_BBmsg.bPoint.p0.x - Localization_ukf_pose.x) * sin_rot + (abs_BBmsg.bPoint.p0.y - Localization_ukf_pose.y) * cos_rot;
	rel_BBmsg.bPoint.p0.z = abs_BBmsg.bPoint.p0.z;
	rel_BBmsg.bPoint.p3.x = (abs_BBmsg.bPoint.p3.x - Localization_ukf_pose.x) * cos_rot - (abs_BBmsg.bPoint.p3.y - Localization_ukf_pose.y) * sin_rot;
	rel_BBmsg.bPoint.p3.y = (abs_BBmsg.bPoint.p3.x - Localization_ukf_pose.x) * sin_rot + (abs_BBmsg.bPoint.p3.y - Localization_ukf_pose.y) * cos_rot;
	rel_BBmsg.bPoint.p3.z = abs_BBmsg.bPoint.p3.z;
	rel_BBmsg.bPoint.p7.x = (abs_BBmsg.bPoint.p7.x - Localization_ukf_pose.x) * cos_rot - (abs_BBmsg.bPoint.p7.y - Localization_ukf_pose.y) * sin_rot;
	rel_BBmsg.bPoint.p7.y = (abs_BBmsg.bPoint.p7.x - Localization_ukf_pose.x) * sin_rot + (abs_BBmsg.bPoint.p7.y - Localization_ukf_pose.y) * cos_rot;
	rel_BBmsg.bPoint.p7.z = abs_BBmsg.bPoint.p7.z;
	rel_BBmsg.bPoint.p4.x = (abs_BBmsg.bPoint.p4.x - Localization_ukf_pose.x) * cos_rot - (abs_BBmsg.bPoint.p4.y - Localization_ukf_pose.y) * sin_rot;
	rel_BBmsg.bPoint.p4.y = (abs_BBmsg.bPoint.p4.x - Localization_ukf_pose.x) * sin_rot + (abs_BBmsg.bPoint.p4.y - Localization_ukf_pose.y) * cos_rot;
	rel_BBmsg.bPoint.p4.z = abs_BBmsg.bPoint.p4.z;
	rel_BBmsg.bPoint.p1.x = rel_BBmsg.bPoint.p0.x;
	rel_BBmsg.bPoint.p1.y = rel_BBmsg.bPoint.p0.y;
	rel_BBmsg.bPoint.p1.z = abs_BBmsg.bPoint.p1.z;
	rel_BBmsg.bPoint.p2.x = rel_BBmsg.bPoint.p3.x;
	rel_BBmsg.bPoint.p2.y = rel_BBmsg.bPoint.p3.y;
	rel_BBmsg.bPoint.p2.z = abs_BBmsg.bPoint.p2.z;
	rel_BBmsg.bPoint.p6.x = rel_BBmsg.bPoint.p7.x;
	rel_BBmsg.bPoint.p6.y = rel_BBmsg.bPoint.p7.y;
	rel_BBmsg.bPoint.p6.z = abs_BBmsg.bPoint.p6.z;
	rel_BBmsg.bPoint.p5.x = rel_BBmsg.bPoint.p4.x;
	rel_BBmsg.bPoint.p5.y = rel_BBmsg.bPoint.p4.y;
	rel_BBmsg.bPoint.p5.z = abs_BBmsg.bPoint.p5.z;
}

void emitRow(const std::string type_name, uint32_t type, double x_pos, double y_pos, double z_pos, double x_ori, double y_ori, double z_ori, double w_ori, float r, float g, float b,
    ros::Duration lifetime, ros::Publisher& pub, visualization_msgs::Marker &marker, bool frame_locked = false, std::string frame_id = std::string("map"),
    float sx = 1.0, float sy = 1.0, float sz = 1.0, int id = 0)
{
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

void abs_obs_marker_static(object Obs_object, int id, msgs::DetectedObject &BBmsg, visualization_msgs::Marker &marker)
{
	tf::Quaternion Obs_q;
	Obs_q.setRPY( Obs_object.roll, Obs_object.pitch, Obs_object.yaw);  // Create this quaternion from roll/pitch/yaw (in radians)
	emitRow("CUBE", visualization_msgs::Marker::CUBE, Obs_object.x, Obs_object.y, Obs_object.z, Obs_q.x(), Obs_q.y(), Obs_q.z(), Obs_q.w(), 0.0, 0.7, 0.3, ros::Duration(0.2), abs_marker_pub, marker,false,"map", Obs_object.sx, Obs_object.sy, Obs_object.sz , id);

	double CY = std::cos(Obs_object.yaw);
	double SY = std::sin(Obs_object.yaw);

	BBmsg.header.frame_id = "lidar";
    BBmsg.header.stamp = ros::Time::now();
	BBmsg.bPoint.p0.x = Obs_object.x + CY*(-Obs_object.sx/2) - SY*Obs_object.sy/2;
	BBmsg.bPoint.p0.y = Obs_object.y + SY*(-Obs_object.sx/2) + CY*Obs_object.sy/2;
	BBmsg.bPoint.p0.z = -3.0;
	BBmsg.bPoint.p3.x = Obs_object.x + CY*(-Obs_object.sx/2) - SY*(-Obs_object.sy)/2;
	BBmsg.bPoint.p3.y = Obs_object.y + SY*(-Obs_object.sx/2) + CY*(-Obs_object.sy)/2;
	BBmsg.bPoint.p3.z = -3.0;
	BBmsg.bPoint.p7.x = Obs_object.x + CY*Obs_object.sx/2 - SY*(-Obs_object.sy/2);
	BBmsg.bPoint.p7.y = Obs_object.y + SY*Obs_object.sx/2 + CY*(-Obs_object.sy/2);
	BBmsg.bPoint.p7.z = -3.0;
	BBmsg.bPoint.p4.x = Obs_object.x + CY*Obs_object.sx/2 - SY*Obs_object.sy/2;
	BBmsg.bPoint.p4.y = Obs_object.y + SY*Obs_object.sx/2 + CY*Obs_object.sy/2;
	BBmsg.bPoint.p4.z = -3.0;
	BBmsg.bPoint.p1.x = BBmsg.bPoint.p0.x;
	BBmsg.bPoint.p1.y = BBmsg.bPoint.p0.y;
	BBmsg.bPoint.p1.z = -1.0;
	BBmsg.bPoint.p2.x = BBmsg.bPoint.p3.x;
	BBmsg.bPoint.p2.y = BBmsg.bPoint.p3.y;
	BBmsg.bPoint.p2.z = -1.0;
	BBmsg.bPoint.p6.x = BBmsg.bPoint.p7.x;
	BBmsg.bPoint.p6.y = BBmsg.bPoint.p7.y;
	BBmsg.bPoint.p6.z = -1.0;
	BBmsg.bPoint.p5.x = BBmsg.bPoint.p4.x;
	BBmsg.bPoint.p5.y = BBmsg.bPoint.p4.y;
	BBmsg.bPoint.p5.z = -1.0;

	// std::cout << "p0: (" << BBmsg.bPoint.p0.x << "," << BBmsg.bPoint.p0.y << ")" << std::endl;
	// std::cout << "p3: (" << BBmsg.bPoint.p3.x << "," << BBmsg.bPoint.p3.y << ")" << std::endl;
	// std::cout << "p7: (" << BBmsg.bPoint.p7.x << "," << BBmsg.bPoint.p7.y << ")" << std::endl;
	// std::cout << "p4: (" << BBmsg.bPoint.p4.x << "," << BBmsg.bPoint.p4.y << ")" << std::endl;	
}

void abs_obs_marker_dynamic(object Obsdy_object, int index, double velocity, int hz, int id, msgs::DetectedObject &BBmsg, visualization_msgs::Marker &marker)
{
	tf::Quaternion Obsdy_q;
	Obsdy_q.setRPY( Obsdy_object.roll, Obsdy_object.pitch, Obsdy_object.yaw);  // Create this quaternion from roll/pitch/yaw (in radians)
	Obsdy_object.x += std::cos(Obsdy_object.yaw)*index*velocity/hz;
	Obsdy_object.y += std::sin(Obsdy_object.yaw)*index*velocity/hz;
	emitRow("CUBE", visualization_msgs::Marker::CUBE, Obsdy_object.x, Obsdy_object.y, Obsdy_object.z, Obsdy_q.x(), Obsdy_q.y(), Obsdy_q.z(), Obsdy_q.w(), 0.7, 0.0, 0.3, ros::Duration(0.2), abs_marker_pub, marker,false,"map", Obsdy_object.sx, Obsdy_object.sy, Obsdy_object.sz, id);

	double CY = std::cos(Obsdy_object.yaw);
	double SY = std::sin(Obsdy_object.yaw);

	BBmsg.header.frame_id = "lidar";
    BBmsg.header.stamp = ros::Time::now();
	BBmsg.bPoint.p0.x = Obsdy_object.x + CY*(-Obsdy_object.sx)/2 - SY*Obsdy_object.sy/2;
	BBmsg.bPoint.p0.y = Obsdy_object.y + SY*(-Obsdy_object.sx)/2 + CY*Obsdy_object.sy/2;
	BBmsg.bPoint.p0.z = -3.0;
	BBmsg.bPoint.p3.x = Obsdy_object.x + CY*(-Obsdy_object.sx/2) - SY*(-Obsdy_object.sy)/2;
	BBmsg.bPoint.p3.y = Obsdy_object.y + SY*(-Obsdy_object.sx/2) + CY*(-Obsdy_object.sy)/2;
	BBmsg.bPoint.p3.z = -3.0;
	BBmsg.bPoint.p7.x = Obsdy_object.x + CY*Obsdy_object.sx/2 - SY*(-Obsdy_object.sy/2);
	BBmsg.bPoint.p7.y = Obsdy_object.y + SY*Obsdy_object.sx/2 + CY*(-Obsdy_object.sy/2);
	BBmsg.bPoint.p7.z = -3.0;
	BBmsg.bPoint.p4.x = Obsdy_object.x + CY*Obsdy_object.sx/2 - SY*Obsdy_object.sy/2;
	BBmsg.bPoint.p4.y = Obsdy_object.y + SY*Obsdy_object.sx/2 + CY*Obsdy_object.sy/2;
	BBmsg.bPoint.p4.z = -3.0;
	BBmsg.bPoint.p1.x = BBmsg.bPoint.p0.x;
	BBmsg.bPoint.p1.y = BBmsg.bPoint.p0.y;
	BBmsg.bPoint.p1.z = -1.0;
	BBmsg.bPoint.p2.x = BBmsg.bPoint.p3.x;
	BBmsg.bPoint.p2.y = BBmsg.bPoint.p3.y;
	BBmsg.bPoint.p2.z = -1.0;
	BBmsg.bPoint.p6.x = BBmsg.bPoint.p7.x;
	BBmsg.bPoint.p6.y = BBmsg.bPoint.p7.y;
	BBmsg.bPoint.p6.z = -1.0;
	BBmsg.bPoint.p5.x = BBmsg.bPoint.p4.x;
	BBmsg.bPoint.p5.y = BBmsg.bPoint.p4.y;
	BBmsg.bPoint.p5.z = -1.0;

	BBmsg.track.id = id;
	BBmsg.track.absolute_velocity.x = std::cos(Obsdy_object.yaw)*velocity;
	BBmsg.track.absolute_velocity.y = std::sin(Obsdy_object.yaw)*velocity;
	BBmsg.track.absolute_velocity.z = 0.0;
	BBmsg.track.absolute_velocity.speed = velocity;

	double center_x, center_y;
	center_x = (BBmsg.bPoint.p0.x + BBmsg.bPoint.p3.x + BBmsg.bPoint.p7.x + BBmsg.bPoint.p4.x)/4.0;
	center_y = (BBmsg.bPoint.p0.y + BBmsg.bPoint.p3.y + BBmsg.bPoint.p7.y + BBmsg.bPoint.p4.y)/4.0;

	int pp_index = std::floor(ppft/ppdt);
	msgs::PathPrediction ppxy;
	BBmsg.track.is_ready_prediction = true;
	int PPid = 0;
	visualization_msgs::MarkerArray PPmarker_array;
	for (int i=1; i<= pp_index; i++)
	{
		ppxy.position.x = center_x + i*ppdt*BBmsg.track.absolute_velocity.x;
		ppxy.position.y = center_y + i*ppdt*BBmsg.track.absolute_velocity.y;
		BBmsg.track.forecasts.push_back(ppxy);
		visualization_msgs::Marker PPmarker;
		PPid ++;
		emitRow("SPHERE", visualization_msgs::Marker::SPHERE, ppxy.position.x, ppxy.position.y, 0, Obsdy_q.x(), Obsdy_q.y(), Obsdy_q.z(), Obsdy_q.w(), 1.0, 0.0, 0.0, ros::Duration(0.2), abs_PPmarker_pub, PPmarker,false,"map", 0.5, 0.5, 0.5, PPid);
		PPmarker_array.markers.push_back(PPmarker);
	}
	abs_PPmarker_array_pub.publish(PPmarker_array);
	// std::cout << "p0: (" << BBmsg.bPoint.p0.x << "," << BBmsg.bPoint.p0.y << ")" << std::endl;
	// std::cout << "p3: (" << BBmsg.bPoint.p3.x << "," << BBmsg.bPoint.p3.y << ")" << std::endl;
	// std::cout << "p7: (" << BBmsg.bPoint.p7.x << "," << BBmsg.bPoint.p7.y << ")" << std::endl;
	// std::cout << "p4: (" << BBmsg.bPoint.p4.x << "," << BBmsg.bPoint.p4.y << ")" << std::endl;	
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

void VirObjTriggerCallback(virbb_pub::virbb_pub_obj_trigger AOTmsg)
{
	AEBObjTrigger = AOTmsg.AEB_obj_trigger;
	StaticObjTrigger = AOTmsg.Static_obj_trigger;
	DynamicObjTrigger = AOTmsg.Dynamic_obj_trigger;
	rvizPubPointObjTrigger = AOTmsg.rvizPP_obj_trigger;
}
void aebobjgen(double dist)
{
	AEB_obj_pose = current_pose;
	AEB_obj_pose.x = current_pose.x + std::cos(current_pose.yaw)*dist;
	AEB_obj_pose.y = current_pose.y + std::sin(current_pose.yaw)*dist;
	AEB_obj_pose.yaw = current_pose.yaw;
}
void aebobj2msg(object Obsaeb_object, bool &pushback_flag, int id, msgs::DetectedObject &BBmsg, visualization_msgs::Marker &marker)
{
	tf::Quaternion Obsdy_q;
	Obsdy_q.setRPY(AEB_obj_pose.roll, AEB_obj_pose.pitch, AEB_obj_pose.yaw);  // Create this quaternion from roll/pitch/yaw (in radians)
	emitRow("CUBE", visualization_msgs::Marker::CUBE, AEB_obj_pose.x, AEB_obj_pose.y, AEB_obj_pose.z, Obsdy_q.x(), Obsdy_q.y(), Obsdy_q.z(), Obsdy_q.w(), 0.0, 0.0, 1.0, ros::Duration(0.2), abs_marker_pub, marker,false,"map", Obsaeb_object.sx, Obsaeb_object.sy, Obsaeb_object.sz, id);

	double CY = std::cos(AEB_obj_pose.yaw);
	double SY = std::sin(AEB_obj_pose.yaw);

	BBmsg.header.frame_id = "lidar";
    BBmsg.header.stamp = ros::Time::now();
	BBmsg.bPoint.p0.x = AEB_obj_pose.x + CY*(-Obsaeb_object.sx)/2 - SY*Obsaeb_object.sy/2;
	BBmsg.bPoint.p0.y = AEB_obj_pose.y + SY*(-Obsaeb_object.sx)/2 + CY*Obsaeb_object.sy/2;
	BBmsg.bPoint.p0.z = -3.0;
	BBmsg.bPoint.p3.x = AEB_obj_pose.x + CY*(-Obsaeb_object.sx/2) - SY*(-Obsaeb_object.sy)/2;
	BBmsg.bPoint.p3.y = AEB_obj_pose.y + SY*(-Obsaeb_object.sx/2) + CY*(-Obsaeb_object.sy)/2;
	BBmsg.bPoint.p3.z = -3.0;
	BBmsg.bPoint.p7.x = AEB_obj_pose.x + CY*Obsaeb_object.sx/2 - SY*(-Obsaeb_object.sy/2);
	BBmsg.bPoint.p7.y = AEB_obj_pose.y + SY*Obsaeb_object.sx/2 + CY*(-Obsaeb_object.sy/2);
	BBmsg.bPoint.p7.z = -3.0;
	BBmsg.bPoint.p4.x = AEB_obj_pose.x + CY*Obsaeb_object.sx/2 - SY*Obsaeb_object.sy/2;
	BBmsg.bPoint.p4.y = AEB_obj_pose.y + SY*Obsaeb_object.sx/2 + CY*Obsaeb_object.sy/2;
	BBmsg.bPoint.p4.z = -3.0;
	BBmsg.bPoint.p1.x = BBmsg.bPoint.p0.x;
	BBmsg.bPoint.p1.y = BBmsg.bPoint.p0.y;
	BBmsg.bPoint.p1.z = -1.0;
	BBmsg.bPoint.p2.x = BBmsg.bPoint.p3.x;
	BBmsg.bPoint.p2.y = BBmsg.bPoint.p3.y;
	BBmsg.bPoint.p2.z = -1.0;
	BBmsg.bPoint.p6.x = BBmsg.bPoint.p7.x;
	BBmsg.bPoint.p6.y = BBmsg.bPoint.p7.y;
	BBmsg.bPoint.p6.z = -1.0;
	BBmsg.bPoint.p5.x = BBmsg.bPoint.p4.x;
	BBmsg.bPoint.p5.y = BBmsg.bPoint.p4.y;
	BBmsg.bPoint.p5.z = -1.0;

	// std::cout << "p0: (" << BBmsg.bPoint.p0.x << "," << BBmsg.bPoint.p0.y << ")" << std::endl;
	// std::cout << "p3: (" << BBmsg.bPoint.p3.x << "," << BBmsg.bPoint.p3.y << ")" << std::endl;
	// std::cout << "p7: (" << BBmsg.bPoint.p7.x << "," << BBmsg.bPoint.p7.y << ")" << std::endl;
	// std::cout << "p4: (" << BBmsg.bPoint.p4.x << "," << BBmsg.bPoint.p4.y << ")" << std::endl;	
}
void abs_obs_marker_AEB(object Obsaeb_object, double dist, bool &pushback_flag, int id, msgs::DetectedObject &BBmsg, visualization_msgs::Marker &marker)
{
	if (AEBObjTrigger == true && AEBObjTrigger_last == false)
	{
		aebobjgen(dist);
		aebobj2msg(Obsaeb_object, pushback_flag, id, BBmsg, marker);
		pushback_flag = true;
	}
	else if (AEBObjTrigger == true && AEBObjTrigger_last == true)
	{
		aebobj2msg(Obsaeb_object, pushback_flag, id, BBmsg, marker);
		pushback_flag = true;
	}
	else
		pushback_flag = false;

	AEBObjTrigger_last = AEBObjTrigger;
}

void obs_main(int hz)
{
	// Initial
	int id = 0;
	msgs::DetectedObject abs_BBmsg;
	msgs::DetectedObjectArray abs_BBArraymsg;
	visualization_msgs::MarkerArray abs_MAmsg;
	visualization_msgs::Marker abs_marker;
	msgs::DetectedObject rel_BBmsg;
	msgs::DetectedObjectArray rel_BBArraymsg;

	rel_BBArraymsg.header.frame_id = "lidar";
    rel_BBArraymsg.header.stamp = ros::Time::now();

    abs_BBArraymsg.header.frame_id = "map";
    abs_BBArraymsg.header.stamp = ros::Time::now();

	// Static	
	if (StaticObjTrigger == true)
	{
		id ++;
		object abs_Obs1_object;
		abs_Obs1_object.x = read_tmp_s[0];//-2.851;
		abs_Obs1_object.y = read_tmp_s[1];//-65.85855;
		abs_Obs1_object.z = read_tmp_s[2];//1.0;
		abs_Obs1_object.roll = read_tmp_s[3];//0;
		abs_Obs1_object.pitch = read_tmp_s[4];//0;
		abs_Obs1_object.yaw = read_tmp_s[5];//1.44;
		abs_Obs1_object.sx = read_tmp_s[6];//4.6;
		abs_Obs1_object.sy = read_tmp_s[7];//1.8;
		abs_Obs1_object.sz = read_tmp_s[8];//1.5;
		abs_obs_marker_static(abs_Obs1_object, id, abs_BBmsg, abs_marker);
		abs2rel(abs_BBmsg,rel_BBmsg);
		abs_BBArraymsg.objects.push_back(abs_BBmsg);
		abs_MAmsg.markers.push_back(abs_marker);
		rel_BBArraymsg.objects.push_back(rel_BBmsg);	
	}

	// Dynamic
	if (DynamicObjTrigger == true)
	{
		id ++;
		object abs_Obsdy1_object;
		abs_Obsdy1_object.x = read_tmp_d[0];//-2.851;
		abs_Obsdy1_object.y = read_tmp_d[1];//-65.85855;
		abs_Obsdy1_object.z = read_tmp_d[2];//1.0;
		abs_Obsdy1_object.roll = read_tmp_d[3];//0;
		abs_Obsdy1_object.pitch = read_tmp_d[4];//0;
		abs_Obsdy1_object.yaw = read_tmp_d[5];//1.44;
		abs_Obsdy1_object.sx = read_tmp_d[6];//4.6;
		abs_Obsdy1_object.sy = read_tmp_d[7];//1.8;
		abs_Obsdy1_object.sz = read_tmp_d[8];//1.5;
		double velocity = read_tmp_d[9];//velocity; //(m/s)
		int time_loop = read_tmp_d[10]; 
		abs_obs_marker_dynamic(abs_Obsdy1_object, dynamicobj_index, velocity, hz, id, abs_BBmsg, abs_marker);
		abs2rel(abs_BBmsg,rel_BBmsg);
		dynamicobj_index ++;
		if (dynamicobj_index > time_loop*hz)
			dynamicobj_index = 0; 
		abs_BBArraymsg.objects.push_back(abs_BBmsg);
		abs_MAmsg.markers.push_back(abs_marker);
		rel_BBArraymsg.objects.push_back(rel_BBmsg);
	}
	if (DynamicObjTrigger == false)
		dynamicobj_index = 0;

	// AEB obj
	id ++;
	object abs_Obsaeb1_object;
	abs_Obsaeb1_object.sx = read_tmp_a[0];//4.6;
	abs_Obsaeb1_object.sy = read_tmp_a[1];//1.8;
	abs_Obsaeb1_object.sz = read_tmp_a[2];//1.5;
	double dist = read_tmp_a[3];
	bool AEB_pushback_flag = false;
	abs_obs_marker_AEB(abs_Obsaeb1_object, dist, AEB_pushback_flag, id, abs_BBmsg, abs_marker);
	abs2rel(abs_BBmsg,rel_BBmsg);
	if (AEB_pushback_flag == true)
	{
		abs_BBArraymsg.objects.push_back(abs_BBmsg);
		abs_MAmsg.markers.push_back(abs_marker);
		rel_BBArraymsg.objects.push_back(rel_BBmsg);
	}

	// rviz PubPoint obj
	if (rvizPubPointObjTrigger == true)
	{
		id ++;
		object abs_ObsrvizPP1_object;
		abs_ObsrvizPP1_object.roll = read_tmp_r[0] + rvizPublisherPoint.roll;
		abs_ObsrvizPP1_object.pitch = read_tmp_r[1] + rvizPublisherPoint.pitch;
		abs_ObsrvizPP1_object.yaw = read_tmp_r[2] + rvizPublisherPoint.yaw;
		abs_ObsrvizPP1_object.x = rvizPublisherPoint.x;
		abs_ObsrvizPP1_object.y = rvizPublisherPoint.y;
		abs_ObsrvizPP1_object.z = rvizPublisherPoint.z;	
		abs_ObsrvizPP1_object.sx = read_tmp_r[3];//4.6;
		abs_ObsrvizPP1_object.sy = read_tmp_r[4];//1.8;
		abs_ObsrvizPP1_object.sz = read_tmp_r[5];//1.5;
		double velocity = read_tmp_r[6];//velocity; //(m/s) 
		abs_obs_marker_dynamic(abs_ObsrvizPP1_object, rvizppobj_index, velocity, hz, id, abs_BBmsg, abs_marker);
		abs2rel(abs_BBmsg,rel_BBmsg);
		rvizppobj_index++;
		abs_BBArraymsg.objects.push_back(abs_BBmsg);
		abs_MAmsg.markers.push_back(abs_marker);
		rel_BBArraymsg.objects.push_back(rel_BBmsg);
	}
	if (rvizPubPointObjTrigger == false)
		rvizppobj_index = 0;
	
	// Publish
	abs_BB_array_pub.publish(abs_BBArraymsg);
	abs_marker_array_pub.publish(abs_MAmsg);
	rel_BB_array_pub.publish(rel_BBArraymsg);
}

template <int size_readtmp>
void read_txt(std::string fpname, double (&read_tmp)[size_readtmp])
{
	int read_index = 0;
	std::string fname = fpname;

  	std::ifstream fin;
    char line[100];
    memset( line, 0, sizeof(line));

    fin.open(fname.c_str(),std::ios::in);
    if(!fin) 
    {
        std::cout << "Fail to import txt" <<std::endl;
        exit(1);
    }

    while(fin.getline(line,sizeof(line),',')) 
    {
		// fin.getline(line,sizeof(line),'\n');
	    std::string nmea_str(line);
	    std::stringstream ss(nmea_str);
	    std::string token;

	    getline(ss,token, ',');
	    read_tmp[read_index] = atof(token.c_str());
	    read_index += 1;
    }
}

void Ini_obs_bytxt()
{
	std::string fpname = ros::package::getPath("virbb_pub");

	std::string fpname_s = fpname + "/data/StaticObj.txt";
	read_txt(fpname_s, read_tmp_s);

	std::string fpname_d = fpname + "/data/DynamicObj.txt";
	read_txt(fpname_d, read_tmp_d);

	std::string fpname_a = fpname + "/data/AEBObj.txt";
	read_txt(fpname_a, read_tmp_a);

	std::string fpname_r = fpname + "/data/rvizPubPointObj.txt";
	read_txt(fpname_r, read_tmp_r);
}

int main( int argc, char **argv )
{
	Ini_obs_bytxt();

	// ros initial
	ros::init(argc, argv, "virbb_pub");
	ros::NodeHandle nh;

	// Ini
	int hz = 100;
	if (ros::param::get(ros::this_node::getName()+"/hz", hz));
	ppdt = 0.5;
	if (ros::param::get(ros::this_node::getName()+"/ppdt", ppdt));
	ppft = 2;
	if (ros::param::get(ros::this_node::getName()+"/ppft", ppft));

	// Subscriber
	// ros::Subscriber object_subscriber = nh.subscribe("current_pose", 1, RelchatterCallback);
	ros::Subscriber current_pose_sub = nh.subscribe("current_pose", 1, CurrentPoseCallback);
	ros::Subscriber AEB_Obj_Trigger_sub = nh.subscribe("vir_obj_trigger", 1, VirObjTriggerCallback);
	ros::Subscriber Localizationukf_sub = nh.subscribe("localization_ukf", 1, LocalizationUKFCallback);
	ros::Subscriber rvizPublisherPoint_sub = nh.subscribe("clicked_point", 1, rvizPublisherPointCallback);

	// Publisher
	// mmtp_pub = nh.advertise<virbb_pub::MM_TP_input_msg>("mmtppubtopic",1);
	abs_BB_array_pub = nh.advertise<msgs::DetectedObjectArray>("abs_virBB_array",1);
	// polygon_pub = nh.advertise<geometry_msgs::PolygonStamped>("polygon_topic",1);
	// polygon0_pub = nh.advertise<geometry_msgs::PolygonStamped>("polygon0_topic",1);
	abs_marker_pub = nh.advertise<visualization_msgs::Marker>("marker_topic",1);
	abs_marker_array_pub = nh.advertise<visualization_msgs::MarkerArray>("marker_array_topic",1);
	rel_BB_array_pub = nh.advertise<msgs::DetectedObjectArray>("rel_virBB_array",1);
	abs_PPmarker_pub = nh.advertise<visualization_msgs::Marker>("PPmarker_topic",1);
	abs_PPmarker_array_pub = nh.advertise<visualization_msgs::MarkerArray>("abs_PPmarker_array",1);

	ros::Rate loop_rate(hz);
	while (ros::ok())
	{
		obs_main(hz);
		ros::spinOnce();
		loop_rate.sleep();
	}
	// ros::spin();
	return 0 ;
} // main