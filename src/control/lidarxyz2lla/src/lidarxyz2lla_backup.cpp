#include <iostream>
#include <iomanip>
#include "geodetic_utils/geodetic_conv.hpp"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float64.h"
#include "msgs/LidLLA.h"
#include <sstream>
#include <ros/package.h>
#include <string.h>
#include <fstream>
#include <tf/tf.h>
#include <geometry_msgs/PoseStamped.h>

//------------------------------------------------------------------------------------------------------
// Global Parameter
double ini_lat; // initial latitude
double ini_lon; // initial longitude
double ini_alt; // initial altitude
double R[3][3]; // XYZ to ENU R
double T[3]; // XYZ to ENU T

ros::Publisher lidarlla_pub; 
ros::Subscriber lidarlla_sub ;
// lidarxyz2lla::lidarxyz lidarxyzmsg;
//------------------------------------------------------------------------------------------------------

geodetic_converter::GeodeticConverter g_geodetic_converter;

void initial_para()
{
	double read_tmp[15];
	int read_index = 0;
	std::string fname = ros::package::getPath("lidarxyz2lla");
	fname += "/data/Taichung_LidXYZ2ENU.txt";
  	std::cout << fname << std::endl;

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
    std::cout << read_tmp[10] << std::endl;
    ini_lon = read_tmp[0];
    ini_lat = read_tmp[1];
    ini_alt = read_tmp[2];
    R[0][0] = read_tmp[3];
    R[0][1] = read_tmp[4];
    R[0][2] = read_tmp[5];
    R[1][0] = read_tmp[6];
    R[1][1] = read_tmp[7];
    R[1][2] = read_tmp[8];
    R[2][0] = read_tmp[9];
    R[2][1] = read_tmp[10];
    R[2][2] = read_tmp[11];
    T[0] = read_tmp[12];
    T[1] = read_tmp[13];
    T[2] = read_tmp[14];
    std::cout << "init_long : " << std::setprecision(20) << ini_lon << std::endl;
    std::cout << "init_lat : " << std::setprecision(20) << ini_lat << std::endl;
    std::cout << "init_alt : " << std::setprecision(20) << ini_alt << std::endl;
}

// void lidarxyztopicCallback(const lidarxyz2lla::LidXYZ::ConstPtr& lidarxyzmsg)
// {

// 	// input lidar_XYZ
// 	double lidar_X = lidarxyzmsg->lidar_X;//-0.43;
// 	double lidar_Y = lidarxyzmsg->lidar_Y;//-32.68;
// 	double lidar_Z = lidarxyzmsg->lidar_Z;//-0.53;

// 	// lidar_XYZ to lidar_ENU
// 	double d[3] = { lidar_X, lidar_Y, lidar_Z };
// 	double d_new[3];
// 	for (int i = 0; i < sizeof(d)/sizeof(d[0]); i++)
// 	{
// 		d_new[i] = R[i][0] * d[0] + R[i][1] * d[1] + R[i][2] * d[2] + T[i];
// 	}
// 	double lidar_E = d_new[0];
// 	double lidar_N = d_new[1];
// 	double lidar_U = d_new[2];
// 	std::cout <<"lidar E : "<< std::setprecision(20) << lidar_E << std::endl;
// 	std::cout <<"lidar N : "<< std::setprecision(20) << lidar_N << std::endl;
// 	std::cout <<"lidar U : "<< std::setprecision(20) << lidar_U << std::endl;

// 	// initial ecef
// 	g_geodetic_converter.initialiseReference(ini_lat, ini_lon, ini_alt);
// 	// std::cout << initial_ecef_x_ << std::endl;

// 	// lidar_ENU to lidar_LLA
// 	double lidar_lat;
// 	double lidar_lon;
// 	double lidar_alt;
// 	g_geodetic_converter.enu2Geodetic(lidar_E, lidar_N, lidar_U, &lidar_lat, &lidar_lon, &lidar_alt);

// 	std::cout <<"lidar Lat : "<< std::setprecision(20) << lidar_lat << std::endl;
// 	std::cout <<"lidar Lon : "<< std::setprecision(20) << lidar_lon << std::endl;
// 	std::cout <<"lidar Alt : "<< std::setprecision(20) << lidar_alt << std::endl;

// 	msgs::LidLLA lidarllamsg;
// 	lidarllamsg.lidar_Lat = lidar_lat;
// 	lidarllamsg.lidar_Lon = lidar_lon;
// 	lidarllamsg.lidar_Alt = lidar_alt;
// 	lidarlla_pub.publish(lidarllamsg);
// }

void lidarxyztopicCallback_1(const geometry_msgs::PoseStamped::ConstPtr& lidarxyzmsg)
{

	// input lidar_XYZ
	double lidar_X = lidarxyzmsg->pose.position.x;//-0.43;
	double lidar_Y = lidarxyzmsg->pose.position.y;//-32.68;
	double lidar_Z = lidarxyzmsg->pose.position.z;//-0.53;

	// lidar_XYZ to lidar_ENU
	double d[3] = { lidar_X, lidar_Y, lidar_Z };
	double d_new[3];
	for (int i = 0; i < sizeof(d)/sizeof(d[0]); i++)
	{
		d_new[i] = R[i][0] * d[0] + R[i][1] * d[1] + R[i][2] * d[2] + T[i];
	}
	double lidar_E = d_new[0];
	double lidar_N = d_new[1];
	double lidar_U = d_new[2];
	std::cout <<"lidar E : "<< std::setprecision(20) << lidar_E << std::endl;
	std::cout <<"lidar N : "<< std::setprecision(20) << lidar_N << std::endl;
	std::cout <<"lidar U : "<< std::setprecision(20) << lidar_U << std::endl;

	// initial ecef
	g_geodetic_converter.initialiseReference(ini_lat, ini_lon, ini_alt);
	// std::cout << initial_ecef_x_ << std::endl;

	// lidar_ENU to lidar_LLA
	double lidar_lat;
	double lidar_lon;
	double lidar_alt;
	g_geodetic_converter.enu2Geodetic(lidar_E, lidar_N, lidar_U, &lidar_lat, &lidar_lon, &lidar_alt);

	std::cout <<"lidar Lat : "<< std::setprecision(20) << lidar_lat << std::endl;
	std::cout <<"lidar Lon : "<< std::setprecision(20) << lidar_lon << std::endl;
	std::cout <<"lidar Alt : "<< std::setprecision(20) << lidar_alt << std::endl;

	msgs::LidLLA lidarllamsg;
	lidarllamsg.lidar_Lat = lidar_lat;
	lidarllamsg.lidar_Lon = lidar_lon;
	lidarllamsg.lidar_Alt = lidar_alt;
	lidarlla_pub.publish(lidarllamsg);
}

int main( int argc, char **argv )
{
	// initial parameter
	initial_para();
	// ros initial
	ros::init(argc, argv, "lidarxyz2lla");
	ros::NodeHandle nh;

	// subscriber
	lidarlla_sub = nh.subscribe("current_pose", 1, lidarxyztopicCallback_1);
	lidarlla_pub = nh.advertise<msgs::LidLLA>("lidar_lla", 1);
	ros::spin();

	return 0 ;
} // main