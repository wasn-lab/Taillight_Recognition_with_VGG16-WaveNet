#include <iostream>
#include <iomanip>
#include "geodetic_utils/geodetic_conv.hpp"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float64.h"
#include "msgs/LidLLA.h"
#include <sstream>
#include <ros/package.h>
#include <cstring>
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
double T1[3],R1[3][3],T2[3],R2[3][3],T3[3],R3[3][3],T4[3],R4[3][3],T5[3],R5[3][3];

int LidXYZ2ENU_siwtch = 1;

ros::Publisher lidarlla_pub; 
ros::Subscriber lidarlla_sub ;
// lidarxyz2lla::lidarxyz lidarxyzmsg;
//------------------------------------------------------------------------------------------------------

geodetic_converter::GeodeticConverter g_geodetic_converter;

void initial_para()
{
	double read_tmp[63];
	int read_index = 0;
	std::string fname = ros::package::getPath("lidarxyz2lla");
	fname += "/data/ITRI_NEW_LidXYZ2ENU_sec.txt";
	// fname += "/data/Shalun_LidXYZ2ENU.txt";
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
    if (LidXYZ2ENU_siwtch == 0)
        {
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
        }
    else
        {
        	R1[0][0] = read_tmp[3];
	        R1[0][1] = read_tmp[4];
	        R1[0][2] = read_tmp[5];
	        R1[1][0] = read_tmp[6];
	        R1[1][1] = read_tmp[7];
	        R1[1][2] = read_tmp[8];
	        R1[2][0] = read_tmp[9];
	        R1[2][1] = read_tmp[10];
	        R1[2][2] = read_tmp[11];
	        R2[0][0] = read_tmp[12];
	        R2[0][1] = read_tmp[13];
	        R2[0][2] = read_tmp[14];
	        R2[1][0] = read_tmp[15];
	        R2[1][1] = read_tmp[16];
	        R2[1][2] = read_tmp[17];
	        R2[2][0] = read_tmp[18];
	        R2[2][1] = read_tmp[19];
	        R2[2][2] = read_tmp[20];
	        R3[0][0] = read_tmp[21];
	        R3[0][1] = read_tmp[22];
	        R3[0][2] = read_tmp[23];
	        R3[1][0] = read_tmp[24];
	        R3[1][1] = read_tmp[25];
	        R3[1][2] = read_tmp[26];
	        R3[2][0] = read_tmp[27];
	        R3[2][1] = read_tmp[28];
	        R3[2][2] = read_tmp[29];
	        R4[0][0] = read_tmp[30];
	        R4[0][1] = read_tmp[31];
	        R4[0][2] = read_tmp[32];
	        R4[1][0] = read_tmp[33];
	        R4[1][1] = read_tmp[34];
	        R4[1][2] = read_tmp[35];
	        R4[2][0] = read_tmp[36];
	        R4[2][1] = read_tmp[37];
	        R4[2][2] = read_tmp[38];
	        R5[0][0] = read_tmp[39];
	        R5[0][1] = read_tmp[40];
	        R5[0][2] = read_tmp[41];
	        R5[1][0] = read_tmp[42];
	        R5[1][1] = read_tmp[43];
	        R5[1][2] = read_tmp[44];
	        R5[2][0] = read_tmp[45];
	        R5[2][1] = read_tmp[46];
	        R5[2][2] = read_tmp[47];
	        T1[0] = read_tmp[48];
	        T1[1] = read_tmp[49];
	        T1[2] = read_tmp[50];
	        T2[0] = read_tmp[51];
	        T2[1] = read_tmp[52];
	        T2[2] = read_tmp[53];
	        T3[0] = read_tmp[54];
	        T3[1] = read_tmp[55];
	        T3[2] = read_tmp[56];
	        T4[0] = read_tmp[57];
	        T4[1] = read_tmp[58];
	        T4[2] = read_tmp[59];
	        T5[0] = read_tmp[60];
	        T5[1] = read_tmp[61];
	        T5[2] = read_tmp[62];
	        std::cout << "T5[0] : " << std::setprecision(20) << T5[0] << std::endl;
        }
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
	double R_final[3][3],T_final[3];

	// input lidar_XYZ
	double lidar_X = lidarxyzmsg->pose.position.x;//-0.43;
	double lidar_Y = lidarxyzmsg->pose.position.y;//-32.68;
	double lidar_Z = lidarxyzmsg->pose.position.z;//-0.53;

	if (LidXYZ2ENU_siwtch == 0)
        {
        	for (int i = 0;i < 3;i++)
	        {
	        	for (int j = 0; j < 3;j++)
	        	{
	        		R_final[i][j] = R[i][j];
	        		T_final[i] = T[i];
	        	}
	        }
        }
    else
        {
        	for (int i = 0;i < 3;i++)
	        {
	        	for (int j = 0; j < 3;j++)
	        	{
	        		if (lidar_X <= 0)
	        		{
	        			R_final[i][j] = R1[i][j];
	        			T_final[i] = T1[i];
	        		}
	        		else if (lidar_X > 0 && lidar_X <= 100)
	        		{
	        			R_final[i][j] = R2[i][j];
	        			T_final[i] = T2[i];
	        		}
	        		else if (lidar_X > 100 && lidar_X <= 225)
	        		{
	        			R_final[i][j] = R3[i][j];
	        			T_final[i] = T3[i];
	        		}
	        		else if (lidar_X > 225 && lidar_X <= 350)
	        		{
	        			R_final[i][j] = R4[i][j];
	        			T_final[i] = T4[i];
	        		}
	        		else
	        		{
	        			R_final[i][j] = R5[i][j];
	        			T_final[i] = T5[i];
	        		}
	        	}
	        	
	        }
        }

	// lidar_XYZ to lidar_ENU
	double d[3] = { lidar_X, lidar_Y, lidar_Z };
	double d_new[3];
	for (int i = 0; i < sizeof(d)/sizeof(d[0]); i++)
	{
		d_new[i] = R_final[i][0] * d[0] + R_final[i][1] * d[1] + R_final[i][2] * d[2] + T_final[i];
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