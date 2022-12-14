#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

// For ROS
#include "ros/ros.h"
#include <std_msgs/Header.h>
#include <std_msgs/Empty.h>

#include "std_msgs/String.h"
#include "msgs/Rad.h"
#include "msgs/BoxPoint.h"
#include "msgs/DynamicPath.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/PathPrediction.h"
#include "msgs/PointXY.h"
#include "msgs/PointXYZ.h"
#include "msgs/PointXYZV.h"
#include "msgs/TrackInfo.h"
#include "sensor_msgs/Imu.h"
#include "msgs/RadObject.h"
#include "msgs/RadObjectArray.h"
#include "msgs/LocalizationToVeh.h"
#include <cstring>
#include <visualization_msgs/Marker.h>

// For PCL
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

// For eigen
#include <Eigen/Dense>

#define PI_OVER_180 (0.0174532925)

using namespace std;

void callbackDelphiFront(const msgs::Rad::ConstPtr& msg);
void callbackAlphaFrontCenter(const msgs::Rad::ConstPtr& msg);
void callbackAlphaFrontLeft(const msgs::Rad::ConstPtr& msg);
void callbackAlphaFrontRight(const msgs::Rad::ConstPtr& msg);
void callbackAlphaSideLeft(const msgs::Rad::ConstPtr& msg);
void callbackAlphaSideRight(const msgs::Rad::ConstPtr& msg);
void callbackAlphaBackLeft(const msgs::Rad::ConstPtr& msg);
void callbackAlphaBackRight(const msgs::Rad::ConstPtr& msg);
void callbackCubtekFront(const msgs::RadObjectArray::ConstPtr& msg);
void callbackIMU(const sensor_msgs::Imu::ConstPtr& input);
void pointCalibration(float* x, float* y, float* z, int type);
void onInit(ros::NodeHandle nh, ros::NodeHandle n);
void transInitGuess(int type);

void msgPublisher();

ros::Publisher RadFrontPub;
ros::Publisher RadAlphaPub;
ros::Publisher RadAlphaPCLPub;
ros::Publisher RadCubtekPub;
ros::Publisher RadCubtekPCLPub;
ros::Publisher HeartbeatPub;

double imu_angular_velocity_z = 0;
int do_rotate = 0;
int print_count = 0;
int debug_message = 0;
int alpha_raw_message = 0;
int cubtek_raw_message = 0;
int delphi_raw_message = 0;

vector<float> Alpha_Front_Center_Param;
vector<float> Alpha_Front_Left_Param;
vector<float> Alpha_Front_Right_Param;
vector<float> Alpha_Side_Left_Param;
vector<float> Alpha_Side_Right_Param;
vector<float> Alpha_Back_Left_Param;
vector<float> Alpha_Back_Right_Param;

vector<float> Cubtek_Front_Center_Param;

vector<float> Zero_Param(6, 0.0);

msgs::Rad delphiRad;
msgs::Rad alphaRad;
msgs::Rad cubtekRad;

vector<msgs::PointXYZV> alphaAllVec;
vector<msgs::PointXYZV> alphaFrontCenterVec;
vector<msgs::PointXYZV> alphaFrontLeftVec;
vector<msgs::PointXYZV> alphaFrontRightVec;
vector<msgs::PointXYZV> alphaSideLeftVec;
vector<msgs::PointXYZV> alphaSideRightVec;
vector<msgs::PointXYZV> alphaBackLeftVec;
vector<msgs::PointXYZV> alphaBackRightVec;

vector<msgs::PointXYZV> cubtekVec;

Eigen::Matrix4f frontCenterInitGuess;
Eigen::Matrix4f frontLeftInitGuess;
Eigen::Matrix4f frontRightInitGuess;
Eigen::Matrix4f sideLeftInitGuess;
Eigen::Matrix4f sideRightInitGuess;
Eigen::Matrix4f backLeftInitGuess;
Eigen::Matrix4f backRightInitGuess;

void callbackDelphiFront(const msgs::Rad::ConstPtr& msg)
{
  int m_rotate = do_rotate;
  msgs::PointXYZV point;

  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    int m_NeedAdd = 0;
    double x = abs(msg->radPoint[i].x);
    double y = abs(msg->radPoint[i].y);
    double speed = abs(msg->radPoint[i].speed);

    if (speed > 80 || x > 80)
    {
      continue;
    }

    if (m_rotate == 1)
    {
      if (y < 1.6 && speed > 4)
      {
        m_NeedAdd = 1;
      }
      else if (y < 4 && y >= 1.6 && speed > 2.2)
      {
        m_NeedAdd = 1;
      }
      else if (y < 7 && y >= 4 && speed > 3)
      {
        m_NeedAdd = 1;
      }
      else if (y < 15 && y >= 7 && speed > 1)
      {
        m_NeedAdd = 1;
      }
    }
    else
    {
      // ??????????????????????????????AEB work
      m_NeedAdd = 1;
      // if (y < 1.2 && speed > 3)
      // {
      //   m_NeedAdd = 1;
      // }
      // else if (y < 15 && y >= 1.2 && speed > 0)
      // {
      //   m_NeedAdd = 1;
      // }
    }

    if (m_NeedAdd == 1)
    {
      point.x = msg->radPoint[i].x;
      point.y = msg->radPoint[i].y;
      point.z = 0;
      point.speed = msg->radPoint[i].speed;

      if (delphi_raw_message)
      {
        cout << "X: " << point.x << ", Y: " << point.y << ", Speed: " << point.speed << endl;
      }
      delphiRad.radPoint.push_back(point);
    }
  }
  if (debug_message)
  {
    cout << "Delphi Radar Data : " << delphiRad.radPoint.size() << endl;
  }
  delphiRad.radHeader.stamp = msg->radHeader.stamp;
  delphiRad.radHeader.seq = msg->radHeader.seq;
  RadFrontPub.publish(delphiRad);
  delphiRad.radPoint.clear();
  msgPublisher();
}

void callbackCubtekFront(const msgs::RadObjectArray::ConstPtr& msg)
{
  cubtekRad.radHeader.stamp = msg->header.stamp;
  cubtekRad.radHeader.seq = msg->header.seq;

  cubtekVec.clear();
  pcl::PointCloud<pcl::PointXYZI> temp_array;
  pcl::PointCloud<pcl::PointXYZI> out_cloud;

  for (int i = 0; i < msg->objects.size(); i++)
  {
    pcl::PointXYZI point;

    float x = msg->objects[i].px;
    float y = msg->objects[i].py;
    float z = -2;

    point.x = x;
    point.y = y;
    point.z = z;
    point.intensity = msg->objects[i].vx;
    temp_array.points.push_back(point);
  }

  float tx = Cubtek_Front_Center_Param[0];
  float ty = -Cubtek_Front_Center_Param[1];
  float tz = Cubtek_Front_Center_Param[2];
  float rx = Cubtek_Front_Center_Param[5] * PI_OVER_180;
  float ry = Cubtek_Front_Center_Param[4] * PI_OVER_180;
  float rz = -Cubtek_Front_Center_Param[3] * PI_OVER_180;

  Eigen::Affine3f mr = Eigen::Affine3f::Identity();

  mr.translation() << tx, ty, tz;
  mr.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));  // The angle of rotation in radians
  mr.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
  mr.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));
  pcl::PointCloud<pcl::PointXYZI>::Ptr output_Ptr = temp_array.makeShared();
  pcl::transformPointCloud(*output_Ptr, out_cloud, mr);  // no cuda

  for (int i = 0; i < out_cloud.points.size(); i++)
  {
    msgs::PointXYZV data;

    float x = out_cloud.points[i].x;
    float y = out_cloud.points[i].y;
    float z = out_cloud.points[i].z;

    data.x = x;
    data.y = -y;
    data.z = z;
    data.speed = out_cloud.points[i].intensity;

    cubtekVec.push_back(data);
  }
}

void callbackAlphaFrontCenter(const msgs::Rad::ConstPtr& msg)
{
  alphaRad.radHeader.stamp = msg->radHeader.stamp;
  alphaRad.radHeader.seq = msg->radHeader.seq;

  alphaFrontCenterVec.clear();
  pcl::PointCloud<pcl::PointXYZI> temp_array;
  pcl::PointCloud<pcl::PointXYZI> out_cloud;

  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    pcl::PointXYZI point;

    float x = msg->radPoint[i].x;
    float y = msg->radPoint[i].y;
    float z = msg->radPoint[i].z;

    point.x = x;
    point.y = y;
    point.z = z;
    point.intensity = msg->radPoint[i].speed;
    temp_array.points.push_back(point);
  }

  float tx = Alpha_Front_Center_Param[0];
  float ty = -Alpha_Front_Center_Param[1];
  float tz = Alpha_Front_Center_Param[2];
  float rx = Alpha_Front_Center_Param[5] * PI_OVER_180;
  float ry = Alpha_Front_Center_Param[4] * PI_OVER_180;
  float rz = -Alpha_Front_Center_Param[3] * PI_OVER_180;

  Eigen::Affine3f mr = Eigen::Affine3f::Identity();

  mr.translation() << tx, ty, tz;
  mr.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));  // The angle of rotation in radians
  mr.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
  mr.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));
  pcl::PointCloud<pcl::PointXYZI>::Ptr output_Ptr = temp_array.makeShared();
  pcl::transformPointCloud(*output_Ptr, out_cloud, mr);  // no cuda

  for (int i = 0; i < out_cloud.points.size(); i++)
  {
    msgs::PointXYZV data;

    float x = out_cloud.points[i].x;
    float y = out_cloud.points[i].y;
    float z = out_cloud.points[i].z;

    data.x = x;
    data.y = y;
    data.z = z;
    data.speed = out_cloud.points[i].intensity;

    alphaFrontCenterVec.push_back(data);
  }
}

void callbackAlphaFrontLeft(const msgs::Rad::ConstPtr& msg)
{
  alphaFrontLeftVec.clear();
  pcl::PointCloud<pcl::PointXYZI> temp_array;
  pcl::PointCloud<pcl::PointXYZI> out_cloud;

  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    pcl::PointXYZI point;

    float x = msg->radPoint[i].x;
    float y = msg->radPoint[i].y;
    float z = msg->radPoint[i].z;

    point.x = x;
    point.y = y;
    point.z = z;
    point.intensity = msg->radPoint[i].speed;
    temp_array.points.push_back(point);
  }

  float tx = Alpha_Front_Left_Param[0];
  float ty = -Alpha_Front_Left_Param[1];
  float tz = Alpha_Front_Left_Param[2];
  float rx = Alpha_Front_Left_Param[5] * PI_OVER_180;
  float ry = Alpha_Front_Left_Param[4] * PI_OVER_180;
  float rz = -Alpha_Front_Left_Param[3] * PI_OVER_180;

  Eigen::Affine3f mr = Eigen::Affine3f::Identity();

  mr.translation() << tx, ty, tz;
  mr.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));  // The angle of rotation in radians
  mr.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
  mr.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));
  pcl::PointCloud<pcl::PointXYZI>::Ptr output_Ptr = temp_array.makeShared();
  pcl::transformPointCloud(*output_Ptr, out_cloud, mr);  // no cuda

  for (int i = 0; i < out_cloud.points.size(); i++)
  {
    msgs::PointXYZV data;

    float x = out_cloud.points[i].x;
    float y = out_cloud.points[i].y;
    float z = out_cloud.points[i].z;

    data.x = x;
    data.y = y;
    data.z = z;
    data.speed = out_cloud.points[i].intensity;

    alphaFrontLeftVec.push_back(data);
  }
}

void callbackAlphaFrontRight(const msgs::Rad::ConstPtr& msg)
{
  alphaFrontRightVec.clear();
  pcl::PointCloud<pcl::PointXYZI> temp_array;
  pcl::PointCloud<pcl::PointXYZI> out_cloud;

  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    pcl::PointXYZI point;

    float x = msg->radPoint[i].x;
    float y = msg->radPoint[i].y;
    float z = msg->radPoint[i].z;

    point.x = x;
    point.y = y;
    point.z = z;
    point.intensity = msg->radPoint[i].speed;
    temp_array.points.push_back(point);
  }

  float tx = Alpha_Front_Right_Param[0];
  float ty = -Alpha_Front_Right_Param[1];
  float tz = Alpha_Front_Right_Param[2];
  float rx = Alpha_Front_Right_Param[5] * PI_OVER_180;
  float ry = Alpha_Front_Right_Param[4] * PI_OVER_180;
  float rz = -Alpha_Front_Right_Param[3] * PI_OVER_180;

  Eigen::Affine3f mr = Eigen::Affine3f::Identity();

  mr.translation() << tx, ty, tz;
  mr.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));  // The angle of rotation in radians
  mr.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
  mr.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));
  pcl::PointCloud<pcl::PointXYZI>::Ptr output_Ptr = temp_array.makeShared();
  pcl::transformPointCloud(*output_Ptr, out_cloud, mr);  // no cuda

  for (int i = 0; i < out_cloud.points.size(); i++)
  {
    msgs::PointXYZV data;

    float x = out_cloud.points[i].x;
    float y = out_cloud.points[i].y;
    float z = out_cloud.points[i].z;

    data.x = x;
    data.y = y;
    data.z = z;
    data.speed = out_cloud.points[i].intensity;

    alphaFrontRightVec.push_back(data);
  }
}

void callbackAlphaSideLeft(const msgs::Rad::ConstPtr& msg)
{
  alphaSideLeftVec.clear();
  pcl::PointCloud<pcl::PointXYZI> temp_array;
  pcl::PointCloud<pcl::PointXYZI> out_cloud;

  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    pcl::PointXYZI point;

    float x = msg->radPoint[i].x;
    float y = msg->radPoint[i].y;
    float z = msg->radPoint[i].z;

    point.x = x;
    point.y = y;
    point.z = z;
    point.intensity = msg->radPoint[i].speed;
    temp_array.points.push_back(point);
  }

  float tx = Alpha_Side_Left_Param[0];
  float ty = -Alpha_Side_Left_Param[1];
  float tz = Alpha_Side_Left_Param[2];
  float rx = Alpha_Side_Left_Param[5] * PI_OVER_180;
  float ry = Alpha_Side_Left_Param[4] * PI_OVER_180;
  float rz = -Alpha_Side_Left_Param[3] * PI_OVER_180;

  Eigen::Affine3f mr = Eigen::Affine3f::Identity();

  mr.translation() << tx, ty, tz;
  mr.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));  // The angle of rotation in radians
  mr.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
  mr.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));
  pcl::PointCloud<pcl::PointXYZI>::Ptr output_Ptr = temp_array.makeShared();
  pcl::transformPointCloud(*output_Ptr, out_cloud, mr);  // no cuda

  for (int i = 0; i < out_cloud.points.size(); i++)
  {
    msgs::PointXYZV data;

    float x = out_cloud.points[i].x;
    float y = out_cloud.points[i].y;
    float z = out_cloud.points[i].z;

    data.x = x;
    data.y = y;
    data.z = z;
    data.speed = out_cloud.points[i].intensity;

    alphaSideLeftVec.push_back(data);
  }
}

void callbackAlphaSideRight(const msgs::Rad::ConstPtr& msg)
{
  alphaSideRightVec.clear();
  pcl::PointCloud<pcl::PointXYZI> temp_array;
  pcl::PointCloud<pcl::PointXYZI> out_cloud;

  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    pcl::PointXYZI point;

    float x = msg->radPoint[i].x;
    float y = msg->radPoint[i].y;
    float z = msg->radPoint[i].z;

    point.x = x;
    point.y = y;
    point.z = z;
    point.intensity = msg->radPoint[i].speed;

    temp_array.points.push_back(point);
  }

  float tx = Alpha_Side_Right_Param[0];
  float ty = -Alpha_Side_Right_Param[1];
  float tz = Alpha_Side_Right_Param[2];
  float rx = Alpha_Side_Right_Param[5] * PI_OVER_180;
  float ry = Alpha_Side_Right_Param[4] * PI_OVER_180;
  float rz = -Alpha_Side_Right_Param[3] * PI_OVER_180;

  Eigen::Affine3f mr = Eigen::Affine3f::Identity();

  mr.translation() << tx, ty, tz;
  mr.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));  // The angle of rotation in radians
  mr.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
  mr.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));
  pcl::PointCloud<pcl::PointXYZI>::Ptr output_Ptr = temp_array.makeShared();
  pcl::transformPointCloud(*output_Ptr, out_cloud, mr);  // no cuda

  for (int i = 0; i < out_cloud.points.size(); i++)
  {
    msgs::PointXYZV data;

    float x = out_cloud.points[i].x;
    float y = out_cloud.points[i].y;
    float z = out_cloud.points[i].z;

    data.x = x;
    data.y = y;
    data.z = z;
    data.speed = out_cloud.points[i].intensity;
    alphaSideRightVec.push_back(data);
  }
}

void callbackAlphaBackLeft(const msgs::Rad::ConstPtr& msg)
{
  alphaBackLeftVec.clear();
  pcl::PointCloud<pcl::PointXYZI> temp_array;
  pcl::PointCloud<pcl::PointXYZI> out_cloud;

  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    pcl::PointXYZI point;

    float x = msg->radPoint[i].x;
    float y = msg->radPoint[i].y;
    float z = msg->radPoint[i].z;

    point.x = x;
    point.y = y;
    point.z = z;
    point.intensity = msg->radPoint[i].speed;

    temp_array.points.push_back(point);
  }

  float tx = Alpha_Back_Left_Param[0];
  float ty = -Alpha_Back_Left_Param[1];
  float tz = Alpha_Back_Left_Param[2];
  float rx = Alpha_Back_Left_Param[5] * PI_OVER_180;
  float ry = Alpha_Back_Left_Param[4] * PI_OVER_180;
  float rz = -Alpha_Back_Left_Param[3] * PI_OVER_180;

  Eigen::Affine3f mr = Eigen::Affine3f::Identity();

  mr.translation() << tx, ty, tz;
  mr.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));  // The angle of rotation in radians
  mr.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
  mr.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));
  pcl::PointCloud<pcl::PointXYZI>::Ptr output_Ptr = temp_array.makeShared();
  pcl::transformPointCloud(*output_Ptr, out_cloud, mr);  // no cuda

  for (int i = 0; i < out_cloud.points.size(); i++)
  {
    msgs::PointXYZV data;

    float x = out_cloud.points[i].x;
    float y = out_cloud.points[i].y;
    float z = out_cloud.points[i].z;

    data.x = x;
    data.y = y;
    data.z = z;
    data.speed = out_cloud.points[i].intensity;

    alphaBackLeftVec.push_back(data);
  }
}

void callbackAlphaBackRight(const msgs::Rad::ConstPtr& msg)
{
  alphaBackRightVec.clear();
  pcl::PointCloud<pcl::PointXYZI> temp_array;
  pcl::PointCloud<pcl::PointXYZI> out_cloud;

  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    pcl::PointXYZI point;

    float x = msg->radPoint[i].x;
    float y = msg->radPoint[i].y;
    float z = msg->radPoint[i].z;

    point.x = x;
    point.y = y;
    point.z = z;
    point.intensity = msg->radPoint[i].speed;
    temp_array.points.push_back(point);
  }

  float tx = Alpha_Back_Right_Param[0];
  float ty = -Alpha_Back_Right_Param[1];
  float tz = Alpha_Back_Right_Param[2];
  float rx = Alpha_Back_Right_Param[5] * PI_OVER_180;
  float ry = Alpha_Back_Right_Param[4] * PI_OVER_180;
  float rz = -Alpha_Back_Right_Param[3] * PI_OVER_180;

  Eigen::Affine3f mr = Eigen::Affine3f::Identity();

  mr.translation() << tx, ty, tz;
  mr.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));  // The angle of rotation in radians
  mr.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
  mr.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));
  pcl::PointCloud<pcl::PointXYZI>::Ptr output_Ptr = temp_array.makeShared();
  pcl::transformPointCloud(*output_Ptr, out_cloud, mr);  // no cuda

  for (int i = 0; i < out_cloud.points.size(); i++)
  {
    msgs::PointXYZV data;

    float x = out_cloud.points[i].x;
    float y = out_cloud.points[i].y;
    float z = out_cloud.points[i].z;

    data.x = x;
    data.y = y;
    data.z = z;
    data.speed = out_cloud.points[i].intensity;

    alphaBackRightVec.push_back(data);
  }
}

void callbackIMU(const sensor_msgs::Imu::ConstPtr& input)
{
  imu_angular_velocity_z = input->angular_velocity.z;
  if (abs(imu_angular_velocity_z) > 0.05)
  {
    do_rotate = 1;
  }
  else
  {
    do_rotate = 0;
  }
}

void pointCalibration(float* x, float* y, float* z, int type)
{
  pcl::Normal pcl_normal(*x, *y, *z);
  Eigen::Vector4f input;
  input << pcl_normal.normal_x, pcl_normal.normal_y, pcl_normal.normal_z, 1;
  Eigen::Vector4f output_1;

  switch (type)
  {
    case 1:
      output_1 = frontCenterInitGuess * input;
      break;
    case 2:
      output_1 = frontLeftInitGuess * input;
      break;
    case 3:
      output_1 = frontRightInitGuess * input;
      break;
    case 4:
      output_1 = sideLeftInitGuess * input;
      break;
    case 5:
      output_1 = sideRightInitGuess * input;
      break;
    case 6:
      output_1 = backLeftInitGuess * input;
      break;
    case 7:
      output_1 = backRightInitGuess * input;
      break;
    default:
      break;
  }

  *x = output_1.x();
  *y = output_1.y();
  *z = output_1.z();
}

void transInitGuess(int type)
{
  float tx;
  float ty;
  float tz;
  float rx;
  float ry;
  float rz;

  switch (type)
  {
    case 1:
      tx = Alpha_Front_Center_Param[0];
      ty = Alpha_Front_Center_Param[1];
      tz = Alpha_Front_Center_Param[2];
      rx = Alpha_Front_Center_Param[3] * PI_OVER_180;
      ry = Alpha_Front_Center_Param[4] * PI_OVER_180;
      rz = Alpha_Front_Center_Param[5] * PI_OVER_180;
      break;
    case 2:
      tx = Alpha_Front_Left_Param[0];
      ty = Alpha_Front_Left_Param[1];
      tz = Alpha_Front_Left_Param[2];
      rx = Alpha_Front_Left_Param[3] * PI_OVER_180;
      ry = Alpha_Front_Left_Param[4] * PI_OVER_180;
      rz = Alpha_Front_Left_Param[5] * PI_OVER_180;
      break;
    case 3:
      tx = Alpha_Front_Right_Param[0];
      ty = Alpha_Front_Right_Param[1];
      tz = Alpha_Front_Right_Param[2];
      rx = Alpha_Front_Right_Param[3] * PI_OVER_180;
      ry = Alpha_Front_Right_Param[4] * PI_OVER_180;
      rz = Alpha_Front_Right_Param[5] * PI_OVER_180;
      break;
    case 4:
      tx = Alpha_Side_Left_Param[0];
      ty = Alpha_Side_Left_Param[1];
      tz = Alpha_Side_Left_Param[2];
      rx = Alpha_Side_Left_Param[3] * PI_OVER_180;
      ry = Alpha_Side_Left_Param[4] * PI_OVER_180;
      rz = Alpha_Side_Left_Param[5] * PI_OVER_180;
      break;
    case 5:
      tx = Alpha_Side_Right_Param[0];
      ty = Alpha_Side_Right_Param[1];
      tz = Alpha_Side_Right_Param[2];
      rx = Alpha_Side_Right_Param[3] * PI_OVER_180;
      ry = Alpha_Side_Right_Param[4] * PI_OVER_180;
      rz = Alpha_Side_Right_Param[5] * PI_OVER_180;
      break;
    case 6:
      tx = Alpha_Back_Left_Param[0];
      ty = Alpha_Back_Left_Param[1];
      tz = Alpha_Back_Left_Param[2];
      rx = Alpha_Back_Left_Param[3] * PI_OVER_180;
      ry = Alpha_Back_Left_Param[4] * PI_OVER_180;
      rz = Alpha_Back_Left_Param[5] * PI_OVER_180;
      break;
    case 7:
      tx = Alpha_Back_Right_Param[0];
      ty = Alpha_Back_Right_Param[1];
      tz = Alpha_Back_Right_Param[2];
      rx = Alpha_Back_Right_Param[3] * PI_OVER_180;
      ry = Alpha_Back_Right_Param[4] * PI_OVER_180;
      rz = Alpha_Back_Right_Param[5] * PI_OVER_180;
      break;
    default:
      break;
  }
  Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();

  Eigen::AngleAxisf init_rotation_x(rx, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf init_rotation_y(ry, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf init_rotation_z(rz, Eigen::Vector3f::UnitZ());

  Eigen::Translation3f init_translation(tx, ty, tz);

  switch (type)
  {
    case 1:
      frontCenterInitGuess = (init_translation * init_rotation_x * init_rotation_y * init_rotation_z).matrix();
      break;
    case 2:
      frontLeftInitGuess = (init_translation * init_rotation_x * init_rotation_y * init_rotation_z).matrix();
      break;
    case 3:
      frontRightInitGuess = (init_translation * init_rotation_x * init_rotation_y * init_rotation_z).matrix();
      break;
    case 4:
      sideLeftInitGuess = (init_translation * init_rotation_x * init_rotation_y * init_rotation_z).matrix();
      break;
    case 5:
      sideRightInitGuess = (init_translation * init_rotation_x * init_rotation_y * init_rotation_z).matrix();
      break;
    case 6:
      backLeftInitGuess = (init_translation * init_rotation_x * init_rotation_y * init_rotation_z).matrix();
      break;
    case 7:
      backRightInitGuess = (init_translation * init_rotation_x * init_rotation_y * init_rotation_z).matrix();
      break;
    default:
      break;
  }
}

void msgPublisher()
{
  std_msgs::Empty empty_msg;
  HeartbeatPub.publish(empty_msg);
}

void alphaRadPub()
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointXYZI temp;

  alphaAllVec.clear();
  for (int i = 0; i < alphaFrontCenterVec.size(); i++)
  {
    alphaAllVec.push_back(alphaFrontCenterVec[i]);
  }
  alphaFrontCenterVec.clear();

  for (int i = 0; i < alphaFrontLeftVec.size(); i++)
  {
    alphaAllVec.push_back(alphaFrontLeftVec[i]);
  }
  alphaFrontLeftVec.clear();

  for (int i = 0; i < alphaFrontRightVec.size(); i++)
  {
    alphaAllVec.push_back(alphaFrontRightVec[i]);
  }
  alphaFrontRightVec.clear();

  for (int i = 0; i < alphaSideLeftVec.size(); i++)
  {
    alphaAllVec.push_back(alphaSideLeftVec[i]);
  }
  alphaSideLeftVec.clear();

  for (int i = 0; i < alphaSideRightVec.size(); i++)
  {
    alphaAllVec.push_back(alphaSideRightVec[i]);
  }
  alphaSideRightVec.clear();

  for (int i = 0; i < alphaBackLeftVec.size(); i++)
  {
    alphaAllVec.push_back(alphaBackLeftVec[i]);
  }
  alphaBackLeftVec.clear();

  for (int i = 0; i < alphaBackRightVec.size(); i++)
  {
    alphaAllVec.push_back(alphaBackRightVec[i]);
  }
  alphaBackRightVec.clear();

  for (int i = 0; i < alphaAllVec.size(); i++)
  {
    float x = alphaAllVec[i].x;
    float y = abs(alphaAllVec[i].y);

    if ((x < 0.6) && (x > -6))
    {
      if (y < 1.2)
      {
        continue;
      }
    }

    {
      // for rviz drawing
      temp.x = alphaAllVec[i].x;
      temp.y = -alphaAllVec[i].y;
      cloud->points.push_back(temp);
      if (alpha_raw_message)
      {
        cout << "X: " << temp.x << ", Y: " << temp.y << endl;
      }
      alphaRad.radPoint.push_back(alphaAllVec[i]);
    }
  }
  if (alpha_raw_message)
  {
    cout << "========================================" << endl;
  }

  sensor_msgs::PointCloud2 msgtemp;
  pcl::toROSMsg(*cloud, msgtemp);
  msgtemp.header = alphaRad.radHeader;
  msgtemp.header.seq = alphaRad.radHeader.seq;
  msgtemp.header.frame_id = "radar_alpha";
  RadAlphaPCLPub.publish(msgtemp);

  RadAlphaPub.publish(alphaRad);

  if (debug_message)
  {
    std::cout << "Alpha Radar Data : " << alphaRad.radPoint.size() << std::endl;
  }

  alphaRad.radPoint.clear();
}

void cubtekRadPub()
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointXYZI temp;

  for (int i = 0; i < cubtekVec.size(); i++)
  {
    // for rviz drawing
    temp.x = cubtekVec[i].x;
    temp.y = -cubtekVec[i].y;
    cloud->points.push_back(temp);
    if (cubtek_raw_message)
    {
      cout << "Cubtek X: " << temp.x << ", Y: " << temp.y << endl;
    }
    cubtekRad.radPoint.push_back(cubtekVec[i]);
  }
  if (cubtek_raw_message)
  {
    cout << "========================================" << endl;
  }

  sensor_msgs::PointCloud2 msgtemp;
  pcl::toROSMsg(*cloud, msgtemp);
  msgtemp.header = cubtekRad.radHeader;
  msgtemp.header.seq = cubtekRad.radHeader.seq;
  msgtemp.header.frame_id = "radar_cubtek";
  RadCubtekPCLPub.publish(msgtemp);

  RadCubtekPub.publish(cubtekRad);

  if (debug_message)
  {
    std::cout << "Cubtek Radar Data : " << cubtekRad.radPoint.size() << std::endl;
  }

  cubtekRad.radPoint.clear();
}

void onInit(ros::NodeHandle nh, ros::NodeHandle n)
{
  nh.param("/debug_message", debug_message, 0);
  nh.param("/delphi_raw_message", delphi_raw_message, 0);
  nh.param("/cubtek_raw_message", cubtek_raw_message, 0);
  nh.param("/alpha_raw_message", alpha_raw_message, 0);

  if (!ros::param::has("/Alpha_Front_Center_Param"))
  {
    nh.setParam("Alpha_Front_Center_Param", Zero_Param);
    nh.setParam("Alpha_Front_Left_Param", Zero_Param);
    nh.setParam("Alpha_Front_Right_Param", Zero_Param);
    nh.setParam("Alpha_Side_Left_Param", Zero_Param);
    nh.setParam("Alpha_Side_Right_Param", Zero_Param);
    nh.setParam("Alpha_Back_Left_Param", Zero_Param);
    nh.setParam("Alpha_Back_Right_Param", Zero_Param);
    nh.setParam("Cubtek_Front_Center_Param", Zero_Param);
    cout << "NO STITCHING PARAMETER INPUT!" << endl;
    cout << "Now is using [0,0,0,0,0,0] as stitching parameter!" << endl;
  }
  else
  {
    nh.param("/Alpha_Front_Center_Param", Alpha_Front_Center_Param, vector<float>());
    nh.param("/Alpha_Front_Left_Param", Alpha_Front_Left_Param, vector<float>());
    nh.param("/Alpha_Front_Right_Param", Alpha_Front_Right_Param, vector<float>());
    nh.param("/Alpha_Side_Left_Param", Alpha_Side_Left_Param, vector<float>());
    nh.param("/Alpha_Side_Right_Param", Alpha_Side_Right_Param, vector<float>());
    nh.param("/Alpha_Back_Left_Param", Alpha_Back_Left_Param, vector<float>());
    nh.param("/Alpha_Back_Right_Param", Alpha_Back_Right_Param, vector<float>());
    nh.param("/Cubtek_Front_Center_Param", Cubtek_Front_Center_Param, vector<float>());
    cout << "STITCHING PARAMETER FIND!" << endl;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "RadAll");

  ros::NodeHandle nh("~");
  ros::NodeHandle n;
  ros::Subscriber DelphiFrontSub = n.subscribe("DelphiFront", 1, callbackDelphiFront);
  ros::Subscriber AlphiFrontCenterSub = n.subscribe("AlphaFrontCenter", 1, callbackAlphaFrontCenter);
  ros::Subscriber AlphiFrontLeftSub = n.subscribe("AlphaFrontLeft", 1, callbackAlphaFrontLeft);
  ros::Subscriber AlphiFrontRightSub = n.subscribe("AlphaFrontRight", 1, callbackAlphaFrontRight);
  ros::Subscriber AlphiSideLeftSub = n.subscribe("AlphaSideLeft", 1, callbackAlphaSideLeft);
  ros::Subscriber AlphiSideRightSub = n.subscribe("AlphaSideRight", 1, callbackAlphaSideRight);
  ros::Subscriber AlphiBackLeftSub = n.subscribe("AlphaBackLeft", 1, callbackAlphaBackLeft);
  ros::Subscriber AlphiBackRightSub = n.subscribe("AlphaBackRight", 1, callbackAlphaBackRight);
  ros::Subscriber CubtekFrontSub = n.subscribe("CubtekFront", 1, callbackCubtekFront);

  ros::Subscriber IMURadSub = n.subscribe("imu_data_rad", 1, callbackIMU);

  RadFrontPub = n.advertise<msgs::Rad>("RadFront", 1);

  RadAlphaPub = n.advertise<msgs::Rad>("RadAlpha", 1);
  RadAlphaPCLPub = n.advertise<sensor_msgs::PointCloud2>("RadAlphaPCL", 1);

  RadCubtekPub = n.advertise<msgs::Rad>("RadCubtek", 1);
  RadCubtekPCLPub = n.advertise<sensor_msgs::PointCloud2>("RadCubtekPCL", 1);

  HeartbeatPub = n.advertise<std_msgs::Empty>("RadFront/heartbeat", 1);

  onInit(nh, n);

  for (int i = 1; i < 8; i++)
  {
    transInitGuess(i);
  }

  ros::Rate rate(20);
  while (ros::ok())
  {
    print_count++;
    alphaRadPub();
    cubtekRadPub();

    if (print_count > 60)
    {
      // ===============for test code start
      // float a = 5;
      // float b = 5;
      // float c = 5;
      // pointCalibration(&a, &b, &c, 1);
      // cout << a << ":" << b << ":" << c << endl;
      // ===============for test code end

      cout << "=== Radar Detection === A : " << alphaRad.radPoint.size() << " D : " << delphiRad.radPoint.size()
           << endl;
      print_count = 0;
    }
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}
