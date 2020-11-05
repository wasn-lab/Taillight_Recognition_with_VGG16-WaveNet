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
#include "msgs/LocalizationToVeh.h"
#include <cstring>
#include <visualization_msgs/Marker.h>

// For PCL
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

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
void callbackIMU(const sensor_msgs::Imu::ConstPtr& input);
void pointCalibration(float* x, float* y, float* z, int type);
void onInit(ros::NodeHandle nh, ros::NodeHandle n);
void transInitGuess(int type);
void msgPublisher();

ros::Publisher RadFrontPub;
ros::Publisher RadAllPub;
ros::Publisher RadAllPCLPub;
ros::Publisher HeartbeatPub;

double imu_angular_velocity_z = 0;
int do_rotate = 0;
int print_count = 0;
int debug_message = 0;
int raw_message = 0;

vector<float> Alpha_Front_Center_Param;
vector<float> Alpha_Front_Left_Param;
vector<float> Alpha_Front_Right_Param;
vector<float> Alpha_Side_Left_Param;
vector<float> Alpha_Side_Right_Param;
vector<float> Alpha_Back_Left_Param;
vector<float> Alpha_Back_Right_Param;
vector<float> Zero_Param(6, 0.0);

msgs::Rad delphiRad;
msgs::Rad alphaRad;

vector<msgs::PointXYZV> alphaAllVec;
vector<msgs::PointXYZV> alphaFrontCenterVec;
vector<msgs::PointXYZV> alphaFrontLeftVec;
vector<msgs::PointXYZV> alphaFrontRightVec;
vector<msgs::PointXYZV> alphaSideLeftVec;
vector<msgs::PointXYZV> alphaSideRightVec;
vector<msgs::PointXYZV> alphaBackLeftVec;
vector<msgs::PointXYZV> alphaBackRightVec;

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
      // 直線先不做判斷，先讓AEB work
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

      if (raw_message)
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

void callbackAlphaFrontCenter(const msgs::Rad::ConstPtr& msg)
{
  alphaRad.radHeader.stamp = msg->radHeader.stamp;
  alphaRad.radHeader.seq = msg->radHeader.seq;

  alphaFrontCenterVec.clear();
  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    msgs::PointXYZV point;

    float x = msg->radPoint[i].x;
    float y = msg->radPoint[i].y;
    float z = msg->radPoint[i].z;

    // cout << "ox : " << x << " oy : " << y << " oz : " << z << endl;

    pointCalibration(&x, &y, &z, 1);

    // cout << "tx : " << x << " ty : " << y << " tz : " << z << endl;

    point.x = x;
    point.y = y;
    point.z = z;
    point.speed = msg->radPoint[i].speed;

    alphaFrontCenterVec.push_back(point);
  }
  // cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
}

void callbackAlphaFrontLeft(const msgs::Rad::ConstPtr& msg)
{
  alphaFrontLeftVec.clear();
  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    msgs::PointXYZV point;

    float x = msg->radPoint[i].x;
    float y = msg->radPoint[i].y;
    float z = msg->radPoint[i].z;

    pointCalibration(&x, &y, &z, 2);

    point.x = x;
    point.y = y;
    point.z = z;
    point.speed = msg->radPoint[i].speed;

    alphaFrontLeftVec.push_back(point);
  }
}

void callbackAlphaFrontRight(const msgs::Rad::ConstPtr& msg)
{
  alphaFrontRightVec.clear();
  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    msgs::PointXYZV point;

    float x = msg->radPoint[i].x;
    float y = msg->radPoint[i].y;
    float z = msg->radPoint[i].z;

    pointCalibration(&x, &y, &z, 3);

    point.x = x;
    point.y = y;
    point.z = z;
    point.speed = msg->radPoint[i].speed;

    alphaFrontRightVec.push_back(point);
  }
}
void callbackAlphaSideLeft(const msgs::Rad::ConstPtr& msg)
{
  alphaSideLeftVec.clear();
  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    msgs::PointXYZV point;

    float x = msg->radPoint[i].x;
    float y = msg->radPoint[i].y;
    float z = msg->radPoint[i].z;

    pointCalibration(&x, &y, &z, 4);

    point.x = x;
    point.y = y;
    point.z = z;
    point.speed = msg->radPoint[i].speed;

    alphaSideLeftVec.push_back(point);
  }
}

void callbackAlphaSideRight(const msgs::Rad::ConstPtr& msg)
{
  alphaSideRightVec.clear();
  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    msgs::PointXYZV point;

    float x = msg->radPoint[i].x;
    float y = msg->radPoint[i].y;
    float z = msg->radPoint[i].z;

    pointCalibration(&x, &y, &z, 5);

    point.x = x;
    point.y = y;
    point.z = z;
    point.speed = msg->radPoint[i].speed;

    alphaSideRightVec.push_back(point);
  }
}
void callbackAlphaBackLeft(const msgs::Rad::ConstPtr& msg)
{
  alphaBackLeftVec.clear();
  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    msgs::PointXYZV point;

    float x = msg->radPoint[i].x;
    float y = msg->radPoint[i].y;
    float z = msg->radPoint[i].z;

    pointCalibration(&x, &y, &z, 6);

    point.x = x;
    point.y = y;
    point.z = z;
    point.speed = msg->radPoint[i].speed;

    alphaBackLeftVec.push_back(point);
  }
}

void callbackAlphaBackRight(const msgs::Rad::ConstPtr& msg)
{
  alphaBackRightVec.clear();
  for (int i = 0; i < msg->radPoint.size(); i++)
  {
    msgs::PointXYZV point;

    float x = msg->radPoint[i].x;
    float y = msg->radPoint[i].y;
    float z = msg->radPoint[i].z;

    pointCalibration(&x, &y, &z, 7);

    point.x = x;
    point.y = y;
    point.z = z;
    point.speed = msg->radPoint[i].speed;

    alphaBackRightVec.push_back(point);
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
  // for(auto n : params) {
  //   cout << n << endl;
  // }

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

  for (int i = 0; i < alphaFrontLeftVec.size(); i++)
  {
    alphaAllVec.push_back(alphaFrontLeftVec[i]);
  }

  for (int i = 0; i < alphaFrontRightVec.size(); i++)
  {
    alphaAllVec.push_back(alphaFrontRightVec[i]);
  }

  for (int i = 0; i < alphaSideLeftVec.size(); i++)
  {
    alphaAllVec.push_back(alphaSideLeftVec[i]);
  }

  for (int i = 0; i < alphaSideRightVec.size(); i++)
  {
    alphaAllVec.push_back(alphaSideRightVec[i]);
  }

  for (int i = 0; i < alphaBackLeftVec.size(); i++)
  {
    alphaAllVec.push_back(alphaBackLeftVec[i]);
  }

  for (int i = 0; i < alphaBackRightVec.size(); i++)
  {
    alphaAllVec.push_back(alphaBackRightVec[i]);
  }

  for (int i = 0; i < alphaAllVec.size(); i++)
  {
    alphaRad.radPoint.push_back(alphaAllVec[i]);

    // for rviz drawing
    temp.x = alphaAllVec[i].x;
    temp.y = -alphaAllVec[i].y;
    cloud->points.push_back(temp);
  }

  sensor_msgs::PointCloud2 msgtemp;
  pcl::toROSMsg(*cloud, msgtemp);
  msgtemp.header = alphaRad.radHeader;
  msgtemp.header.seq = alphaRad.radHeader.seq;
  msgtemp.header.frame_id = "radar_alpha";
  RadAllPCLPub.publish(msgtemp);

  RadAllPub.publish(alphaRad);

  if (debug_message)
  {
    std::cout << "Alpha Radar Data : " << alphaRad.radPoint.size() << std::endl;
  }

  alphaRad.radPoint.clear();

  // msgPublisher();
}

void onInit(ros::NodeHandle nh, ros::NodeHandle n)
{
  nh.param("/debug_message", debug_message, 0);
  nh.param("/raw_message", raw_message, 0);

  if (!ros::param::has("/Alpha_Front_Center_Param"))
  {
    nh.setParam("Alpha_Front_Center_Param", Zero_Param);
    nh.setParam("Alpha_Front_Left_Param", Zero_Param);
    nh.setParam("Alpha_Front_Right_Param", Zero_Param);
    nh.setParam("Alpha_Side_Left_Param", Zero_Param);
    nh.setParam("Alpha_Side_Right_Param", Zero_Param);
    nh.setParam("Alpha_Back_Left_Param", Zero_Param);
    nh.setParam("Alpha_Back_Right_Param", Zero_Param);
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

  ros::Subscriber IMURadSub = n.subscribe("imu_data_rad", 1, callbackIMU);

  RadFrontPub = n.advertise<msgs::Rad>("RadFront", 1);
  RadAllPub = n.advertise<msgs::Rad>("RadAll", 1);
  RadAllPCLPub = n.advertise<sensor_msgs::PointCloud2>("RadAlphaPCL", 1);
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
