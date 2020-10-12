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
void callbackAlphaFront(const msgs::Rad::ConstPtr& msg);
void callbackIMU(const sensor_msgs::Imu::ConstPtr& input);
void pointCalibration(float* x, float* y, float* z, int type);
void msgPublisher();

ros::Publisher RadFrontPub;
ros::Publisher HeartbeatPub;

double imu_angular_velocity_z = 0;
int do_rotate = 0;
int print_count = 0;

msgs::Rad delphiRad;

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

      // debug msg
      cout << "X: " << point.x << ", Y: " << point.y << ", Speed: " << point.speed << endl;

      delphiRad.radPoint.push_back(point);
    }
  }
  std::cout << "Radar Data : " << delphiRad.radPoint.size() << std::endl;
  delphiRad.radHeader.stamp = msg->radHeader.stamp;
  delphiRad.radHeader.seq = msg->radHeader.seq;
  RadFrontPub.publish(delphiRad);
  delphiRad.radPoint.clear();
  msgPublisher();
}

void callbackAlphaFront(const msgs::Rad::ConstPtr& msg)
{
  // y: 往前 , x: 右正
  // 1: front center, 2: front left, 3: front right,
  // 4: side left, 5: side right,
  // 6: back left, 7: back right
  //
  //            2__1__3
  //            4|   |5
  //             |   |
  //             |   |
  //             |   |
  //             |   |
  //            6|___|7
  //
  //
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
  std::vector<float> params = { 0, -10, 0, 0.0, 0.0, 0.0 };  // (x,y,z,roll,pitch,yaw)

  pcl::Normal pcl_normal(*x, *y, *z);
	Eigen::Vector4f input;
	input << pcl_normal.normal_x, pcl_normal.normal_y, pcl_normal.normal_z, 1;

  Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
  float tx = params[0];
  float ty = params[1];
  float tz = params[2];
  float rx = params[3] * PI_OVER_180;
  float ry = params[4] * PI_OVER_180;
  float rz = params[5] * PI_OVER_180;

  Eigen::AngleAxisf init_rotation_x(rx, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf init_rotation_y(ry, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf init_rotation_z(rz, Eigen::Vector3f::UnitZ());

  Eigen::Translation3f init_translation(tx, ty, tz);

  Eigen::Matrix4f init_guess = (init_translation * init_rotation_x * init_rotation_y * init_rotation_z).matrix();

  transform_1 = init_guess;
  // std::cout << "1:" << std::endl << transform_1.matrix() << std::endl;
  Eigen::Vector4f output_1 = transform_1 * input;
  // std::cout << std::endl << output_1 << std::endl;
  *x =  output_1.x();
  *y =  output_1.y();
  *z =  output_1.z();
}

void msgPublisher()
{
  std_msgs::Empty empty_msg;
  HeartbeatPub.publish(empty_msg);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "RadAll");

  ros::NodeHandle nh("~");
  ros::NodeHandle n;
  ros::Subscriber DelphiFrontSub = n.subscribe("DelphiFront", 1, callbackDelphiFront);
  // ros::Subscriber AlphiFrontSub = n.subscribe("AlphaFrontCenter", 1, callbackAlphaFront);
  ros::Subscriber IMURadSub = n.subscribe("imu_data_rad", 1, callbackIMU);

  RadFrontPub = n.advertise<msgs::Rad>("RadFront", 1);
  HeartbeatPub = n.advertise<std_msgs::Empty>("RadFront/heartbeat", 1);

  ros::Rate rate(20);
  while (ros::ok())
  {
    print_count++;
    if (print_count > 60)
    {
      // float a = 5;
      // float b = 5;
      // float c = 5;
      // alphaRadAlignment(&a, &b, &c);

      // cout << a << ":" << b << ":" << c << endl;
      std::cout << "================ Radar Detection ================" << std::endl;
      print_count = 0;
    }
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}
