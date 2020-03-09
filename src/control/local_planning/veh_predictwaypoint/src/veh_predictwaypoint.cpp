#include <ros/ros.h>
#include <autoware_msgs/Lane.h>
#include <autoware_msgs/Waypoint.h> 
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <ros/package.h>
#include <fstream>
#include <nav_msgs/Path.h>
#include <math.h>
#include "msgs/VehInfo.h"
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PolygonStamped.h>
#include <std_msgs/Bool.h>

ros::Publisher veh_predictpath_pub;
ros::Publisher veh_predictpath_rel_pub;
ros::Publisher vehbb_pub;

#define RT_PI 3.14159265358979323846

nav_msgs::Path veh_predictpath;
nav_msgs::Path astar_finalpath_10;

double speed_mps = 0;
double angular_vz = 0;
double veh_width = 2.42;
double veh_length = 7.0;
double veh_pose_left = 0.5;
double veh_pose_front = 0.615;
double wheel_dis = 3.8;
double predict_s = 10 + wheel_dis;
double Resolution = 50;
double out_dis = 0.5;

bool astarpath_ini = false;
bool vehinfo_ini = false;
bool imudata_ini = false;
bool rearcurrentpose_ini = false;

struct BB
{
  double frontleft_point_x;
  double frontleft_point_y;
  double frontleft_point_z;
  double frontright_point_x;
  double frontright_point_y;
  double frontright_point_z;
  double rearleft_point_x;
  double rearleft_point_y;
  double rearleft_point_z;
  double rearright_point_x;
  double rearright_point_y;
  double rearright_point_z;
};

BB veh_bb;

struct pose
{
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
};

pose rear_pose;

void outofpathcheck(geometry_msgs::PolygonStamped veh_poly, bool& flag)
{
  if (astarpath_ini)
  { 
    double min_value[4] = {100,100,100,100};
    int size = astar_finalpath_10.poses.size();
    // std::cout << "size : " << size << std::endl;
    for (int i = 0; i < 4; i++)
    {
      double V_X = veh_poly.polygon.points[i].x;
      double V_Y = veh_poly.polygon.points[i].y;
      for (int j = 0; j < size; j++)
      {
        double P_X = astar_finalpath_10.poses[j].pose.position.x;
        double P_Y = astar_finalpath_10.poses[j].pose.position.y;
        double distance = std::sqrt((V_X-P_X)*(V_X-P_X) + (V_Y-P_Y)*(V_Y-P_Y));
        if (distance < min_value[i])
        {
          min_value[i] = distance;
        }
      }
    }
    
    if (min_value[0] > (1.2 + out_dis) && min_value[0] != 100)
    {
      flag = true;
    }
    if (min_value[1] > (1.2 + out_dis) && min_value[1] != 100)
    {
      flag = true;
    }
    double dis_RR2RC = std::sqrt(((veh_length*(1-veh_pose_front))*(veh_length*(1-veh_pose_front)) + veh_width*(1-veh_pose_left) * veh_width*(1-veh_pose_left)));
    // std::cout << "dis_RR2RC : " << dis_RR2RC << ", min_value[2] : " << min_value[2] << std::endl;
    if (min_value[2] > (dis_RR2RC + out_dis) && min_value[2] != 100)
    {
      flag = true;
    }
    double dis_LR2RC = std::sqrt(((veh_length*(1-veh_pose_front))*(veh_length*(1-veh_pose_front)) + veh_width*veh_pose_left * veh_width*veh_pose_left));
    if (min_value[3] > (dis_LR2RC + out_dis) && min_value[3] != 100)
    {
      flag = true;
    }
  } 
}

void vehpredictpathgen_pub(bool flag)
{
  // rviz path
  nav_msgs::Path Dpath;
  geometry_msgs::PoseStamped Dpose;
  Dpath.header.frame_id = "map";
  Dpose.header.frame_id = "map";

  // rviz rel path
  nav_msgs::Path Relpath;
  geometry_msgs::PoseStamped Relpose;
  Relpath.header.frame_id = "rear_wheel";
  Relpose.header.frame_id = "rear_wheel";

  if (flag && rearcurrentpose_ini && vehinfo_ini && imudata_ini)
  {
    double yaw_rate =  -angular_vz * RT_PI / 180.0;
    double r = speed_mps / yaw_rate;
    if (yaw_rate > 0.005 || yaw_rate < -0.005)
    {
      if (r > 0.1)
      {
        for (int i = 0; i < predict_s * 100; i++)
        {
          double theta_t = double(i/100) / r;
          Relpose.pose.position.x = r * std::sin(theta_t);
          if (theta_t < RT_PI/2.0 || theta_t > 3*RT_PI/2.0)
          {
            Relpose.pose.position.y = -std::sqrt(r*r - Relpose.pose.position.x*Relpose.pose.position.x) + r;
          }
          else
          {
            Relpose.pose.position.y = std::sqrt(r*r - Relpose.pose.position.x*Relpose.pose.position.x) + r;
          }
          Relpose.pose.position.z = -3;
          Relpath.poses.push_back(Relpose);

          Dpose.pose.position.x = rear_pose.x + Relpose.pose.position.x * std::cos(rear_pose.yaw) - Relpose.pose.position.y * std::sin(rear_pose.yaw);
          Dpose.pose.position.y = rear_pose.y + Relpose.pose.position.x * std::sin(rear_pose.yaw) + Relpose.pose.position.y * std::cos(rear_pose.yaw);
          Dpose.pose.position.z = rear_pose.z + Relpose.pose.position.z;
          Dpath.poses.push_back(Dpose);
        }
        // std::cout << "1111111111111111" << std::endl;
      }
      else if (r < -0.1)
      {
        for (int i = 0; i < predict_s * 100; i++)
        {
          double theta_t = double(i/100) / r;
          Relpose.pose.position.x = r * std::sin(theta_t);
          if (theta_t < RT_PI/2.0 || theta_t > 3*RT_PI/2.0)
          {
            Relpose.pose.position.y = std::sqrt(r*r - Relpose.pose.position.x*Relpose.pose.position.x) + r;
          }
          else
          {
            Relpose.pose.position.y = -std::sqrt(r*r - Relpose.pose.position.x*Relpose.pose.position.x) + r;
          }
          Relpose.pose.position.z = -3;
          Relpath.poses.push_back(Relpose);

          Dpose.pose.position.x = rear_pose.x + Relpose.pose.position.x * std::cos(rear_pose.yaw) - Relpose.pose.position.y * std::sin(rear_pose.yaw);
          Dpose.pose.position.y = rear_pose.y + Relpose.pose.position.x * std::sin(rear_pose.yaw) + Relpose.pose.position.y * std::cos(rear_pose.yaw);
          Dpose.pose.position.z = rear_pose.z + Relpose.pose.position.z;
          Dpath.poses.push_back(Dpose);
        }
        // std::cout << "22222222222222222" << std::endl;
      }
      else
      {
        for (int i = 0; i < predict_s * 100; i++)
        {
          Relpose.pose.position.x = double(i/100);
          Relpose.pose.position.y = 0;
          Relpose.pose.position.z = -3;
          Relpath.poses.push_back(Relpose);

          Dpose.pose.position.x = rear_pose.x + Relpose.pose.position.x * std::cos(rear_pose.yaw) - Relpose.pose.position.y * std::sin(rear_pose.yaw);
          Dpose.pose.position.y = rear_pose.y + Relpose.pose.position.x * std::sin(rear_pose.yaw) + Relpose.pose.position.y * std::cos(rear_pose.yaw);
          Dpose.pose.position.z = rear_pose.z + Relpose.pose.position.z;
          Dpath.poses.push_back(Dpose);
        }
        // std::cout << "333333333333333333" << std::endl;
      }
    }
    else
    {
      for (int i = 0; i < predict_s * 100; i++)
      {
        Relpose.pose.position.x = double(i/100);
        Relpose.pose.position.y = 0;
        Relpose.pose.position.z = -3;
        Relpath.poses.push_back(Relpose);

        Dpose.pose.position.x = rear_pose.x + Relpose.pose.position.x * std::cos(rear_pose.yaw) - Relpose.pose.position.y * std::sin(rear_pose.yaw);
        Dpose.pose.position.y = rear_pose.y + Relpose.pose.position.x * std::sin(rear_pose.yaw) + Relpose.pose.position.y * std::cos(rear_pose.yaw);
        Dpose.pose.position.z = rear_pose.z + Relpose.pose.position.z;
        Dpath.poses.push_back(Dpose);
      }
      // std::cout << "4444444444444444444" << std::endl;
    }
    veh_predictpath_rel_pub.publish(Relpath);
    veh_predictpath_pub.publish(Dpath);
    std::cout << "publish" << std::endl;
  }
  // else if (astarpath_ini && vehinfo_ini)
  // {
  //   double max_i = (speed_mps * predict_t) * Resolution;
  //   for (int i = 0; i < max_i; i++)
  //   {
  //     Dpose = astar_finalpath_10.poses[i];
  //     Dpath.poses.push_back(Dpose);
  //   }
  //   std::cout << "55555555555555555555" << std::endl;
  //   veh_predictpath_pub.publish(Dpath);
  // }
  else
  {
    Relpose.pose.position.x = 0;
    Relpose.pose.position.y = 0;
    Relpose.pose.position.z = -3;
    Relpath.poses.push_back(Relpose);

    Dpose.pose.position.x = rear_pose.x + Relpose.pose.position.x;
    Dpose.pose.position.y = rear_pose.y + Relpose.pose.position.y;
    Dpose.pose.position.z = rear_pose.z + Relpose.pose.position.z;
    Dpath.poses.push_back(Dpose);
  }
  veh_predictpath_rel_pub.publish(Relpath);
  veh_predictpath_pub.publish(Dpath);
}

void RearCurrentPoseCallback(const geometry_msgs::PoseStamped& RCPmsg)
{
  double roll, pitch, yaw;
  tf::Quaternion lidar_q(RCPmsg.pose.orientation.x, RCPmsg.pose.orientation.y, RCPmsg.pose.orientation.z,RCPmsg.pose.orientation.w);
  tf::Matrix3x3 lidar_m(lidar_q);
  lidar_m.getRPY(roll, pitch, yaw);

  rear_pose.x = RCPmsg.pose.position.x;
  rear_pose.y = RCPmsg.pose.position.y;
  rear_pose.z = RCPmsg.pose.position.z;
  rear_pose.yaw = yaw;
  rear_pose.roll = roll;
  rear_pose.pitch = pitch;

  rearcurrentpose_ini = true;

  double CY = std::cos(yaw);
  double SY = std::sin(yaw);
  // 
  veh_bb.frontleft_point_x = RCPmsg.pose.position.x + veh_length*veh_pose_front*CY - veh_width*veh_pose_left*SY;
  veh_bb.frontleft_point_y = RCPmsg.pose.position.y + veh_length*veh_pose_front*SY + veh_width*veh_pose_left*CY;
  veh_bb.frontleft_point_z = RCPmsg.pose.position.z - 3;
  veh_bb.frontright_point_x = RCPmsg.pose.position.x + veh_length*veh_pose_front*CY - (-veh_width)*(1-veh_pose_left)*SY;
  veh_bb.frontright_point_y = RCPmsg.pose.position.y + veh_length*veh_pose_front*SY + (-veh_width)*(1-veh_pose_left)*CY;
  veh_bb.frontright_point_z = RCPmsg.pose.position.z - 3;
  veh_bb.rearleft_point_x = RCPmsg.pose.position.x + (-veh_length)*(1-veh_pose_front)*CY - veh_width*veh_pose_left*SY;
  veh_bb.rearleft_point_y = RCPmsg.pose.position.y + (-veh_length)*(1-veh_pose_front)*SY + veh_width*veh_pose_left*CY;
  veh_bb.rearleft_point_z = RCPmsg.pose.position.z - 3;
  veh_bb.rearright_point_x = RCPmsg.pose.position.x + (-veh_length)*(1-veh_pose_front)*CY - (-veh_width)*(1-veh_pose_left)*SY;
  veh_bb.rearright_point_y = RCPmsg.pose.position.y + (-veh_length)*(1-veh_pose_front)*SY + (-veh_width)*(1-veh_pose_left)*CY;
  veh_bb.rearright_point_z = RCPmsg.pose.position.z - 3;

  geometry_msgs::PolygonStamped polygon0;
  polygon0.header.frame_id = "map";
  geometry_msgs::Point32 point0;
  point0.x = veh_bb.frontleft_point_x;
  point0.y = veh_bb.frontleft_point_y;
  point0.z = veh_bb.frontleft_point_z;
  polygon0.polygon.points.push_back(point0);
  point0.x = veh_bb.frontright_point_x;
  point0.y = veh_bb.frontright_point_y;
  point0.z = veh_bb.frontright_point_z;
  polygon0.polygon.points.push_back(point0);
  point0.x = veh_bb.rearright_point_x;
  point0.y = veh_bb.rearright_point_y;
  point0.z = veh_bb.rearright_point_z;
  polygon0.polygon.points.push_back(point0);
  point0.x = veh_bb.rearleft_point_x;
  point0.y = veh_bb.rearleft_point_y;
  point0.z = veh_bb.rearleft_point_z;
  polygon0.polygon.points.push_back(point0);
  vehbb_pub.publish(polygon0);

  bool outofpathcheck_flag = false;
  outofpathcheck(polygon0,outofpathcheck_flag);
  std::cout << "outofpathcheck_flag : " << outofpathcheck_flag << std::endl;
  vehpredictpathgen_pub(outofpathcheck_flag);
}

void vehinfoCallback(const msgs::VehInfo::ConstPtr& VImsg)
{
  speed_mps = VImsg->ego_speed;
  vehinfo_ini = true;
  // std::cout << "vehinfo_ini : " << vehinfo_ini << std::endl;
}

void imudataCallback(const sensor_msgs::Imu::ConstPtr& imumsg)
{
  angular_vz = imumsg->angular_velocity.z;
  imudata_ini = true;
  // std::cout << "imudata_ini : " << imudata_ini << std::endl;
}

void astarpathfinalCallback(const nav_msgs::Path& apfmsg)
{
  nav_msgs::Path astar_finalpath;
  geometry_msgs::PoseStamped Dpose;

  uint size = apfmsg.poses.size(); 
  for(uint i=1;i<size;i++)
  {
    for(int j=0;j<Resolution;j++)
    {
      Dpose.pose.position.x = apfmsg.poses[i-1].pose.position.x + j*(1/Resolution)*(apfmsg.poses[i].pose.position.x - apfmsg.poses[i-1].pose.position.x);
      Dpose.pose.position.y = apfmsg.poses[i-1].pose.position.y + j*(1/Resolution)*(apfmsg.poses[i].pose.position.y - apfmsg.poses[i-1].pose.position.y);
      Dpose.pose.position.z = apfmsg.poses[i-1].pose.position.z + j*(1/Resolution)*(apfmsg.poses[i].pose.position.z - apfmsg.poses[i-1].pose.position.z);
      astar_finalpath.poses.push_back(Dpose);
    } 
  }
  astar_finalpath_10 = astar_finalpath;
  astarpath_ini = true;
  // std::cout << "astarpath_ini : " << astarpath_ini << std::endl;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "veh_predictwaypoint");
  ros::NodeHandle node;

  ros::Subscriber vehinfo_sub = node.subscribe("veh_info", 1, vehinfoCallback);
  ros::Subscriber imudata_sub = node.subscribe("imu_data", 1, imudataCallback);
  ros::Subscriber rear_current_pose_sub = node.subscribe("rear_current_pose", 1, RearCurrentPoseCallback);
  ros::Subscriber astar_path_final_sub = node.subscribe("nav_path_astar_final", 1, astarpathfinalCallback);
  vehbb_pub = node.advertise<geometry_msgs::PolygonStamped>("veh_bb", 10, true);
  veh_predictpath_pub = node.advertise<nav_msgs::Path>("veh_predictpath", 10, true);
  veh_predictpath_rel_pub = node.advertise<nav_msgs::Path>("veh_predictpath_rel", 10, true);
  // ros::Rate loop_rate(0.0001);
  // while (ros::ok())
  // { 
  
  //   ros::spinOnce();
  //   loop_rate.sleep();   
  // }

  ros::spin();
  return 0;
};
