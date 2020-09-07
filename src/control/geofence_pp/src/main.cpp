#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "Geofence_Class.h"

// For ROS
#include "std_msgs/Header.h"
#include "msgs/BoxPoint.h"
#include "msgs/DynamicPath.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/PathPrediction.h"
#include "msgs/PointXY.h"
#include "msgs/PointXYZ.h"
#include "msgs/PointXYZV.h"
#include "msgs/TrackInfo.h"
#include "msgs/LocalizationToVeh.h"
#include "msgs/VehInfo.h"
#include "ros/ros.h"
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>

// For PCL
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// For CAN
#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <typeinfo>
#define CAN_DLC 8
#define CAN_INTERFACE_NAME "can1"

// Specify running mode
//#define VIRTUAL
//#define RADARBOX
//#define PEDESTRIAN

static double Heading, SLAM_x, SLAM_y, SLAM_z;
static double current_x, current_y, current_z;
static Geofence BBox_Geofence(1.2);
static Geofence PedCross_Geofence(1.2);
static double Ego_speed_ms;
static int PP_Stop = 0;
static int PP_Stop_PedCross = 0;
static int PP_Distance = 1000;
static int PP_Distance_PedCross = 1000;
static int PP_Speed = 0;
static int PP_Speed_PedCross = 0;
ros::Publisher PP_geofence_line;
ros::Publisher PPCloud_pub;

void LocalizationToVehCallback(const msgs::LocalizationToVeh::ConstPtr& LTVmsg)
{
  Heading = LTVmsg->heading;
  SLAM_x = LTVmsg->x;
  SLAM_y = LTVmsg->y;
  SLAM_z = LTVmsg->z;
}

void CurrentPoseCallback(const geometry_msgs::PoseStamped& msg)
{
  current_x = msg.pose.position.x;
  current_y = msg.pose.position.y;
  current_z = msg.pose.position.z;
}

void VehinfoCallback(const msgs::VehInfo::ConstPtr& VImsg)
{
  Ego_speed_ms = VImsg->ego_speed;
  // Ego_speed_ms = 7.6;
}

void chatterCallbackPoly(const msgs::DynamicPath::ConstPtr& msg)
{
  vector<double> XX{ msg->XP1_0, msg->XP1_1, msg->XP1_2, msg->XP1_3, msg->XP1_4, msg->XP1_5,
                     msg->XP2_0, msg->XP2_1, msg->XP2_2, msg->XP2_3, msg->XP2_4, msg->XP2_5 };
  vector<double> YY{ msg->YP1_0, msg->YP1_1, msg->YP1_2, msg->YP1_3, msg->YP1_4, msg->YP1_5,
                     msg->YP2_0, msg->YP2_1, msg->YP2_2, msg->YP2_3, msg->YP2_4, msg->YP2_5 };
  vector<Point> Position;
  double Resolution = 0.001;
  Point Pos;
  for (double i = 0.0; i < 1.0; i += Resolution)
  {
    Pos.X = XX[0] + XX[1] * i + XX[2] * pow(i, 2) + XX[3] * pow(i, 3) + XX[4] * pow(i, 4) + XX[5] * pow(i, 5);
    Pos.Y = YY[0] + YY[1] * i + YY[2] * pow(i, 2) + YY[3] * pow(i, 3) + YY[4] * pow(i, 4) + YY[5] * pow(i, 5);
    Position.push_back(Pos);
  }
  for (double i = 0.0; i < 1.0; i += Resolution)
  {
    Pos.X = XX[6] + XX[7] * i + XX[8] * pow(i, 2) + XX[9] * pow(i, 3) + XX[10] * pow(i, 4) + XX[11] * pow(i, 5);
    Pos.Y = YY[6] + YY[7] * i + YY[8] * pow(i, 2) + YY[9] * pow(i, 3) + YY[10] * pow(i, 4) + YY[11] * pow(i, 5);
    Position.push_back(Pos);
  }
  BBox_Geofence.setPath(Position);
  PedCross_Geofence.setPath(Position);
}

void astar_callback(const nav_msgs::Path::ConstPtr& msg)
{
  vector<Point> Position;
  Point Pos;
  int size = 200;
  if (msg->poses.size() < size)
  {
    size = msg->poses.size();
  }

  double Resolution = 10;
  for (int i = 1; i < size; i++)
  {
    for (int j = 0; j < Resolution; j++)
    {
      Pos.X = msg->poses[i - 1].pose.position.x +
              j * (1 / Resolution) * (msg->poses[i].pose.position.x - msg->poses[i - 1].pose.position.x);
      Pos.Y = msg->poses[i - 1].pose.position.y +
              j * (1 / Resolution) * (msg->poses[i].pose.position.y - msg->poses[i - 1].pose.position.y);
      Position.push_back(Pos);
    }
  }
  BBox_Geofence.setPath(Position);
  PedCross_Geofence.setPath(Position);
}

void Plot_geofence(Point temp)
{
  visualization_msgs::Marker line_list;
  line_list.header.frame_id = "/map";
  // line_list.header.stamp = ros::Time::now();
  line_list.ns = "PP_line";
  line_list.lifetime = ros::Duration(0.5);
  line_list.action = visualization_msgs::Marker::ADD;
  line_list.pose.orientation.w = 1.0;
  line_list.id = 1;
  line_list.type = visualization_msgs::Marker::LINE_LIST;
  line_list.scale.x = 0.3;
  line_list.color.b = 1.0;
  line_list.color.a = 1.0;

  geometry_msgs::Point p;
  p.x = temp.X + 1.5 * cos(temp.Direction - M_PI / 2);
  p.y = temp.Y + 1.5 * sin(temp.Direction - M_PI / 2);
  p.z = SLAM_z - 2.0;
  line_list.points.push_back(p);
  p.x = temp.X - 1.5 * cos(temp.Direction - M_PI / 2);
  p.y = temp.Y - 1.5 * sin(temp.Direction - M_PI / 2);
  p.z = SLAM_z - 2.0;
  line_list.points.push_back(p);
  PP_geofence_line.publish(line_list);
  // cout << "============" << temp.Direction << "============" << endl;
}

void chatterCallbackPP(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointXYZI temp;
  for (int i = 0; i < msg->objects.size(); i++)
  {
    if (msg->objects[i].track.is_ready_prediction == 1)
    {
      for (uint j = 0; j < msg->objects[i].track.forecasts.size(); j++)
      {
        temp.x = msg->objects[i].track.forecasts[j].position.x;
        temp.y = msg->objects[i].track.forecasts[j].position.y;
        cloud->points.push_back(temp);
      }
    }
  }
  sensor_msgs::PointCloud2 msgtemp;
  pcl::toROSMsg(*cloud, msgtemp);
  // msgtemp.header.stamp = msg->radHeader.stamp;
  // msgtemp.header.seq = msg->radHeader.seq;
  msgtemp.header.frame_id = "/base_link";
  PPCloud_pub.publish(msgtemp);

  PP_Stop = 0;
  PP_Distance = 100;
  PP_Speed = 0;
  for (uint i = 0; i < msg->objects.size(); i++)
  {
    // cout << "Start point: " << msg->objects[i].bPoint.p0.x << "," <<  msg->objects[i].bPoint.p0.y << endl;
    // double Center_X = (msg->objects[i].bPoint.p0.x + msg->objects[i].bPoint.p3.x + msg->objects[i].bPoint.p4.x +
    // msg->objects[i].bPoint.p7.x)/4; double Center_Y = (msg->objects[i].bPoint.p0.y + msg->objects[i].bPoint.p3.y +
    // msg->objects[i].bPoint.p4.y + msg->objects[i].bPoint.p7.y)/4;
    if (msg->objects[i].track.is_ready_prediction == 1)
    {
      for (uint j = 0; j < msg->objects[i].track.forecasts.size(); j++)
      {
        Point Point_temp;
        vector<Point> PointCloud_temp;
        // double time = (j+1)*0.5;
        // double Range_front = time*Ego_speed_ms;
        // double Range_back = time*Ego_speed_ms-7; // Length of bus = 7m
        /*
        if (Range_back < 0)
        {
          Range_back = 0;
        }
        //cout << "prediction_ready: "  << msg->objects[i].track.is_ready_prediction << endl;
        //cout << "x:"  << msg->objects[i].track.forecasts[j].position.x << "+++++++++++++++++++++++++" <<endl;
        //cout << "y:"  << msg->objects[i].track.forecasts[j].position.y << "+++++++++++++++++++++++++" <<endl;
        if(msg->objects[i].track.forecasts[j].position.y>-2 && msg->objects[i].track.forecasts[j].position.y<2)
        {
          cout << "x:"  << msg->objects[i].track.forecasts[j].position.x << "+++++++++++++++++++++++++" <<endl;
          cout << "y:"  << msg->objects[i].track.forecasts[j].position.y << "+++++++++++++++++++++++++" <<endl;
        }
        */
        Point_temp.X = msg->objects[i].track.forecasts[j].position.x;
        // cout << "x:" << Point_temp.X << endl;
        Point_temp.Y = msg->objects[i].track.forecasts[j].position.y;
        // cout << "y:" <<Point_temp.Y << endl;
        Point_temp.Speed = msg->objects[i].relSpeed;
        PointCloud_temp.push_back(Point_temp);
        /*
        Point_temp.X = msg->objects[i].track.forecasts[j].position.x - (Center_X - msg->objects[i].bPoint.p0.x);
        Point_temp.Y = msg->objects[i].track.forecasts[j].position.y - (Center_Y - msg->objects[i].bPoint.p0.y);
        Point_temp.Speed = msg->objects[i].relSpeed;
        PointCloud_temp.push_back(Point_temp);
        Point_temp.X = msg->objects[i].track.forecasts[j].position.x - (Center_X - msg->objects[i].bPoint.p3.x);
        Point_temp.Y = msg->objects[i].track.forecasts[j].position.y - (Center_Y - msg->objects[i].bPoint.p3.y);
        Point_temp.Speed = msg->objects[i].relSpeed;
        PointCloud_temp.push_back(Point_temp);
        Point_temp.X = msg->objects[i].track.forecasts[j].position.x - (Center_X - msg->objects[i].bPoint.p4.x);
        Point_temp.Y = msg->objects[i].track.forecasts[j].position.y - (Center_Y - msg->objects[i].bPoint.p4.y);
        Point_temp.Speed = msg->objects[i].relSpeed;
        PointCloud_temp.push_back(Point_temp);
        Point_temp.X = msg->objects[i].track.forecasts[j].position.x - (Center_X - msg->objects[i].bPoint.p7.x);
        Point_temp.Y = msg->objects[i].track.forecasts[j].position.y - (Center_Y - msg->objects[i].bPoint.p7.y);
        Point_temp.Speed = msg->objects[i].relSpeed;
        PointCloud_temp.push_back(Point_temp);
        */
        // cout << msg->objects[i].track.forecasts[j].position.x << "," << msg->objects[i].track.forecasts[j].position.y
        // << endl;

#ifdef VIRTUAL
        BBox_Geofence.setPointCloud(PointCloud_temp, false, SLAM_x, SLAM_y, Heading);
#else
        BBox_Geofence.setPointCloud(PointCloud_temp, true, SLAM_x, SLAM_y, Heading);
#endif
        if (BBox_Geofence.Calculator() == 1)
        {
          cerr << "Please initialize all PCloud parameters first" << endl;
          return;
        }
        if (BBox_Geofence.getDistance() < 80)
        {
          // cout << "PP Points in boundary: " << BBox_Geofence.getDistance() << " - " << BBox_Geofence.getFarest() <<
          // endl; cout << "(x,y): " << BBox_Geofence.getNearest_X() << "," << BBox_Geofence.getNearest_Y() << endl;
          // Plot geofence PP
          if (BBox_Geofence.getDistance() < PP_Distance && BBox_Geofence.getDistance() > 3.8)
          {
            PP_Distance = BBox_Geofence.getDistance();
            PP_Speed = BBox_Geofence.getObjSpeed();
            Plot_geofence(BBox_Geofence.findDirection());
          }
          // if(!(BBox_Geofence.getDistance()>Range_front || BBox_Geofence.getFarest()<Range_back))
          {
            // cout << "Collision appears" << endl;
            PP_Stop = 1;
          }
        }
      }
    }
  }
}

void chatterCallbackPP_PedCross(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  PP_Stop_PedCross = 0;
  PP_Distance_PedCross = 100;
  PP_Speed_PedCross = 0;
  for (uint i = 0; i < msg->objects.size(); i++)
  {
    if (msg->objects[i].track.is_ready_prediction == 1)
    {
      for (uint j = 0; j < msg->objects[i].track.forecasts.size(); j++)
      {
        Point Point_temp;
        vector<Point> PointCloud_temp;
        Point_temp.X = msg->objects[i].track.forecasts[j].position.x;
        // cout << "x:" << Point_temp.X << endl;
        Point_temp.Y = msg->objects[i].track.forecasts[j].position.y;
        // cout << "y:" <<Point_temp.Y << endl;
        Point_temp.Speed = msg->objects[i].relSpeed;
        PointCloud_temp.push_back(Point_temp);
        // cout << msg->objects[i].track.forecasts[j].position.x << "," << msg->objects[i].track.forecasts[j].position.y
        // << endl;

        PedCross_Geofence.setPointCloud(PointCloud_temp, true, SLAM_x, SLAM_y, Heading);

        if (PedCross_Geofence.Calculator() == 1)
        {
          cerr << "Please initialize all PCloud parameters first" << endl;
          return;
        }
        if (PedCross_Geofence.getDistance() < 80)
        {
          // cout << "PP Points in boundary: " << PedCross_Geofence.getDistance() << " - " <<
          // PedCross_Geofence.getFarest() << endl; cout << "(x,y): " << PedCross_Geofence.getNearest_X() << "," <<
          // PedCross_Geofence.getNearest_Y() << endl; Plot geofence PP
          if (PedCross_Geofence.getDistance() < PP_Distance_PedCross && PedCross_Geofence.getDistance() > 3.8)
          {
            PP_Distance_PedCross = PedCross_Geofence.getDistance();
            PP_Speed_PedCross = PedCross_Geofence.getObjSpeed();
            Plot_geofence(PedCross_Geofence.findDirection());
          }
          // if(!(PedCross_Geofence.getDistance()>Range_front || PedCross_Geofence.getFarest()<Range_back))
          {
            // cout << "Collision appears" << endl;
            PP_Stop_PedCross = 1;
          }
        }
      }
    }
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "Geofence_PP");
  ros::NodeHandle n;
  ros::Subscriber PCloudGeofenceSub = n.subscribe("dynamic_path_para", 1, chatterCallbackPoly);
  ros::Subscriber LTVSub = n.subscribe("localization_to_veh", 1, LocalizationToVehCallback);
  ros::Subscriber VI_sub = n.subscribe("veh_info", 1, VehinfoCallback);
  ros::Subscriber AstarSub = n.subscribe("nav_path_astar_final", 1, astar_callback);
  ros::Subscriber current_pose_sub = n.subscribe("current_pose", 1, CurrentPoseCallback);
#ifdef VIRTUAL
  ros::Subscriber BBoxGeofenceSub = n.subscribe("abs_virBB_array", 1, chatterCallbackPP);
#elif defined RADARBOX
  ros::Subscriber BBoxGeofenceSub = n.subscribe("PathPredictionOutput/radar", 1, chatterCallbackPP);
#elif defined PEDESTRIAN
  ros::Subscriber BBoxGeofenceSub = n.subscribe("/PedCross/Alert", 1, chatterCallbackPP);
#else
  ros::Subscriber BBoxGeofenceSub = n.subscribe("PathPredictionOutput", 1, chatterCallbackPP);
#endif
  ros::Subscriber PedCrossGeofenceSub = n.subscribe("/PedCross/Alert", 1, chatterCallbackPP_PedCross);

  PP_geofence_line = n.advertise<visualization_msgs::Marker>("PP_geofence_line", 1);
  PPCloud_pub = n.advertise<sensor_msgs::PointCloud2>("pp_point_cloud", 1);

  ros::Rate loop_rate(10);

  int test123;
  ros::param::get("test123", test123);
  cout << "test123: " << test123 << endl;

  int s;
  int nbytes;
  struct sockaddr_can addr;
  struct can_frame frame;
  struct ifreq ifr;
  const char* ifname = CAN_INTERFACE_NAME;
  if ((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
  {
    perror("Error while opening socket");
  }
  strcpy(ifr.ifr_name, ifname);
  ioctl(s, SIOCGIFINDEX, &ifr);
  addr.can_family = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;
  if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) < 0)
  {
    perror("Error in socket bind");
  }
  frame.can_dlc = CAN_DLC;

  while (ros::ok())
  {
    ros::spinOnce();
    if (PP_Stop == 0)
    {
      // cout << "No Collision" << endl;
    }
    else
    {
      cout << "Collision appears" << endl;
      cout << "Distance:" << PP_Distance << endl;
      cout << "Speed:" << PP_Speed << endl;
    }
    frame.can_id = 0x595;
    frame.data[0] = (short int)(PP_Stop * 100);
    frame.data[1] = (short int)(PP_Stop * 100) >> 8;
    frame.data[2] = (short int)(PP_Distance * 100);
    frame.data[3] = (short int)(PP_Distance * 100) >> 8;
    frame.data[4] = (short int)(PP_Speed * 100);
    frame.data[5] = (short int)(PP_Speed * 100) >> 8;
    nbytes = write(s, &frame, sizeof(struct can_frame));

    if (PP_Stop_PedCross == 0)
    {
      // cout << "No Collision" << endl;
    }
    else
    {
      cout << "PedCross collision appears" << endl;
      cout << "PedCross distance:" << PP_Distance_PedCross << endl;
      cout << "PedCross speed:" << PP_Speed_PedCross << endl;
    }
    frame.can_id = 0x596;
    frame.data[0] = (short int)(PP_Stop_PedCross * 100);
    frame.data[1] = (short int)(PP_Stop_PedCross * 100) >> 8;
    frame.data[2] = (short int)(PP_Distance_PedCross * 100);
    frame.data[3] = (short int)(PP_Distance_PedCross * 100) >> 8;
    frame.data[4] = (short int)(PP_Speed_PedCross * 100);
    frame.data[5] = (short int)(PP_Speed_PedCross * 100) >> 8;
    nbytes = write(s, &frame, sizeof(struct can_frame));
    // printf("Wrote %d bytes\n", nbytes);
    loop_rate.sleep();
  }
  close(s);
  return 0;
}
