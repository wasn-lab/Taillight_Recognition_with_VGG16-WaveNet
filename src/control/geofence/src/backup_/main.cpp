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
#include "msgs/MMTPInfo.h"
#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Int32.h"
#include <visualization_msgs/Marker.h>
#include <nav_msgs/Path.h>

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
//#define TRACKINGBOX

static Geofence PCloud_Geofence(1.2);
static Geofence CPoint_Geofence(1.2);
static Geofence BBox_Geofence(1.2);
static Geofence Radar_Geofence(1.6);
static Geofence PCloud_Geofence_original(1.2);
static Geofence Deviate_Geofence(1.2);
static double Heading, SLAM_x, SLAM_y, SLAM_z;
// static uint Deadend_flag;
static uint overtake_over_flag;
ros::Publisher Radar_marker;
ros::Publisher Geofence_line;

void LocalizationToVehCallback(const msgs::LocalizationToVeh::ConstPtr& LTVmsg)
{
  Heading = LTVmsg->heading;
  SLAM_x = LTVmsg->x;
  SLAM_y = LTVmsg->y;
  SLAM_z = LTVmsg->z;
}

/*
void mm_tp_infoCallback(const msgs::MMTPInfo::ConstPtr& MMTPInfo){
  Deadend_flag = MMTPInfo->Deadend_flag;
}
*/

void overtake_over_Callback(const std_msgs::Int32::ConstPtr& msg)
{
  overtake_over_flag = msg->data;
}

void chatterCallbackPCloud(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  Point Point_temp;
  vector<Point> PointCloud_temp;
  for (uint i = 0; i < msg->objects.size(); i++)
  {
    Point_temp.X = msg->objects[i].bPoint.p0.x;
    Point_temp.Y = msg->objects[i].bPoint.p0.y;
    Point_temp.Speed = msg->objects[i].relSpeed;
    PointCloud_temp.push_back(Point_temp);
    Point_temp.X = msg->objects[i].bPoint.p3.x;
    Point_temp.Y = msg->objects[i].bPoint.p3.y;
    Point_temp.Speed = msg->objects[i].relSpeed;
    PointCloud_temp.push_back(Point_temp);
    Point_temp.X = msg->objects[i].bPoint.p4.x;
    Point_temp.Y = msg->objects[i].bPoint.p4.y;
    Point_temp.Speed = msg->objects[i].relSpeed;
    PointCloud_temp.push_back(Point_temp);
    Point_temp.X = msg->objects[i].bPoint.p7.x;
    Point_temp.Y = msg->objects[i].bPoint.p7.y;
    Point_temp.Speed = msg->objects[i].relSpeed;
    PointCloud_temp.push_back(Point_temp);
    Point_temp.X = (msg->objects[i].bPoint.p0.x + msg->objects[i].bPoint.p3.x + msg->objects[i].bPoint.p4.x +
                    msg->objects[i].bPoint.p7.x) /
                   4;
    Point_temp.Y = (msg->objects[i].bPoint.p0.y + msg->objects[i].bPoint.p3.y + msg->objects[i].bPoint.p4.y +
                    msg->objects[i].bPoint.p7.y) /
                   4;
    Point_temp.Speed = msg->objects[i].relSpeed;
    PointCloud_temp.push_back(Point_temp);
  }
#ifdef VIRTUAL
  BBox_Geofence.setPointCloud(PointCloud_temp, false, SLAM_x, SLAM_y, Heading);
#else
  BBox_Geofence.setPointCloud(PointCloud_temp, true, SLAM_x, SLAM_y, Heading);
#endif
}

void chatterCallbackCPoint(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  Point Point_temp;
  vector<Point> PointCloud_temp;
  for (uint i = 0; i < msg->objects.size(); i++)
  {
    for (uint j = 0; j < msg->objects[i].cPoint.lowerAreaPoints.size(); j++)
    {
      Point_temp.X = msg->objects[i].cPoint.lowerAreaPoints[j].x;
      Point_temp.Y = msg->objects[i].cPoint.lowerAreaPoints[j].y;
      Point_temp.Speed = msg->objects[i].relSpeed;
      PointCloud_temp.push_back(Point_temp);
    }
  }
  CPoint_Geofence.setPointCloud(PointCloud_temp, true, SLAM_x, SLAM_y, Heading);
}

void chatterCallbackPCloud_Radar(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  Point Point_temp;
  vector<Point> PointCloud_temp;
  for (uint i = 0; i < msg->objects.size(); i++)
  {
    Point_temp.X = msg->objects[i].bPoint.p0.x;
    Point_temp.Y = msg->objects[i].bPoint.p0.y;
    Point_temp.Speed = msg->objects[i].relSpeed;
    PointCloud_temp.push_back(Point_temp);
    Point_temp.X = msg->objects[i].bPoint.p3.x;
    Point_temp.Y = msg->objects[i].bPoint.p3.y;
    Point_temp.Speed = msg->objects[i].relSpeed;
    PointCloud_temp.push_back(Point_temp);
    Point_temp.X = msg->objects[i].bPoint.p4.x;
    Point_temp.Y = msg->objects[i].bPoint.p4.y;
    Point_temp.Speed = msg->objects[i].relSpeed;
    PointCloud_temp.push_back(Point_temp);
    Point_temp.X = msg->objects[i].bPoint.p7.x;
    Point_temp.Y = msg->objects[i].bPoint.p7.y;
    Point_temp.Speed = msg->objects[i].relSpeed;
    PointCloud_temp.push_back(Point_temp);
    Point_temp.X = (msg->objects[i].bPoint.p0.x + msg->objects[i].bPoint.p3.x + msg->objects[i].bPoint.p4.x +
                    msg->objects[i].bPoint.p7.x) /
                   4;
    Point_temp.Y = (msg->objects[i].bPoint.p0.y + msg->objects[i].bPoint.p3.y + msg->objects[i].bPoint.p4.y +
                    msg->objects[i].bPoint.p7.y) /
                   4;
    Point_temp.Speed = msg->objects[i].relSpeed;
    PointCloud_temp.push_back(Point_temp);
  }
  Radar_Geofence.setPointCloud(PointCloud_temp, true, SLAM_x, SLAM_y, Heading);
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
  PCloud_Geofence.setPath(Position);
  BBox_Geofence.setPath(Position);
  Radar_Geofence.setPath(Position);
  CPoint_Geofence.setPath(Position);
}

void astar_callback(const nav_msgs::Path::ConstPtr& msg)
{
  vector<Point> Position;
  Point Pos;
  uint size = 200;
  if (msg->poses.size() < size)
  {
    size = msg->poses.size();
  }

  double Resolution = 10;
  for (uint i = 1; i < size; i++)
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
  PCloud_Geofence.setPath(Position);
  BBox_Geofence.setPath(Position);
  Radar_Geofence.setPath(Position);
  CPoint_Geofence.setPath(Position);
}

void astar_original_callback(const nav_msgs::Path::ConstPtr& msg)
{
  vector<Point> Position;
  Point Pos;
  uint size = 200;
  if (msg->poses.size() < size)
  {
    size = msg->poses.size();
  }
  double Resolution = 10;
  for (uint i = 1; i < size; i++)
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
  PCloud_Geofence_original.setPath(Position);
}

void deviate_path_callback(const nav_msgs::Path::ConstPtr& msg)
{
  vector<Point> Position;
  Point Pos;
  uint size = 1000;
  if (msg->poses.size() < size)
  {
    size = msg->poses.size();
  }
  // cout << size << "--------------------------------------------------" << endl ;
  double Resolution = 10;
  for (uint i = 1; i < size; i++)
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
  Deviate_Geofence.setPath(Position);
}

void callback_LidarAll(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg, *cloud);
  Point Point_temp;
  vector<Point> PointCloud_temp;
  for (size_t i = 0; i < cloud->points.size(); ++i)
  {
    Point_temp.X = cloud->points[i].x;
    Point_temp.Y = cloud->points[i].y;
    // double z = cloud->points[i].z;
    // int intensity = cloud->points[i].intensity;
    Point_temp.Speed = 0.0;
    PointCloud_temp.push_back(Point_temp);
  }
  PCloud_Geofence.setPointCloud(PointCloud_temp, true, SLAM_x, SLAM_y, Heading);
  PCloud_Geofence_original.setPointCloud(PointCloud_temp, true, SLAM_x, SLAM_y, Heading);
  Deviate_Geofence.setPointCloud(PointCloud_temp, true, SLAM_x, SLAM_y, Heading);
}

void Publish_Marker_Radar(double X, double Y)
{
  uint32_t shape = visualization_msgs::Marker::SPHERE;
  visualization_msgs::Marker marker;
  marker.header.frame_id = "/map";
  marker.header.stamp = ros::Time::now();
  marker.ns = "RadarPlotter";
  marker.id = 0;
  marker.type = shape;
  marker.action = visualization_msgs::Marker::ADD;
  marker.lifetime = ros::Duration(0.5);

  marker.pose.position.x = X;
  marker.pose.position.y = Y;
  marker.pose.position.z = SLAM_z - 2;  // Set pooint to groud in /map frame
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 0.0;

  // Set the scale of the marker -- 1x1x1 here means 1m on a side
  marker.scale.x = 1.0;
  marker.scale.y = 1.0;
  marker.scale.z = 1.0;

  // Set the color -- be sure to set alpha to something non-zero!
  marker.color.r = 1.0f;
  marker.color.g = 0.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0;

  Radar_marker.publish(marker);
}

void Plot_geofence(Point temp)
{
  visualization_msgs::Marker line_list;
  line_list.header.frame_id = "/map";
  line_list.header.stamp = ros::Time::now();
  line_list.ns = "PC_line";
  line_list.lifetime = ros::Duration(0.5);
  line_list.action = visualization_msgs::Marker::ADD;
  line_list.pose.orientation.w = 1.0;
  line_list.id = 1;
  line_list.type = visualization_msgs::Marker::LINE_LIST;
  line_list.scale.x = 0.3;
  line_list.color.r = 1.0;
  line_list.color.g = 0.0;
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
  Geofence_line.publish(line_list);
}

void Plot_geofence_yellow(Point temp)
{
  visualization_msgs::Marker line_list;
  line_list.header.frame_id = "/map";
  line_list.header.stamp = ros::Time::now();
  line_list.ns = "PC_line";
  line_list.lifetime = ros::Duration(0.5);
  line_list.action = visualization_msgs::Marker::ADD;
  line_list.pose.orientation.w = 1.0;
  line_list.id = 1;
  line_list.type = visualization_msgs::Marker::LINE_LIST;
  line_list.scale.x = 0.3;
  line_list.color.r = 1.0;
  line_list.color.g = 1.0;
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
  Geofence_line.publish(line_list);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "Geofence");
  ros::NodeHandle n;
  ros::Subscriber LidAllSub = n.subscribe("ring_edge_point_cloud", 1, callback_LidarAll);
  ros::Subscriber AstarSub = n.subscribe("nav_path_astar_final", 1, astar_callback);
  ros::Subscriber AstarSub_original =
      n.subscribe("nav_path_astar_base_30", 1, astar_original_callback);  // For objects on original path
  ros::Subscriber deviate_path = n.subscribe("veh_predictpath", 1, deviate_path_callback);

  ros::Subscriber PCloudGeofenceSub = n.subscribe("dynamic_path_para", 1, chatterCallbackPoly);
  ros::Subscriber LTVSub = n.subscribe("localization_to_veh", 1, LocalizationToVehCallback);
  // ros::Subscriber MMTPSub = n.subscribe("mm_tp_info", 1, mm_tp_infoCallback);
  // ros::Subscriber avoidpath = n.subscribe("avoiding_path", 1, overtake_over_Callback);
  ros::Subscriber avoidpath = n.subscribe("astar_reach_goal", 1, overtake_over_Callback);
  ros::Subscriber RadarGeofenceSub = n.subscribe("PathPredictionOutput/radar", 1, chatterCallbackPCloud_Radar);

#ifdef VIRTUAL
  ros::Subscriber BBoxGeofenceSub = n.subscribe("abs_virBB_array", 1, chatterCallbackPCloud);
#else
  // ros::Subscriber BBoxGeofenceSub = n.subscribe("PathPredictionOutput", 1, chatterCallbackPCloud);
  ros::Subscriber BBoxGeofenceSub = n.subscribe("/CameraDetection/polygon", 1, chatterCallbackCPoint);
#endif
  Radar_marker = n.advertise<visualization_msgs::Marker>("RadarMarker", 1);
  Geofence_line = n.advertise<visualization_msgs::Marker>("Geofence_line", 1);
  ros::Publisher Geofence_PC = n.advertise<std_msgs::Float64>("Geofence_PC", 1);
  ros::Publisher Geofence_original = n.advertise<std_msgs::Float64>("Geofence_original", 1);
  ros::Rate loop_rate(20);

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
    cout << "========PCloud========" << endl;
    if (PCloud_Geofence.Calculator() == 0)
    {
      frame.can_id = 0x590;
      cout << "Trigger: " << PCloud_Geofence.getTrigger() << " ";
      cout << "Distance: " << setprecision(6) << PCloud_Geofence.getDistance() << "\t";
      cout << "Distance_wide: " << setprecision(6) << PCloud_Geofence.getDistance_w() << "\t";
      cout << "Speed: " << setprecision(6) << PCloud_Geofence.getObjSpeed() << endl;
      cout << "(X,Y): "
           << "(" << PCloud_Geofence.getNearest_X() << "," << PCloud_Geofence.getNearest_Y() << ")" << endl;
      // cout << "Speed: " << PCloud_Geofence.Xpoly_one.size() << "\t" << PCloud_Geofence.Xpoly_two.size() << "\t" <<
      // PCloud_Geofence.Ypoly_one.size() << "\t" << PCloud_Geofence.Ypoly_two.size() << endl; cout << "Pointcloud: " <<
      // PCloud_Geofence.PointCloud.size() << endl;
      frame.data[0] = (short int)(PCloud_Geofence.getDistance() * 100);
      frame.data[1] = (short int)(PCloud_Geofence.getDistance() * 100) >> 8;
      frame.data[2] = (short int)(PCloud_Geofence.getObjSpeed() * 100);
      frame.data[3] = (short int)(PCloud_Geofence.getObjSpeed() * 100) >> 8;
      frame.data[4] = (short int)(PCloud_Geofence.getNearest_X() * 10);
      frame.data[5] = (short int)(PCloud_Geofence.getNearest_X() * 10) >> 8;
      frame.data[6] = (short int)(PCloud_Geofence.getNearest_Y() * 10);
      frame.data[7] = (short int)(PCloud_Geofence.getNearest_Y() * 10) >> 8;
      nbytes = write(s, &frame, sizeof(struct can_frame));
      printf("Wrote %d bytes\n", nbytes);
      std_msgs::Float64 Geofence_temp;
      Geofence_temp.data = PCloud_Geofence.getDistance_w();
      Geofence_PC.publish(Geofence_temp);
      if (PCloud_Geofence.getDistance() < 80)
      {
        Plot_geofence(PCloud_Geofence.findDirection());
      }
    }
    else
    {
      cerr << "Please initialize all PCloud parameters first" << endl;
    }

    if (PCloud_Geofence_original.Calculator() == 0)
    {
      cout << "Origianl path's geofence: " << PCloud_Geofence_original.getDistance() << endl;
      std_msgs::Float64 Geofence_temp;
      Geofence_temp.data = PCloud_Geofence_original.getDistance();
      Geofence_original.publish(Geofence_temp);
    }
    else
    {
      cerr << "Please initialize PCloud original_path parameters first" << endl;
    }

    if (Deviate_Geofence.Calculator() == 0)
    {
      frame.can_id = 0x593;
      cout << "Deviate path's geofence: " << Deviate_Geofence.getDistance() << endl;
      frame.data[0] = (short int)(Deviate_Geofence.getDistance() * 100);
      frame.data[1] = (short int)(Deviate_Geofence.getDistance() * 100) >> 8;
      frame.data[2] = (short int)(Deviate_Geofence.getObjSpeed() * 100);
      frame.data[3] = (short int)(Deviate_Geofence.getObjSpeed() * 100) >> 8;
      frame.data[4] = (short int)(Deviate_Geofence.getNearest_X() * 10);
      frame.data[5] = (short int)(Deviate_Geofence.getNearest_X() * 10) >> 8;
      frame.data[6] = (short int)(Deviate_Geofence.getNearest_Y() * 10);
      frame.data[7] = (short int)(Deviate_Geofence.getNearest_Y() * 10) >> 8;
      nbytes = write(s, &frame, sizeof(struct can_frame));
      printf("Wrote %d bytes\n", nbytes);
      if (Deviate_Geofence.getDistance() < 80)
      {
        Plot_geofence(Deviate_Geofence.findDirection());
      }
    }
    else
    {
      cerr << "Please initialize deviate_path parameters first" << endl;
    }

    /*
    cout << "=========BBox=========" << endl;
    if(BBox_Geofence.Calculator()==0){
      frame.can_id  = 0x591;
      cout << "Trigger: " << BBox_Geofence.getTrigger() << " ";
      cout << "Distance: " <<  setprecision(6) << BBox_Geofence.getDistance() << "\t";
      cout << "Farest: " <<  setprecision(6) << BBox_Geofence.getFarest() << "\t";
      cout << "Speed: " << setprecision(6) << BBox_Geofence.getObjSpeed() << endl;
      cout << "(X,Y): " << "(" << BBox_Geofence.getNearest_X() << "," << BBox_Geofence.getNearest_Y() << ")" << endl;
      frame.data[0] = (short int)(BBox_Geofence.getDistance()*100);
      frame.data[1] = (short int)(BBox_Geofence.getDistance()*100)>>8;
      frame.data[2] = (short int)(BBox_Geofence.getObjSpeed()*100);
      frame.data[3] = (short int)(BBox_Geofence.getObjSpeed()*100)>>8;
      frame.data[4] = (short int)(BBox_Geofence.getNearest_X()*10);
      frame.data[5] = (short int)(BBox_Geofence.getNearest_X()*10)>>8;
      frame.data[6] = (short int)(BBox_Geofence.getNearest_Y()*10);
      frame.data[7] = (short int)(BBox_Geofence.getNearest_Y()*10)>>8;
      nbytes = write(s, &frame, sizeof(struct can_frame));
      printf("Wrote %d bytes\n", nbytes);
      //Publish_Marker(BBox_Geofence.getNearest_X(), BBox_Geofence.getNearest_Y());
      if(BBox_Geofence.getDistance()<80)
      {
        Plot_geofence(BBox_Geofence.findDirection());
      }
    }
    else{
      cerr << "Please initialize all BBox parameters first" << endl;
    }
    */

    cout << "=========CPoint(cam)=========" << endl;
    if (CPoint_Geofence.Calculator() == 0)
    {
      frame.can_id = 0x591;
      cout << "Trigger: " << CPoint_Geofence.getTrigger() << " ";
      cout << "Distance: " << setprecision(6) << CPoint_Geofence.getDistance() << "\t";
      cout << "Farest: " << setprecision(6) << CPoint_Geofence.getFarest() << "\t";
      cout << "Speed: " << setprecision(6) << CPoint_Geofence.getObjSpeed() << endl;
      cout << "(X,Y): "
           << "(" << CPoint_Geofence.getNearest_X() << "," << CPoint_Geofence.getNearest_Y() << ")" << endl;
      frame.data[0] = (short int)(CPoint_Geofence.getDistance() * 100);
      frame.data[1] = (short int)(CPoint_Geofence.getDistance() * 100) >> 8;
      frame.data[2] = (short int)(CPoint_Geofence.getObjSpeed() * 100);
      frame.data[3] = (short int)(CPoint_Geofence.getObjSpeed() * 100) >> 8;
      frame.data[4] = (short int)(CPoint_Geofence.getNearest_X() * 10);
      frame.data[5] = (short int)(CPoint_Geofence.getNearest_X() * 10) >> 8;
      frame.data[6] = (short int)(CPoint_Geofence.getNearest_Y() * 10);
      frame.data[7] = (short int)(CPoint_Geofence.getNearest_Y() * 10) >> 8;
      nbytes = write(s, &frame, sizeof(struct can_frame));
      printf("Wrote %d bytes\n", nbytes);
      if (CPoint_Geofence.getDistance() < 80)
      {
        Plot_geofence_yellow(CPoint_Geofence.findDirection());
      }
    }
    else
    {
      cerr << "Please initialize all CPoint parameters first" << endl;
    }

    cout << "========Radar=========" << endl;
    if (Radar_Geofence.Calculator() == 0)
    {
      frame.can_id = 0x592;
      cout << "Trigger: " << Radar_Geofence.getTrigger() << " ";
      cout << "Distance: " << setprecision(6) << Radar_Geofence.getDistance() << "\t";
      cout << "Speed: " << setprecision(6) << Radar_Geofence.getObjSpeed() << endl;
      cout << "(X,Y): "
           << "(" << Radar_Geofence.getNearest_X() << "," << Radar_Geofence.getNearest_Y() << ")" << endl
           << endl;
      frame.data[0] = (short int)(Radar_Geofence.getDistance() * 100);
      frame.data[1] = (short int)(Radar_Geofence.getDistance() * 100) >> 8;
      frame.data[2] = (short int)(Radar_Geofence.getObjSpeed() * 100);
      frame.data[3] = (short int)(Radar_Geofence.getObjSpeed() * 100) >> 8;
      frame.data[4] = (short int)(Radar_Geofence.getNearest_X() * 10);
      frame.data[5] = (short int)(Radar_Geofence.getNearest_X() * 10) >> 8;
      frame.data[6] = (short int)(Radar_Geofence.getNearest_Y() * 10);
      frame.data[7] = (short int)(Radar_Geofence.getNearest_Y() * 10) >> 8;
      nbytes = write(s, &frame, sizeof(struct can_frame));
      printf("Wrote %d bytes\n", nbytes);
      Publish_Marker_Radar(Radar_Geofence.getNearest_X(), Radar_Geofence.getNearest_Y());
    }
    else
    {
      cerr << "Please initialize all Radar parameters first" << endl;
    }

    frame.can_id = 0x599;
    cout << "overtake_over: " << overtake_over_flag << " ";
    frame.data[0] = (short int)(overtake_over_flag);
    frame.data[1] = (short int)(overtake_over_flag) >> 8;
    nbytes = write(s, &frame, sizeof(struct can_frame));
    printf("Wrote %d bytes\n", nbytes);
    cout << "******************************************" << endl;
    loop_rate.sleep();
  }
  close(s);
  return 0;
}
