#include "Geofence.h"

// For ROS
#include <visualization_msgs/Marker.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// For itriadv
#include "msgs/DynamicPath.h"
#include "msgs/DetectedObjectArray.h"
#include <autoware_perception_msgs/DynamicObjectArray.h>
#include "msgs/PointXYZV.h"
#include "msgs/LocalizationToVeh.h"
#include "msgs/VehInfo.h"
#include <nav_msgs/Path.h>

// For PCL
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

#define DEBUG 0  // 0: OFF; 1: ON

int g_input_source = 0;  // 0: PP; 1: VIRTUAL; 2: RADARBOX; 3: NONE
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

tf2_ros::Buffer tf_buffer;

void callbackLocalizationToVeh(const msgs::LocalizationToVeh::ConstPtr& LTVmsg)
{
  Heading = LTVmsg->heading;
  SLAM_x = LTVmsg->x;
  SLAM_y = LTVmsg->y;
  SLAM_z = LTVmsg->z;
  // std::cout << "SLAM_x " << SLAM_x << " SLAM_y " << SLAM_y << " SLAM_z " << SLAM_y << std::endl;
}

void callbackCurrentPose(const geometry_msgs::PoseStamped& msg)
{
  current_x = msg.pose.position.x;
  current_y = msg.pose.position.y;
  current_z = msg.pose.position.z;
}

void callbackVehInfo(const msgs::VehInfo::ConstPtr& VImsg)
{
  Ego_speed_ms = VImsg->ego_speed;
  // Ego_speed_ms = 7.6;
}

void callbackPoly(const msgs::DynamicPath::ConstPtr& msg)
{
  std::vector<double> XX{ msg->XP1_0, msg->XP1_1, msg->XP1_2, msg->XP1_3, msg->XP1_4, msg->XP1_5,
                          msg->XP2_0, msg->XP2_1, msg->XP2_2, msg->XP2_3, msg->XP2_4, msg->XP2_5 };
  std::vector<double> YY{ msg->YP1_0, msg->YP1_1, msg->YP1_2, msg->YP1_3, msg->YP1_4, msg->YP1_5,
                          msg->YP2_0, msg->YP2_1, msg->YP2_2, msg->YP2_3, msg->YP2_4, msg->YP2_5 };

  std::vector<Point> Position;
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

void callbackAStar(const nav_msgs::Path::ConstPtr& msg)
{
  std::vector<Point> Position;
  Point Pos;
  int size = std::min((int)msg->poses.size(), 200);
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
  line_list.ns = "PP_line";
  line_list.lifetime = ros::Duration(0.5);
  line_list.action = visualization_msgs::Marker::ADD;
  line_list.pose.orientation.w = 1.0;
  line_list.id = 1;
  line_list.type = visualization_msgs::Marker::LINE_LIST;
  line_list.scale.x = 0.3;

  // Assign color
  line_list.color.r = 0.0;
  line_list.color.g = 0.0;
  line_list.color.b = 1.0;
  line_list.color.a = 1.0;

  geometry_msgs::Point p;
  line_list.points.reserve(2);
  double half_pi = M_PI / 2;
  double c1 = 1.5;
  double c2 = 2.0;

  // Add one endpoint
  p.x = temp.X + c1 * cos(temp.Direction - half_pi);
  p.y = temp.Y + c1 * sin(temp.Direction - half_pi);
  p.z = SLAM_z - c2;
  line_list.points.push_back(p);

  // Add another endpoint
  p.x = temp.X - c1 * cos(temp.Direction - half_pi);
  p.y = temp.Y - c1 * sin(temp.Direction - half_pi);
  p.z = SLAM_z - c2;
  line_list.points.push_back(p);

  // Publish
  PP_geofence_line.publish(line_list);
}

void publishPPCloud(const msgs::DetectedObjectArray::ConstPtr& msg, const int numForecastsOfObject)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  cloud->points.reserve(msg->objects.size() * numForecastsOfObject);

  for (const auto& obj : msg->objects)
  {
    if (obj.track.is_ready_prediction)
    {
      for (const auto& forecast : obj.track.forecasts)
      {
        pcl::PointXYZI p;
        p.x = forecast.position.x;
        p.y = forecast.position.y;
        cloud->points.push_back(p);
      }
    }
  }

  sensor_msgs::PointCloud2 msg_pub;
  pcl::toROSMsg(*cloud, msg_pub);
  msg_pub.header.frame_id = "/base_link";

  PPCloud_pub.publish(msg_pub);
}

void plotPP(const msgs::DetectedObjectArray::ConstPtr& msg, Geofence& g, int& pp_stop, int& pp_distance, int& pp_speed)
{
  for (const auto& obj : msg->objects)
  {
    if (obj.track.is_ready_prediction)
    {
      for (const auto& forecast : obj.track.forecasts)
      {
        Point p;
        std::vector<Point> p_vec;
        p_vec.reserve(1);
        p.X = forecast.position.x;
        p.Y = forecast.position.y;
        p.Speed = obj.relSpeed;
        p_vec.push_back(p);

        bool isLocal = true;  // setPointCloud(): no TF
        if (g_input_source == 1)
        {
          isLocal = false;  // setPointCloud(): TF to /map
        }

        g.setPointCloud(p_vec, isLocal, SLAM_x, SLAM_y, Heading);

        if (g.Calculator() == 1)
        {
          std::cerr << "Please initialize all PointCloud parameters first!" << std::endl;
          return;
        }

        if (g.getDistance() < 80)
        {
          if (g.getDistance() < pp_distance && g.getDistance() > 3.8)
          {
            // update
            pp_distance = g.getDistance();
            pp_speed = g.getObjSpeed();

            // plot geofence
            Plot_geofence(g.findDirection());
            std::cout << "Plot geofence of PP!" << std::endl;
          }
          pp_stop = 1;
        }
      }
    }
  }
}

void callbackPP(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  publishPPCloud(msg, 20);  // require TF /base_link
  PP_Stop = 0;
  PP_Distance = 100;
  PP_Speed = 0;
  plotPP(msg, BBox_Geofence, PP_Stop, PP_Distance, PP_Speed);  // require TF /map
}

void callbackPP_PedCross(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  PP_Stop_PedCross = 0;
  PP_Distance_PedCross = 100;
  PP_Speed_PedCross = 0;
  plotPP(msg, PedCross_Geofence, PP_Stop_PedCross, PP_Distance_PedCross, PP_Speed_PedCross);  // require TF /map
}

geometry_msgs::TransformStamped getTF(const std::string& target_frame, const std::string& source_fram,
                                      const ros::Time timestamp)
{
  geometry_msgs::TransformStamped tf_stamped;

  try
  {
    tf_stamped = tf_buffer.lookupTransform("base_link", "map", timestamp);
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s", ex.what());
    try
    {
      tf_stamped = tf_buffer.lookupTransform("base_link", "map", ros::Time(0));
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN("%s", ex.what());
    }
  }

  return tf_stamped;
}

void tf_map_to_base_link(const double x1, const double y1, double& x2, double& y2,
                         const geometry_msgs::TransformStamped tf_stamped)
{
  // TF (map-to-base_link) for object position (3 steps)
  // Step 1. Create pose
  geometry_msgs::Pose pose;
  pose.position.x = x1;
  pose.position.y = y1;
  pose.position.z = 0;
  pose.orientation.x = 0;
  pose.orientation.y = 0;
  pose.orientation.z = 0;
  pose.orientation.w = 1;

  // Step 2. Apply TF to pose
  tf2::Transform tf1;
  tf2::fromMsg(pose, tf1);

  tf2::Transform tf2;
  geometry_msgs::Transform tf_tmp = tf_stamped.transform;
  tf2::fromMsg(tf_tmp, tf2);

  tf2::Transform tf3 = tf2 * tf1;

  geometry_msgs::Pose pose_in_base_link;
  tf2::toMsg(tf3, pose_in_base_link);

  // Step 3. Save transformed pose
  x2 = pose_in_base_link.position.x;
  y2 = pose_in_base_link.position.y;
}

void publishPPCloud2(const autoware_perception_msgs::DynamicObjectArray::ConstPtr& msg, const int numForecastsOfObject,
                     geometry_msgs::TransformStamped tf_stamped)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  cloud->points.reserve(msg->objects.size() * numForecastsOfObject);

  for (const auto& obj : msg->objects)
  {
    for (const auto& predicted_path : obj.state.predicted_paths)
    {
      for (const auto& forecast : predicted_path.path)
      {
        pcl::PointXYZI p;
        p.x = forecast.pose.pose.position.x;
        p.y = forecast.pose.pose.position.y;
        double tmpx = 0;
        double tmpy = 0;
        tf_map_to_base_link(p.x, p.y, tmpx, tmpy, tf_stamped);
        p.x = tmpx;
        p.y = tmpy;
        cloud->points.push_back(p);
      }
    }
  }

  sensor_msgs::PointCloud2 msg_pub;
  pcl::toROSMsg(*cloud, msg_pub);
  msg_pub.header.frame_id = "/base_link";

  PPCloud_pub.publish(msg_pub);
}

void plotPP2(const autoware_perception_msgs::DynamicObjectArray::ConstPtr& msg, Geofence& g, int& pp_stop,
             int& pp_distance, int& pp_speed)
{
  for (const auto& obj : msg->objects)
  {
    for (const auto& predicted_path : obj.state.predicted_paths)
    {
      for (const auto& forecast : predicted_path.path)
      {
        Point p;
        p.X = forecast.pose.pose.position.x;
        p.Y = forecast.pose.pose.position.y;
        p.Speed = std::sqrt(std::pow(obj.state.twist_covariance.twist.linear.x, 2) +
                            std::pow(obj.state.twist_covariance.twist.linear.y, 2));

        std::vector<Point> p_vec;
        p_vec.reserve(1);
        p_vec.push_back(p);

        g.setPointCloud(p_vec, false, SLAM_x, SLAM_y, Heading);  // no TF

        if (g.Calculator() == 1)
        {
          std::cerr << "Please initialize all PointCloud parameters first!" << std::endl;
          return;
        }

        if (g.getDistance() < 80)
        {
          if (g.getDistance() < pp_distance && g.getDistance() > 3.8)
          {
            // update
            pp_distance = g.getDistance();
            pp_speed = g.getObjSpeed();

            // plot geofence
            Plot_geofence(g.findDirection());
            std::cout << "Plot geofence of MAP PP!" << std::endl;
          }
          pp_stop = 1;
        }
      }
    }
  }
}

void callbackPP2(const autoware_perception_msgs::DynamicObjectArray::ConstPtr& msg)
{
  if (msg->objects.empty())
  {
    return;
  }

  geometry_msgs::TransformStamped tf_stamped = getTF("base_link", "map", msg->header.stamp);

  publishPPCloud2(msg, (int)msg->objects[0].state.predicted_paths.size(), tf_stamped);  // require TF /base_link
  PP_Stop = 0;
  PP_Distance = 100;
  PP_Speed = 0;
  plotPP2(msg, BBox_Geofence, PP_Stop, PP_Distance, PP_Speed);  // require TF /map
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "Geofence_Map_PP");

  ros::NodeHandle n;
  ros::Subscriber PCloudGeofenceSub = n.subscribe("dynamic_path_para", 1, callbackPoly);
  ros::Subscriber LTVSub = n.subscribe("localization_to_veh", 1, callbackLocalizationToVeh);
  ros::Subscriber VI_sub = n.subscribe("veh_info", 1, callbackVehInfo);
  ros::Subscriber AStarSub = n.subscribe("nav_path_astar_final", 1, callbackAStar);
  ros::Subscriber current_pose_sub = n.subscribe("current_pose", 1, callbackCurrentPose);

  ros::Subscriber PedCrossGeofenceSub = n.subscribe("PedCross/Alert", 1, callbackPP_PedCross);
  std::cout << "Input Source: PEDCROSS" << std::endl;
  ros::Subscriber MapPPGeofenceSub = n.subscribe("objects", 1, callbackPP2);
  std::cout << "Input Source: MAP PP" << std::endl;

  ros::Subscriber BBoxGeofenceSub;

  if (g_input_source == 0)
  {
    BBoxGeofenceSub = n.subscribe("PathPredictionOutput", 1, callbackPP);
    std::cout << "Input Source: PP" << std::endl;
  }
  else if (g_input_source == 1)
  {
    BBoxGeofenceSub = n.subscribe("abs_virBB_array", 1, callbackPP);
    std::cout << "Input Source: VIRTUAL" << std::endl;
  }
  else if (g_input_source == 2)
  {
    BBoxGeofenceSub = n.subscribe("PathPredictionOutput/radar", 1, callbackPP);
    std::cout << "Input Source: RADARBOX" << std::endl;
  }

  PP_geofence_line = n.advertise<visualization_msgs::Marker>("PP_geofence_line", 1);
  PPCloud_pub = n.advertise<sensor_msgs::PointCloud2>("pp_point_cloud", 1);

  tf2_ros::TransformListener tf_listener(tf_buffer);

  ros::Rate loop_rate(10);

  int test123;
  ros::param::get("test123", test123);
  std::cout << "test123: " << test123 << std::endl;

  int s;
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

  int nbytes;

  while (ros::ok())
  {
    ros::spinOnce();

    if (PP_Stop != 0)
    {
      std::cout << "Collision appears: Distance=" << PP_Distance << " Speed=" << PP_Speed << std::endl;
    }

    frame.can_id = 0x595;
    frame.data[0] = (short int)(PP_Stop * 100);
    frame.data[1] = (short int)(PP_Stop * 100) >> 8;
    frame.data[2] = (short int)(PP_Distance * 100);
    frame.data[3] = (short int)(PP_Distance * 100) >> 8;
    frame.data[4] = (short int)(PP_Speed * 100);
    frame.data[5] = (short int)(PP_Speed * 100) >> 8;
    nbytes = write(s, &frame, sizeof(struct can_frame));

    if (PP_Stop_PedCross != 0)
    {
      std::cout << "Collision appears (PedCross): Distance=" << PP_Distance_PedCross << " Speed=" << PP_Speed_PedCross
                << std::endl;
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

    nbytes = nbytes;  // for preventing warning of unused variable during compile time

    loop_rate.sleep();
  }

  close(s);
}
