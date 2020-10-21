#include <ros/ros.h>
#include <msgs/BehaviorSceneRegister.h>
#include <autoware_planning_msgs/StopReason.h>
#include <autoware_planning_msgs/StopReasonArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <ros/package.h>
#include <fstream>

#define RT_PI 3.14159265358979323846

ros::Publisher bus_stop_register_pub;

msgs::BehaviorSceneRegister bus_stop_register_;

struct pose
{
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
    double speed;
};
pose current_pose;

struct pose_with_header
{
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
    double speed;
    std_msgs::Header header;
};
pose_with_header nearest_bus_stop;

bool bus_stop_ini = false;

double busstop_BusStopNum[2000] = {};
double busstop_BuildingNum[2000] = {};
double busstop_BusStopId[2000] = {};
int read_index = 0;
bool busstop_txt_ini = false;

template <int size_readtmp>
void read_txt(std::string fpname, double (&BusStop_BusStopNum)[size_readtmp],double (&BusStop_BuildingNum)[size_readtmp],double (&BusStop_BusStopId)[size_readtmp])
{
  std::string fname = fpname;

    std::ifstream fin;
    char line[300];
    memset( line, 0, sizeof(line));

    fin.open(fname.c_str(),std::ios::in);
    if(!fin) 
    {
        std::cout << "Fail to import txt" <<std::endl;
        exit(1);
    }

    while(fin.getline(line,sizeof(line),'\n')) 
    {
      std::string nmea_str(line);
      std::stringstream ss(nmea_str);
      std::string token;

      getline(ss,token, ',');
      BusStop_BusStopNum[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      BusStop_BuildingNum[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      BusStop_BusStopId[read_index] = atof(token.c_str());
      read_index += 1;
    }
}

void Ini_busstop_bytxt()
{
  std::string fpname = ros::package::getPath("planning_initial");
  std::string fpname_s = fpname + "/data/HDmap_bus_stop_info.txt"; // full route

  read_txt(fpname_s, busstop_BusStopNum, busstop_BuildingNum, busstop_BusStopId);

  busstop_txt_ini = true;
}

void bus_stop_register(const msgs::BehaviorSceneRegister::ConstPtr& msg)
{
  bus_stop_register_.Module =  msg->Module;
  bus_stop_register_.ModuleId =  msg->ModuleId;
  bus_stop_register_.RegisterFlag = msg->RegisterFlag;
  for (int i=0; i<read_index; i++)
  {
    std::cout << "bus_stop_register_.ModuleId : " << bus_stop_register_.ModuleId << std::endl;
    std::cout << "busstop_BusStopId[" << i << "] : " << busstop_BusStopId[i] << std::endl;
    if (bus_stop_register_.ModuleId == busstop_BusStopId[i])
    {
      bus_stop_register_.BusStopNum = busstop_BusStopNum[i];
    }
  }
}

void register_callback(const msgs::BehaviorSceneRegister::ConstPtr& msg)
{
  if (msg->Module == "bus_stop")
  {
    bus_stop_register(msg);
  }
}

void stop_reasons_callback(const autoware_planning_msgs::StopReasonArray::ConstPtr& msg)
{
  std::cout << "msg->stop_reasons[0].reason : " << msg->stop_reasons[0].reason << std::endl;
  for (int i=0; i<msg->stop_reasons.size(); i++)
  {
    if (msg->stop_reasons[i].reason == "\"BusStop\"")
    {
      nearest_bus_stop.header = msg->header;
      nearest_bus_stop.x = msg->stop_reasons[i].stop_factors[0].stop_factor_points[0].x;
      nearest_bus_stop.y = msg->stop_reasons[i].stop_factors[0].stop_factor_points[0].y;
      nearest_bus_stop.z = msg->stop_reasons[i].stop_factors[0].stop_factor_points[0].z;
      bus_stop_ini = true;
      break;
    }
  }
}

void bus_stop_()
{
  bus_stop_register_.header.frame_id = "map";
  bus_stop_register_.header.stamp = ros::Time::now();
  ros::Duration time_check = bus_stop_register_.header.stamp - nearest_bus_stop.header.stamp;
  if (time_check.toSec() < 0.5)
  {
    bus_stop_register_.StopZone = 1;
    bus_stop_register_.Distance = sqrt((current_pose.x-nearest_bus_stop.x)*(current_pose.x-nearest_bus_stop.x) + (current_pose.y-nearest_bus_stop.y)*(current_pose.y-nearest_bus_stop.y));
  }
  else
  {
    bus_stop_register_.Module =  {};
    bus_stop_register_.ModuleId =  {};
    bus_stop_register_.BusStopNum = 99;
    bus_stop_register_.StopZone = 0;
    bus_stop_register_.Distance = 1000;
  }
  bus_stop_register_pub.publish(bus_stop_register_);
}

void current_pose_callback(const geometry_msgs::PoseStamped::ConstPtr& PSmsg)
{
  tf::Quaternion lidar_q(PSmsg->pose.orientation.x, PSmsg->pose.orientation.y, PSmsg->pose.orientation.z,PSmsg->pose.orientation.w);
  tf::Matrix3x3 lidar_m(lidar_q);

  current_pose.x = PSmsg->pose.position.x;
  current_pose.y = PSmsg->pose.position.y;
  current_pose.z = PSmsg->pose.position.z;
  lidar_m.getRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);
  if (current_pose.yaw < 0)
  {
    current_pose.yaw = current_pose.yaw + 2*RT_PI;
  }
  if (current_pose.yaw >= 2 * RT_PI)
  {
    current_pose.yaw = current_pose.yaw - 2*RT_PI;
  }
  if (busstop_txt_ini)
  {
    bus_stop_();
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "scene_register_checker");
  ros::NodeHandle node;

  ros::Subscriber behavior_scene_register_sub = node.subscribe("/planning/scenario_planning/status/behavior_scene_register", 1, register_callback);
  ros::Subscriber stop_reasons_sub = node.subscribe("/planning/scenario_planning/status/stop_reasons", 1, stop_reasons_callback);
  ros::Subscriber current_pose_sub = node.subscribe("/current_pose", 1, current_pose_callback);
  bus_stop_register_pub = node.advertise<msgs::BehaviorSceneRegister>("/bus_stop_register_info",1);

  Ini_busstop_bytxt();

  ros::spin();
  return 0;
};