#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <sensor_msgs/Imu.h>
#include <tf/tf.h>
#include <fstream>
#include <std_msgs/Int32.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>
#include <cmath>
#include <autoware_perception_msgs/DynamicObject.h>
#include <autoware_perception_msgs/DynamicObjectArray.h>
#include <msgs/VehInfo.h>
#include <msgs/Spat.h>
#include <autoware_perception_msgs/LampState.h>
#include <autoware_perception_msgs/TrafficLightState.h>
#include <autoware_perception_msgs/TrafficLightStateArray.h>
#include <msgs/BusStop.h>
#include <msgs/BusStopArray.h>
#include <ros/package.h>

#include <nav_msgs/OccupancyGrid.h>
#include <msgs/Flag_Info.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>

//For PCL
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// test
#include <geometry_msgs/Point32.h>

ros::Publisher rearcurrentpose_pub;
ros::Publisher objects_pub;
ros::Publisher nogroundpoints_pub;
ros::Publisher twist_pub;
ros::Publisher trafficlight_pub;
ros::Publisher busstop_pub;
ros::Publisher occ_maporigin_pub;
ros::Publisher occ_wayareamaporigin_pub;
// ros::Publisher enable_avoid_pub;
// ros::Publisher avoiding_path_pub;

static sensor_msgs::Imu imu_data_rad;

#define RT_PI 3.14159265358979323846

double wheel_dis = 3.8;

geometry_msgs::PoseStamped current_pose;

bool current_pose_init_flag = false;
double current_roll, current_pitch, current_yaw;

std::string map_frame_ = "map";
std::string location_name_ = "ITRI";

double busstop_BusStopNum[2000] = {};
double busstop_BuildingNum[2000] = {};
double busstop_BusStopId[2000] = {};
double trafficlight_IntersectionId[2000] = {};
double trafficlight_RoadId[2000] = {};
double trafficlight_Id[2000] = {};
int read_index_busstop = 0;
int read_index_trafficlight = 0;
bool busstop_ini = false;
bool trafficlight_ini = false;
bool use_virtual_objects_ = false;

// bool enable_avoidance_ = false;
// bool force_disable_avoidance_ = false;
// int obs_index_base = 0;
// double project_dis = 100;

std::string toBinary(int n)
{
  std::string r;
  while(n!=0){r=(n%2==0 ? "0":"1")+r;n/=2;}
  int length = r.length();
  for (int i=0;i<8-length;i++)
  {
      r = "0" + r;
  }
  return r;
}

template <int size_readtmp>
void read_busstop_txt(std::string fpname, double (&BusStop_BusStopNum)[size_readtmp],double (&BusStop_BuildingNum)[size_readtmp],double (&BusStop_BusStopId)[size_readtmp])
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
      BusStop_BusStopNum[read_index_busstop] = atof(token.c_str());
      getline(ss,token, ',');
      BusStop_BuildingNum[read_index_busstop] = atof(token.c_str());
      getline(ss,token, ',');
      BusStop_BusStopId[read_index_busstop] = atof(token.c_str());
      read_index_busstop += 1;
    }
}

void Ini_busstop_bytxt()
{
  std::string fpname = ros::package::getPath("planning_initial");
  // std::string fpname_s = fpname + "/data/ITRI_HDmap_bus_stop_info.txt";
  // std::string fpname_s = fpname + "/data/" + location_name_ + "_HDmap_bus_stop_info.txt";
  std::string fpname_s = fpname + "/data/" + location_name_ + "/" + location_name_ + "_HDmap_bus_stop_info.txt";

  std::cout << "Ini_busstop_bytxt : " << fpname_s << std::endl;

  read_busstop_txt(fpname_s, busstop_BusStopNum, busstop_BuildingNum, busstop_BusStopId);

  busstop_ini = true;
}

template <int size_readtmp>
void read_traffic_light_txt(std::string fpname, double (&TrafficLight_IntersectionId)[size_readtmp],double (&TrafficLight_RoadId)[size_readtmp],double (&TrafficLight_Id)[size_readtmp])
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
      TrafficLight_IntersectionId[read_index_trafficlight] = atof(token.c_str());
      getline(ss,token, ',');
      TrafficLight_RoadId[read_index_trafficlight] = atof(token.c_str());
      getline(ss,token, ',');
      TrafficLight_Id[read_index_trafficlight] = atof(token.c_str());
      read_index_trafficlight += 1;
    }
}

void Ini_traffic_light_bytxt()
{
  std::string fpname = ros::package::getPath("planning_initial");
  // std::string fpname_s = fpname + "/data/ITRI_HDmap_traffic_light_info.txt";
  // std::string fpname_s = fpname + "/data/" + location_name_ + "_HDmap_traffic_light_info.txt";
  std::string fpname_s = fpname + "/data/" + location_name_ + "/" + location_name_ + "_HDmap_traffic_light_info.txt";

  std::cout << "Ini_traffic_light_bytxt : " << fpname_s << std::endl;

  read_traffic_light_txt(fpname_s, trafficlight_IntersectionId, trafficlight_RoadId, trafficlight_Id);

  trafficlight_ini = true;
}

void CurrentPoseCallback(const geometry_msgs::PoseStamped& CPmsg)
{
  current_pose = CPmsg;
  geometry_msgs::PoseStamped rear_pose = current_pose;

  // double roll, pitch, yaw;
  tf::Quaternion lidar_q(CPmsg.pose.orientation.x, CPmsg.pose.orientation.y, CPmsg.pose.orientation.z,CPmsg.pose.orientation.w);
  tf::Matrix3x3 lidar_m(lidar_q);
  lidar_m.getRPY(current_roll, current_pitch, current_yaw);

  rear_pose.pose.position.x = current_pose.pose.position.x - wheel_dis*std::cos(current_yaw);
  rear_pose.pose.position.y = current_pose.pose.position.y - wheel_dis*std::sin(current_yaw);
  rearcurrentpose_pub.publish(rear_pose);

  current_pose_init_flag = true;
}

// bool transformPose(
//   const geometry_msgs::PoseStamped & input_pose, geometry_msgs::PoseStamped * output_pose,
//   const std::string target_frame)
// {
//   tf2_ros::Buffer tf_buffer_;
//   tf2_ros::TransformListener tfListener(tf_buffer_);
//   sleep(1);
//   geometry_msgs::TransformStamped transform;
//   try {
//     transform = tf_buffer_.lookupTransform(target_frame, input_pose.header.frame_id, ros::Time(0));
//     tf2::doTransform(input_pose, *output_pose, transform);
//     return true;
//   } catch (tf2::TransformException & ex) {
//     ROS_WARN("%s", ex.what());
//     return false;
//   }
// }

void objectsCallback(const autoware_perception_msgs::DynamicObjectArray& objectsmsg)
{
  autoware_perception_msgs::DynamicObjectArray object = objectsmsg;
  int size = object.objects.size();
  for (int i = 0; i < size; i++)
  {
    // object.header.frame_id = "map";
    // object.header.stamp = ros::Time::now();
    // object.objects[i].semantic.confidence = 1.0;

    // object.objects[i].state.pose_covariance.pose.position.x = 2039.41529092;
    // object.objects[i].state.pose_covariance.pose.position.y = 41618.2901704;
    // object.objects[i].state.pose_covariance.pose.position.z = -4.5338178017;
    // object.objects[i].state.pose_covariance.pose.orientation.x = 0.00383098085566;
    // object.objects[i].state.pose_covariance.pose.orientation.y = -0.00115185644512;
    // object.objects[i].state.pose_covariance.pose.orientation.z = -0.951199478151;
    // object.objects[i].state.pose_covariance.pose.orientation.w = 0.30855072448;

    // object.objects[i].state.twist_covariance.twist.linear.x = 0;
    // object.objects[i].state.twist_covariance.twist.linear.y = 0;
    // object.objects[i].state.twist_covariance.twist.linear.z = 0;
    // object.objects[i].state.twist_covariance.twist.angular.x = 0;
    // object.objects[i].state.twist_covariance.twist.angular.y = 0;
    // object.objects[i].state.twist_covariance.twist.angular.z = 0;

    // object.objects[i].state.pose_covariance.covariance = {9.999999747378752e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.999999747378752e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.999999747378752e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007615434937179089};
    // object.objects[i].shape.type = 0;
    // object.objects[i].shape.dimensions.x = 1.5;
    // object.objects[i].shape.dimensions.y = 1.5;
    // object.objects[i].shape.dimensions.z = 1.5;

    int size_poly = object.objects[i].shape.footprint.points.size();
    for (int j = 0; j < size_poly; j++)
    {
      geometry_msgs::Point32 input_point;
      geometry_msgs::Point32 out_point;

      input_point.x = object.objects[i].shape.footprint.points[j].x;
      input_point.y = object.objects[i].shape.footprint.points[j].y;
      input_point.z = object.objects[i].shape.footprint.points[j].z;

      out_point.x = current_pose.pose.position.x + input_point.x*std::cos(current_yaw) - input_point.y*std::sin(current_yaw);
      out_point.y = current_pose.pose.position.y + input_point.x*std::sin(current_yaw) + input_point.y*std::cos(current_yaw);
      out_point.z = current_pose.pose.position.z + input_point.z;
      object.objects[i].shape.footprint.points[j] = out_point;
    }
  }
  objects_pub.publish(object);
}

void objectsPub()
{
  autoware_perception_msgs::DynamicObjectArray object;
  autoware_perception_msgs::DynamicObject object_;
  // int size = 1;
  // for (int i = 0; i < size; i++)
  // {
    object.header.frame_id = "map";
    object.header.stamp = ros::Time::now();
    object_.semantic.confidence = 1.0;

    object_.state.pose_covariance.pose.position.x = 0.0;//2039.41529092;
    object_.state.pose_covariance.pose.position.y = 0.0;//41618.2901704;
    object_.state.pose_covariance.pose.position.z = -4.5338178017;
    object_.state.pose_covariance.pose.orientation.x = 0.00383098085566;
    object_.state.pose_covariance.pose.orientation.y = -0.00115185644512;
    object_.state.pose_covariance.pose.orientation.z = -0.951199478151;
    object_.state.pose_covariance.pose.orientation.w = 0.30855072448;

    object_.state.twist_covariance.twist.linear.x = 0;
    object_.state.twist_covariance.twist.linear.y = 0;
    object_.state.twist_covariance.twist.linear.z = 0;
    object_.state.twist_covariance.twist.angular.x = 0;
    object_.state.twist_covariance.twist.angular.y = 0;
    object_.state.twist_covariance.twist.angular.z = 0;

    // object.objects[i].state.pose_covariance.covariance = {9.999999747378752e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.999999747378752e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.999999747378752e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007615434937179089};
    object_.shape.type = 0;
    object_.shape.dimensions.x = 3;
    object_.shape.dimensions.y = 1.5;
    object_.shape.dimensions.z = 1.5;

    geometry_msgs::Point32 point_1;
    point_1.x = 1.5+0.0;//2039.41529092;
    point_1.y = 1.5+0.0;//41618.2901704;
    point_1.z = 5-4.5338178017;
    object_.shape.footprint.points.push_back(point_1);
    geometry_msgs::Point32 point_2;
    point_2.x = 1.5+0.0;//2039.41529092;
    point_2.y = -1.5+0.0;//41618.2901704;
    point_2.z = 5-4.5338178017;
    object_.shape.footprint.points.push_back(point_2);
    geometry_msgs::Point32 point_3;
    point_3.x = -1.5+0.0;//2039.41529092;
    point_3.y = -1.5+0.0;//41618.2901704;
    point_3.z = 5-4.5338178017;
    object_.shape.footprint.points.push_back(point_3);
    geometry_msgs::Point32 point_4;
    point_2.x = -1.5+0.0;//2039.41529092;
    point_2.y = 1.5+0.0;//41618.2901704;
    point_2.z = 5-4.5338178017;
    object_.shape.footprint.points.push_back(point_4);

    object.objects.push_back(object_);
  // }
  objects_pub.publish(object);
}

void LidnogroundpointCallback(const sensor_msgs::PointCloud2& Lngpmsg)
{
  nogroundpoints_pub.publish(Lngpmsg);
}

void currentVelocityCallback(const msgs::VehInfo::ConstPtr& msg)
{
  geometry_msgs::TwistStamped veh_twist;
  veh_twist.header.frame_id = "rear_wheel";
  veh_twist.header.stamp = ros::Time::now();
  veh_twist.twist.linear.x = msg->ego_speed;
  veh_twist.twist.angular.x = imu_data_rad.angular_velocity.x;
  veh_twist.twist.angular.y = imu_data_rad.angular_velocity.y;
  veh_twist.twist.angular.z = imu_data_rad.angular_velocity.z;
  twist_pub.publish(veh_twist);
}

void imudataCallback(const sensor_msgs::Imu& msg)
{
  imu_data_rad = msg;
}

// void trafficCallback(const msgs::Spat::ConstPtr& msg)
// {
//   int signal_state = msg->signal_state;
//   int light_status = (int)(msg->spat_state);
//   int RoadId = msg->road_id;
//   int IntersectionId = msg->intersection_id;
//   double confidence = 1.0;
//   autoware_perception_msgs::LampState lampstate;
//   autoware_perception_msgs::TrafficLightState trafficlightstate;
//   autoware_perception_msgs::TrafficLightStateArray trafficlightstatearray;
//   trafficlightstatearray.header.frame_id = "map";
//   trafficlightstatearray.header.stamp = ros::Time::now();
//   if (trafficlight_ini)
//   {
//     for (int i = 0; i < read_index_trafficlight; i++)
//     {
//       if (IntersectionId == trafficlight_IntersectionId[i] && RoadId == trafficlight_RoadId[i])
//       {
//         trafficlightstate.id = trafficlight_Id[i];
//       }
//     }
//     // trafficlightstate.id = 402079;
//   }
  
//   if (signal_state == 0)
//   {
//     if (light_status == 129) // red
//     {
//       lampstate.type = autoware_perception_msgs::LampState::RED;
//       lampstate.confidence = confidence;
//       trafficlightstate.lamp_states.push_back(lampstate);
//     }
//     else if(light_status == 130) // yellow
//     {
//       lampstate.type = autoware_perception_msgs::LampState::YELLOW;
//       lampstate.confidence = confidence;
//       trafficlightstate.lamp_states.push_back(lampstate);
//     }
//     else if(light_status == 48) // green straight + green right
//     {
//       // lampstate.type = autoware_perception_msgs::LampState::GREEN;
//       // lampstate.confidence = confidence;
//       // trafficlightstate.lamp_states.push_back(lampstate);
//       lampstate.type = autoware_perception_msgs::LampState::UP;
//       lampstate.confidence = confidence;
//       trafficlightstate.lamp_states.push_back(lampstate);
//       lampstate.type = autoware_perception_msgs::LampState::RIGHT;
//       lampstate.confidence = confidence;
//       trafficlightstate.lamp_states.push_back(lampstate);
//     }
//     else if(light_status == 9) // red + green left
//     {
//       lampstate.type = autoware_perception_msgs::LampState::RED;
//       lampstate.confidence = confidence;
//       trafficlightstate.lamp_states.push_back(lampstate);
//       lampstate.type = autoware_perception_msgs::LampState::LEFT;
//       lampstate.confidence = confidence;
//       trafficlightstate.lamp_states.push_back(lampstate);
//     }
//     else // unknown
//     {
//       lampstate.type = autoware_perception_msgs::LampState::UNKNOWN;
//       lampstate.confidence = 0.0;
//       trafficlightstate.lamp_states.push_back(lampstate);
//     }
//   }
//   else
//   {
//     lampstate.type = autoware_perception_msgs::LampState::UNKNOWN;
//     lampstate.confidence = 0.0;
//     trafficlightstate.lamp_states.push_back(lampstate);
//   }
//   trafficlightstatearray.states.push_back(trafficlightstate);
//   trafficlight_pub.publish(trafficlightstatearray);
// }

void trafficCallback(const msgs::Spat::ConstPtr& msg)
{
  int signal_state = msg->signal_state;
  int light_status = (int)(msg->spat_state);
  int RoadId = msg->road_id;
  int IntersectionId = msg->intersection_id;
  double confidence = 1.0;
  autoware_perception_msgs::LampState lampstate;
  autoware_perception_msgs::TrafficLightState trafficlightstate;
  autoware_perception_msgs::TrafficLightStateArray trafficlightstatearray;
  trafficlightstatearray.header.frame_id = "map";
  trafficlightstatearray.header.stamp = ros::Time::now();
  if (trafficlight_ini)
  {
    for (int i = 0; i < read_index_trafficlight; i++)
    {
      if (IntersectionId == trafficlight_IntersectionId[i] && RoadId == trafficlight_RoadId[i])
      {
        trafficlightstate.id = trafficlight_Id[i];
      }
    }
    // trafficlightstate.id = 402079;
  }
  
  if (signal_state == 0)
  {
    // 0:ped red 1:ped green 2:right 3:straight 4:left 5:green 6:yellow 7:red
    std::string light_status_binary = toBinary(light_status);
    // std::cout << "light_status_binary = " << light_status_binary << std::endl;
    // std::cout << "light_status_binary[0] = " << int(light_status_binary[0]) << std::endl;
    // std::cout << "light_status_binary[1] = " << int(light_status_binary[1]) << std::endl;
    if (light_status_binary[2] == '1') // right
    {
      lampstate.type = autoware_perception_msgs::LampState::RIGHT;
      lampstate.confidence = confidence;
      trafficlightstate.lamp_states.push_back(lampstate);
    }
    if (light_status_binary[3] == '1') // straight
    {
      lampstate.type = autoware_perception_msgs::LampState::UP;
      lampstate.confidence = confidence;
      trafficlightstate.lamp_states.push_back(lampstate);
    }
    if (light_status_binary[4] == '1') // left
    {
      lampstate.type = autoware_perception_msgs::LampState::LEFT;
      lampstate.confidence = confidence;
      trafficlightstate.lamp_states.push_back(lampstate);
    }
    if (light_status_binary[5] == '1') // green
    {
      lampstate.type = autoware_perception_msgs::LampState::GREEN;
      lampstate.confidence = confidence;
      trafficlightstate.lamp_states.push_back(lampstate);
    }
    if (light_status_binary[6] == '1') // yellow
    {
      lampstate.type = autoware_perception_msgs::LampState::YELLOW;
      lampstate.confidence = confidence;
      trafficlightstate.lamp_states.push_back(lampstate);
    }
    if (light_status_binary[7] == '1') // red
    {
      // std::cout << "REDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD " << std::endl;
      lampstate.type = autoware_perception_msgs::LampState::RED;
      lampstate.confidence = confidence;
      trafficlightstate.lamp_states.push_back(lampstate);
    }
    if (trafficlightstate.lamp_states.empty())// unknown
    {
      lampstate.type = autoware_perception_msgs::LampState::UNKNOWN;
      lampstate.confidence = 0.0;
      trafficlightstate.lamp_states.push_back(lampstate);
    }
  }
  else
  {
    lampstate.type = autoware_perception_msgs::LampState::UNKNOWN;
    lampstate.confidence = 0.0;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  trafficlightstatearray.states.push_back(trafficlightstate);
  trafficlight_pub.publish(trafficlightstatearray);
}

void trafficDspaceCallback(const msgs::Flag_Info::ConstPtr& msg)
{
  int light_status = int(msg->Dspace_Flag02);
  int countDown = int(msg->Dspace_Flag03);
  double confidence = 1.0;
  autoware_perception_msgs::LampState lampstate;
  autoware_perception_msgs::TrafficLightState trafficlightstate;
  autoware_perception_msgs::TrafficLightStateArray trafficlightstatearray;
  trafficlightstatearray.header.frame_id = "map";
  trafficlightstatearray.header.stamp = ros::Time::now();
  trafficlightstate.id = 402079;
  if (light_status == 129) // red
  {
    lampstate.type = autoware_perception_msgs::LampState::RED;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  else if(light_status == 130) // yellow
  {
    lampstate.type = autoware_perception_msgs::LampState::YELLOW;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  else if(light_status == 48) // green straight + green right
  {
    lampstate.type = autoware_perception_msgs::LampState::GREEN;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
    lampstate.type = autoware_perception_msgs::LampState::UP;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
    lampstate.type = autoware_perception_msgs::LampState::RIGHT;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  else if(light_status == 9) // red + green left
  {
    lampstate.type = autoware_perception_msgs::LampState::RED;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
    lampstate.type = autoware_perception_msgs::LampState::LEFT;
    lampstate.confidence = confidence;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  else // unknown
  {
    lampstate.type = autoware_perception_msgs::LampState::UNKNOWN;
    lampstate.confidence = 0.0;
    trafficlightstate.lamp_states.push_back(lampstate);
  }
  // trafficlightstatearray.states.push_back(trafficlightstate);
  // trafficlight_pub.publish(trafficlightstatearray);
}

void busstopinfoCallback(const msgs::Flag_Info::ConstPtr& msg)
{
  float Dspace_Flag[8] = {msg->Dspace_Flag01,msg->Dspace_Flag02,msg->Dspace_Flag03,msg->Dspace_Flag04,msg->Dspace_Flag05,msg->Dspace_Flag06,msg->Dspace_Flag07,msg->Dspace_Flag08};
  // float BuildingNum[8] = {0,0,0,0,0,0,0,0}; //{11,51,71,58,14,0,0,0};
  // float BusStopID[8] = {0,0,0,0,0,0,0,0}; //{402087,402140,402118,402125,402132,0,0,0};
  msgs::BusStopArray busstoparray;
  busstoparray.header.frame_id = "map";
  busstoparray.header.stamp = ros::Time::now();
  if (busstop_ini)
  {
    if (read_index_busstop > 8)
    {
      ROS_ERROR("Bus stop size error !");
    }
    for (int i = 0; i < read_index_busstop; i++)
    {
      if (Dspace_Flag[i] == 1)
      {
        msgs::BusStop busstop;
        busstop.BuildingNum = busstop_BuildingNum[i];
        busstop.BusStopId = busstop_BusStopId[i];
        busstoparray.busstops.push_back(busstop);
      }
    }
    busstop_pub.publish(busstoparray);
  }
}

void occgridCallback(const nav_msgs::OccupancyGrid& costmap)
{
  nav_msgs::OccupancyGrid costmap_maporigin = costmap;
  if (current_pose_init_flag)
  {
    costmap_maporigin.info.origin = current_pose.pose;
    costmap_maporigin.info.origin.position.x = current_pose.pose.position.x - 15*std::cos(current_yaw) + 15*std::sin(current_yaw);
    costmap_maporigin.info.origin.position.y = current_pose.pose.position.y - 15*std::sin(current_yaw) - 15*std::cos(current_yaw);
    costmap_maporigin.header.frame_id = "map";
  }
  // int height = costmap_maporigin.info.height;
  // int width = costmap_maporigin.info.width;
  // for (int i = 0; i < height; i++)
  // {
  //   for (int j = 0; j < width; j++)
  //   {
  //     int og_index = i * width + j;
  //     costmap_maporigin.data[og_index] = 0;
  //   }
  // }
  occ_maporigin_pub.publish(costmap_maporigin);
}

void occgridwayareaCallback(const nav_msgs::OccupancyGrid& costmap)
{
  nav_msgs::OccupancyGrid costmap_maporigin = costmap;
  if (current_pose_init_flag)
  {
    costmap_maporigin.info.origin = current_pose.pose;
    costmap_maporigin.info.origin.position.x = current_pose.pose.position.x - 15*std::cos(current_yaw) + 15*std::sin(current_yaw);
    costmap_maporigin.info.origin.position.y = current_pose.pose.position.y - 15*std::sin(current_yaw) - 15*std::cos(current_yaw);
    costmap_maporigin.header.frame_id = "map";
  }
  // int height = costmap_maporigin.info.height;
  // int width = costmap_maporigin.info.width;
  // for (int i = 0; i < height; i++)
  // {
  //   for (int j = 0; j < width; j++)
  //   {
  //     int og_index = i * width + j;
  //     costmap_maporigin.data[og_index] = 0;
  //   }
  // }
  occ_wayareamaporigin_pub.publish(costmap_maporigin);
}

// void obsdisbaseCallback(const std_msgs::Float64::ConstPtr& obsdismsg_base)
// {
//   double obswaypoints_data_base = obsdismsg_base->data;
//   if (obswaypoints_data_base > 40) ///////////////////////
//   {
//     obswaypoints_data_base = -1;
//   }

//   // if (obswaypoints_data_base == -1)
//   // {
//   //   obs_index_base += 1;
//   // }
//   // else
//   // {
//   //   obs_index_base = 0;
//   // }

//   std_msgs::Int32 avoid_path;
//   // if (obs_index_base > 20)
//   // {
//   //   avoid_path.data = 1;
//   // }
//   // else
//   // {
//   //   avoid_path.data = 0;
//   // }

//   if (enable_avoidance_ == 1 && obswaypoints_data_base == -1 && project_dis < 0.5)
//   {
//     avoid_path.data = 1;
//   }
//   else
//   {
//     avoid_path.data = 0;
//   }
  
//   avoiding_path_pub.publish(avoid_path);
// }

// void avoidstatesubCallback(const msgs::Flag_Info& msg)
// {
//   double avoid_state_index_ = msg.Dspace_Flag03;
//   // std::cout << "avoid_state_index_ : " << avoid_state_index_ << std::endl;
//   std_msgs::Bool enable_avoidance;// = false;
//   // std::cout << "force_disable_avoidance_ : " << force_disable_avoidance_ << std::endl;
//   if (avoid_state_index_ == 1 && !force_disable_avoidance_)
//   {
//     enable_avoidance.data = true;
//   }
//   else
//   {
//     enable_avoidance.data = false;
//   }
//   enable_avoidance_ = enable_avoidance.data;
//   enable_avoid_pub.publish(enable_avoidance);
// }

// void overshootorigdisCallback(const std_msgs::Float64& msg)
// {
//   project_dis = msg.data;
// }

int main(int argc, char** argv)
{
  ros::init(argc, argv, "planning_initial");
  ros::NodeHandle node;

  // ros::param::get(ros::this_node::getName()+"/force_disable_avoidance", force_disable_avoidance_);
  ros::param::get(ros::this_node::getName()+"/use_virtual_objects", use_virtual_objects_);
  ros::param::get(ros::this_node::getName()+"/location_name", location_name_);

  ros::Subscriber current_pose_sub = node.subscribe("current_pose", 1, CurrentPoseCallback);
  ros::Subscriber objects_sub = node.subscribe("input/objects", 1, objectsCallback);
  ros::Subscriber Lidnogroundpoint_sub = node.subscribe("input/lidar_no_ground", 1, LidnogroundpointCallback); // /LidarAll/NonGround2
  ros::Subscriber velocity_sub = node.subscribe("veh_info",1,currentVelocityCallback);
  ros::Subscriber imu_data_sub = node.subscribe("imu_data_rad",1,imudataCallback);
  ros::Subscriber traffic_sub = node.subscribe("/traffic", 1, trafficCallback);
  ros::Subscriber traffic_Dspace_sub = node.subscribe("/Flag_Info02", 1, trafficDspaceCallback);
  ros::Subscriber busstop_info_sub = node.subscribe("/BusStop/Info", 1, busstopinfoCallback);
  ros::Subscriber occ_grid_sub = node.subscribe("occupancy_grid", 1, occgridCallback);
  ros::Subscriber occ_grid_wayarea_sub = node.subscribe("occupancy_grid_wayarea", 1, occgridwayareaCallback);
  // ros::Subscriber avoid_state_sub = node.subscribe("Flag_Info01", 1, avoidstatesubCallback);
  // ros::Subscriber obstacle_dis_1_sub = node.subscribe("Geofence_original", 1, obsdisbaseCallback);
  // ros::Subscriber veh_overshoot_orig_dis_sub = node.subscribe("veh_overshoot_orig_dis", 1, overshootorigdisCallback);
  rearcurrentpose_pub = node.advertise<geometry_msgs::PoseStamped>("rear_current_pose", 1, true);
  objects_pub = node.advertise<autoware_perception_msgs::DynamicObjectArray>("output/objects", 1, true);
  nogroundpoints_pub = node.advertise<sensor_msgs::PointCloud2>("output/lidar_no_ground", 1, true);
  twist_pub = node.advertise<geometry_msgs::TwistStamped>("/localization/twist", 1, true);
  trafficlight_pub = node.advertise<autoware_perception_msgs::TrafficLightStateArray>("output/traffic_light", 1, true);
  busstop_pub = node.advertise<msgs::BusStopArray>("BusStop/Reserve", 1, true);
  occ_maporigin_pub = node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_maporigin", 1, true);
  occ_wayareamaporigin_pub = node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_wayarea_maporigin", 1, true);
  // enable_avoid_pub = node.advertise<std_msgs::Bool>("/planning/scenario_planning/lane_driving/motion_planning/obstacle_avoidance_planner/enable_avoidance", 10, true);
  // avoiding_path_pub = node.advertise<std_msgs::Int32>("avoidpath_reach_goal", 10, true);

  Ini_busstop_bytxt();
  Ini_traffic_light_bytxt();
  if (use_virtual_objects_)
  {
    ros::Rate loop_rate(10);
    while (ros::ok())
    { 
      objectsPub();
      ros::spinOnce();
      loop_rate.sleep();   
    }
  }

  ros::spin();
  return 0;
};
