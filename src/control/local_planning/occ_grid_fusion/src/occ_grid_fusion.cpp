#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <cmath>
#include <vector>

#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_msgs/GridMap.h>

ros::Publisher occ_grid_all_pub;
ros::Publisher occ_grid_sensor_all_pub;
ros::Publisher occ_grid_all_expand_pub;
ros::Publisher occ_grid_wayarea_pub;
// ros::Publisher path_vehicle_pub;
nav_msgs::OccupancyGrid costmap_;
nav_msgs::OccupancyGrid lidcostmap;
nav_msgs::OccupancyGrid camcostmap;
nav_msgs::OccupancyGrid costmap_sensor_all;
nav_msgs::OccupancyGrid costmap_all;
nav_msgs::OccupancyGrid costmap_all_expand;
nav_msgs::OccupancyGrid wayareaoccgridmap;

bool lid_ini = false;
bool cam_ini = false;
bool wayarea_ini = false;
double expand_size = 1.0;
double expand_size_0 = 1.4;

double right_waylength = 2.5;
double left_waylength = 5.5;

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

struct index_range
{
  int min_index;
  int max_index;
};

double grid_length_x_ = 50;
double grid_length_y_ = 30;
double grid_resolution_ = 0.2;
int grid_position_x_ = 10;
int grid_position_y_ = 0;
double grid_min_value_ = 0.0;
double grid_max_value_ = 1.0;
double in_min_value = 0.0;
double in_max_value = 100.0;
std::string layer_name_ = "wayarea_layer";

void setwayareaoccgridmap()
{
  grid_map::GridMap wayareagridmap;
  wayareagridmap.add(layer_name_);
  wayareagridmap.setFrameId("base_link");
  wayareagridmap.setGeometry(grid_map::Length(grid_length_x_, grid_length_y_), grid_resolution_, grid_map::Position(grid_position_x_, grid_position_y_));
  wayareagridmap[layer_name_].setConstant(100.0);

  // nav_msgs::OccupancyGrid wayareaoccgridmap;
  grid_map::GridMapRosConverter::toOccupancyGrid(wayareagridmap, layer_name_, in_min_value, in_max_value, wayareaoccgridmap);
  wayareaoccgridmap.info.origin.position.z = 0;
}

void CurrentPoseCallback(const geometry_msgs::PoseStamped& CPmsg)
{
  geometry_msgs::PoseStamped pose = CPmsg;

  double roll, pitch, yaw;
  tf::Quaternion lidar_q(CPmsg.pose.orientation.x, CPmsg.pose.orientation.y, CPmsg.pose.orientation.z,CPmsg.pose.orientation.w);
  tf::Matrix3x3 lidar_m(lidar_q);
  lidar_m.getRPY(roll, pitch, yaw);

  current_pose.x = pose.pose.position.x;
  current_pose.y = pose.pose.position.y;
  current_pose.roll = roll;
  current_pose.pitch = pitch;
  current_pose.yaw = yaw;
}

void lid_occgridCallback(const nav_msgs::OccupancyGrid& costmap)
{
  lidcostmap = costmap;
  lid_ini = true;
}

void cam_occgridCallback(const nav_msgs::OccupancyGrid& costmap)
{
  camcostmap = costmap;
  cam_ini = true;
}

void basewaypoints30Callback(const nav_msgs::Path& path)
{
  nav_msgs::Path path_vehicle = path;
  // std::cout << "path size : " << path.poses.size() << std::endl;
  double rot_ang = -current_pose.yaw;
  for (int i = 0; i < path.poses.size(); i++)
  {
    path_vehicle.poses[i].pose.position.x = (path.poses[i].pose.position.x-current_pose.x)*std::cos(rot_ang) - (path.poses[i].pose.position.y-current_pose.y)*std::sin(rot_ang);
    path_vehicle.poses[i].pose.position.y = (path.poses[i].pose.position.x-current_pose.x)*std::sin(rot_ang) + (path.poses[i].pose.position.y-current_pose.y)*std::cos(rot_ang);
    // path_vehicle.poses[i].pose.position.z = 0;
  }
  // path_vehicle.header.frame_id = "base_link";
  // path_vehicle.header.stamp = ros::Time::now();
  // path_vehicle_pub.publish(path_vehicle);

  setwayareaoccgridmap();
  int height = wayareaoccgridmap.info.height; // lateral
  int width = wayareaoccgridmap.info.width; // longitude
  std::vector<index_range> j_index;
  for (int j = 0; j < width; j++)
  {
    double x_j = j*grid_length_x_/width - grid_length_x_/2 + grid_position_x_;
    int min_m = 0;
    double min_value = 100;
    for (int m = 0; m < path_vehicle.poses.size(); m++)
    {
      double m_value = std::fabs(path_vehicle.poses[m].pose.position.x - x_j);
      if (m_value < min_value)
      {
        min_value = m_value;
        min_m = m;
      }
    }

    double min_y = path_vehicle.poses[min_m].pose.position.y - right_waylength;
    double max_y = path_vehicle.poses[min_m].pose.position.y + left_waylength;
    
    index_range i_index_range;
    i_index_range.min_index = std::ceil(height*(min_y - (-grid_length_y_/2 + grid_position_y_))/grid_length_y_);
    if (i_index_range.min_index < 0)
      i_index_range.min_index = 0;
    i_index_range.max_index = std::ceil(height*(max_y - (-grid_length_y_/2 + grid_position_y_))/grid_length_y_);
    if (i_index_range.max_index > height)
      i_index_range.max_index = height;

    j_index.push_back(i_index_range);
  }
  for (int j = 0; j < width; j++)
  {
    for (int i = j_index[j].min_index; i < j_index[j].max_index; i++)
    {
      int og_index = i * width + j;
      wayareaoccgridmap.data[og_index] = 0;
    }
  }
  occ_grid_wayarea_pub.publish(wayareaoccgridmap);

  wayarea_ini = true;
}

void occgridCallback(const nav_msgs::OccupancyGrid& costmap)
{
  // std::cout << "expand_size : " << expand_size << std::endl;
  costmap_ = costmap;
  static double resolution = costmap_.info.resolution;
  costmap_sensor_all = costmap;
  nav_msgs::OccupancyGrid lidcostmap_ = costmap;
  if (lid_ini == true)
  {
    lidcostmap_ = lidcostmap;
    ROS_INFO("LidarDetection/grid pub");
  }
  nav_msgs::OccupancyGrid camcostmap_ = costmap;
  if (cam_ini == true)
  {
    camcostmap_ = camcostmap;
    ROS_INFO("CameraDetection/grid pub");
  }
  nav_msgs::OccupancyGrid wayareaoccgridmap_ = costmap;
  if (wayarea_ini == true)
  {
    wayareaoccgridmap_ = wayareaoccgridmap;
    ROS_INFO("wayarea/grid pub");
  }
  costmap_all = wayareaoccgridmap_;
  costmap_all_expand = wayareaoccgridmap_;

  int height = costmap_.info.height;
  int width = costmap_.info.width;

  // cost initialization
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      // Index of subscribing OccupancyGrid message
      int og_index = i * width + j;
      int cost_ = costmap_.data[og_index];
      int wayareacost_ = wayareaoccgridmap_.data[og_index];
      int lidcost = lidcostmap_.data[og_index];
      int camcost = camcostmap_.data[og_index];

      // if (cost_ > 0)
      //   continue;
      if (cost_ > 0 || lidcost > 0 || camcost > 0)
      {
        costmap_sensor_all.data[og_index] = 100;
        costmap_all.data[og_index] = 100;
        costmap_all_expand.data[og_index] = 100;
      }

      if (wayareacost_ > 0)
        continue;

      if (costmap_sensor_all.data[og_index] == 0)
      {
        for (int k = -int(expand_size/resolution) ; k < int(expand_size/resolution) ; k++)
        {
          int m = i+k;
          if (m < 0)
            continue;
          if (m >= height)
            continue;
          for (int l = -int(expand_size/resolution) ; l < int(expand_size/resolution) ; l++)
          {
            int n = j+l;
            if (n < 0)
              continue;
            if (n >= width)
              continue;

            int og_index_1 = m * width + n;
            if (costmap_sensor_all.data[og_index_1] > 0)
            {
              costmap_all_expand.data[og_index] = 50;
              k = int(expand_size/resolution);
              break;
            }
          }
        }
      }
      if (costmap_all_expand.data[og_index] == 50)
      {
        for (int o = -int(expand_size_0/resolution) ; o < int(expand_size_0/resolution) ; o++)
        {
          int q = i+o;
          if (q < 0)
            continue;
          if (q >= height)
            continue;
          for (int p = -int(expand_size_0/resolution) ; p < int(expand_size_0/resolution) ; p++)
          {
            int r = j+p;
            if (r < 0)
              continue;
            if (r >= width)
              continue;

            int og_index_2 = q * width + r;
            if (costmap_sensor_all.data[og_index_2] > 0)
            {
              costmap_all_expand.data[og_index] = 75;
              o = int(expand_size_0/resolution);
              break;
            }
          }
        }
      }
    }
  }
  occ_grid_sensor_all_pub.publish(costmap_sensor_all);
  occ_grid_all_pub.publish(costmap_all);
  occ_grid_all_expand_pub.publish(costmap_all_expand);
  ROS_INFO("Pub costmap_all");
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "occ_grid_fusion");
  ros::NodeHandle node;

  ros::param::get(ros::this_node::getName()+"/expand_size", expand_size);

  ros::Subscriber occ_grid_sub = node.subscribe("occupancy_grid", 1, occgridCallback);
  ros::Subscriber liddetect_grid_sub = node.subscribe("LidarDetection/grid", 1, lid_occgridCallback);
  ros::Subscriber cameradetect_grid_sub = node.subscribe("CameraDetection/occupancy_grid", 1, cam_occgridCallback);
  ros::Subscriber basewaypoints30_sub = node.subscribe("nav_path_astar_base_30", 1, basewaypoints30Callback);
  ros::Subscriber current_pose_sub = node.subscribe("current_pose", 1, CurrentPoseCallback);

  occ_grid_sensor_all_pub = node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_sensor_all", 10, true);
  occ_grid_all_pub = node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_all", 10, true);
  occ_grid_all_expand_pub = node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_all_expand", 10, true);
  occ_grid_wayarea_pub = node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_wayarea", 10, true);
  // path_vehicle_pub = node.advertise<nav_msgs::Path>("path_vehicle", 10, true);
  // ros::Rate loop_rate(0.0001);
  // while (ros::ok())
  // { 
  
  //   ros::spinOnce();
  //   loop_rate.sleep();   
  // }

  ros::spin();
  return 0;
};