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
ros::Publisher path_vehicle_pub;
ros::Publisher path_vehicle_left_pub;
ros::Publisher path_vehicle_right_pub;
nav_msgs::OccupancyGrid costmap_;
nav_msgs::OccupancyGrid lidcostmap;
nav_msgs::OccupancyGrid camcostmap;
nav_msgs::OccupancyGrid ppcostmap;
nav_msgs::OccupancyGrid costmap_sensor_all;
nav_msgs::OccupancyGrid costmap_all;
nav_msgs::OccupancyGrid costmap_all_expand;
nav_msgs::OccupancyGrid wayareaoccgridmap;

bool lid_ini = false;
bool cam_ini = false;
bool wayarea_ini = false;
bool pp_ini = false;
double expand_size = 1.0;
double expand_size_0 = 1.4;

double right_waylength = 2.5;
double left_waylength = 5;

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

void pp_occgridCallback(const nav_msgs::OccupancyGrid& costmap)
{
  ppcostmap = costmap;
  pp_ini = true;
}

void basewaypoints30Callback(const nav_msgs::Path& path)
{
  double path_max_x = 0;
  nav_msgs::Path path_vehicle = path;
  // std::cout << "path size : " << path.poses.size() << std::endl;
  double rot_ang = -current_pose.yaw;
  for (int i = 0; i < path.poses.size(); i++)
  {
    path_vehicle.poses[i].pose.position.x = (path.poses[i].pose.position.x-current_pose.x)*std::cos(rot_ang) - (path.poses[i].pose.position.y-current_pose.y)*std::sin(rot_ang);
    path_vehicle.poses[i].pose.position.y = (path.poses[i].pose.position.x-current_pose.x)*std::sin(rot_ang) + (path.poses[i].pose.position.y-current_pose.y)*std::cos(rot_ang);
    path_vehicle.poses[i].pose.position.z = -3;
    if (path_vehicle.poses[i].pose.position.x > path_max_x)
    {
      path_max_x = path_vehicle.poses[i].pose.position.x;
    }
  }
  path_vehicle.header.frame_id = "base_link";
  path_vehicle.header.stamp = ros::Time::now();
  path_vehicle_pub.publish(path_vehicle);




  nav_msgs::Path path_vehicle_left;
  geometry_msgs::PoseStamped path_vehicle_left_pose;
  nav_msgs::Path path_vehicle_right;
  geometry_msgs::PoseStamped path_vehicle_right_pose;
  for (int i = 1; i < path_vehicle.poses.size(); i+=3)
  {
    double a = path_vehicle.poses[i].pose.position.y - path_vehicle.poses[i-1].pose.position.y;
    double b = path_vehicle.poses[i-1].pose.position.x - path_vehicle.poses[i].pose.position.x;
    double dis_ab = std::sqrt(a*a + b*b);
    double x_c = (path_vehicle.poses[i].pose.position.x + path_vehicle.poses[i-1].pose.position.x) / 2.0;
    double y_c = (path_vehicle.poses[i].pose.position.y + path_vehicle.poses[i-1].pose.position.y) / 2.0;
    // left point
    path_vehicle_left_pose.pose.position.x = a * (-left_waylength)/dis_ab + x_c;
    path_vehicle_left_pose.pose.position.y = b * (-left_waylength)/dis_ab + y_c;
    path_vehicle_left_pose.pose.position.z = path_vehicle.poses[i].pose.position.z;
    path_vehicle_left.poses.push_back(path_vehicle_left_pose);
    // right point
    path_vehicle_right_pose.pose.position.x = a * right_waylength/dis_ab + x_c;
    path_vehicle_right_pose.pose.position.y = b * right_waylength/dis_ab + y_c;
    path_vehicle_right_pose.pose.position.z = path_vehicle.poses[i].pose.position.z;
    path_vehicle_right.poses.push_back(path_vehicle_right_pose);
  }
  path_vehicle_left.header.frame_id = "base_link";
  path_vehicle_left.header.stamp = ros::Time::now();
  path_vehicle_left_pub.publish(path_vehicle_left);
  path_vehicle_right.header.frame_id = "base_link";
  path_vehicle_right.header.stamp = ros::Time::now();
  path_vehicle_right_pub.publish(path_vehicle_right);

  // nav_msgs::Path path_vehicle_right;
  // geometry_msgs::PoseStamped path_vehicle_right_pose;
  // for (int i = 1; i < path_vehicle.poses.size(); i+=3)
  // {
  //   double a = path_vehicle.poses[i].pose.position.y - path_vehicle.poses[i-1].pose.position.y;
  //   double b = path_vehicle.poses[i-1].pose.position.x - path_vehicle.poses[i].pose.position.x;
  //   double dis_ab = std::sqrt(a*a + b*b);
  //   double x_c = (path_vehicle.poses[i].pose.position.x + path_vehicle.poses[i-1].pose.position.x) / 2.0;
  //   double y_c = (path_vehicle.poses[i].pose.position.y + path_vehicle.poses[i-1].pose.position.y) / 2.0;
  //   path_vehicle_right_pose.pose.position.x = a * right_waylength/dis_ab + x_c;
  //   path_vehicle_right_pose.pose.position.y = b * right_waylength/dis_ab + y_c;
  //   path_vehicle_right_pose.pose.position.z = path_vehicle.poses[i].pose.position.z;
  //   path_vehicle_right.poses.push_back(path_vehicle_right_pose);
  // }
  // path_vehicle_right.header.frame_id = "base_link";
  // path_vehicle_right.header.stamp = ros::Time::now();
  // path_vehicle_right_pub.publish(path_vehicle_right);

  setwayareaoccgridmap();
  int height = wayareaoccgridmap.info.height; // lateral
  int width = wayareaoccgridmap.info.width; // longitude
  std::vector<index_range> j_index;
  // for (int j = 0; j < width; j++)
  // {
  //   double x_j = j*grid_length_x_/width - grid_length_x_/2 + grid_position_x_;
  //   int min_m = 0;
  //   double min_value = 100;
  //   for (int m = 0; m < path_vehicle.poses.size(); m++)
  //   {
  //     double m_value = std::fabs(path_vehicle.poses[m].pose.position.x - x_j);
  //     if (m_value < min_value)
  //     {
  //       min_value = m_value;
  //       min_m = m;
  //     }
  //   }

  //   double min_y = path_vehicle.poses[min_m].pose.position.y - right_waylength;
  //   double max_y = path_vehicle.poses[min_m].pose.position.y + left_waylength;
    
  //   index_range i_index_range;
  //   i_index_range.min_index = std::ceil(height*(min_y - (-grid_length_y_/2 + grid_position_y_))/grid_length_y_);
  //   if (i_index_range.min_index < 0)
  //   {
  //     i_index_range.min_index = 0;
  //   }
  //   i_index_range.max_index = std::ceil(height*(max_y - (-grid_length_y_/2 + grid_position_y_))/grid_length_y_);
  //   if (i_index_range.max_index > height)
  //   {
  //     i_index_range.max_index = height;
  //   }

  //   j_index.push_back(i_index_range);
  // }
  // for (int j = 0; j < width; j++)
  // {
  //   for (int i = j_index[j].min_index; i < j_index[j].max_index; i++)
  //   {
  //     int og_index = i * width + j;
  //     wayareaoccgridmap.data[og_index] = 0;
  //   }
  // }
  for (int j = 0; j < width; j++)
  {
    double x_j = j*grid_length_x_/width - grid_length_x_/2 + grid_position_x_;
    double x_j_max = x_j + 5;
    double x_j_min = x_j - 5;
    if (x_j > (path_max_x+left_waylength) && x_j > (path_max_x+right_waylength))
    {
      break;
    }
    for (int i = 0; i < height; i++)
    {
      int og_index = i * width + j;
      double y_i = i*grid_length_y_/height - grid_length_y_/2 + grid_position_y_;
      for (int k = 0; k < path_vehicle_left.poses.size(); k++)
      {
        double left_x_k = path_vehicle_left.poses[k].pose.position.x;
        double right_x_k = path_vehicle_right.poses[k].pose.position.x;
        double left_y_k = path_vehicle_left.poses[k].pose.position.y;
        double right_y_k = path_vehicle_right.poses[k].pose.position.y;

        if (left_x_k < x_j_max && left_x_k > x_j_min)
        {
          double path_width = right_waylength + left_waylength;
          double grid2left = std::sqrt((x_j-left_x_k)*(x_j-left_x_k) + (y_i-left_y_k)*(y_i-left_y_k));
          double grid2right = std::sqrt((x_j-right_x_k)*(x_j-right_x_k) + (y_i-right_y_k)*(y_i-right_y_k));
          if (path_width > grid2left && path_width > grid2right)
          {
            wayareaoccgridmap.data[og_index] = 0;
            break;
          }
        }
        else if (right_x_k < x_j_max && right_x_k > x_j_min)
        {
          double path_width = right_waylength + left_waylength;
          double grid2left = std::sqrt((x_j-left_x_k)*(x_j-left_x_k) + (y_i-left_y_k)*(y_i-left_y_k));
          double grid2right = std::sqrt((x_j-right_x_k)*(x_j-right_x_k) + (y_i-right_y_k)*(y_i-right_y_k));
          if (path_width > grid2left && path_width > grid2right)
          {
            wayareaoccgridmap.data[og_index] = 0;
            break;
          }
        }
      }

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
  nav_msgs::OccupancyGrid ppcostmap_ = costmap;
  if (pp_ini == true)
  {
    ppcostmap_ = ppcostmap;
    ROS_INFO("PathPredictionOutput/grid pub");
  }
  nav_msgs::OccupancyGrid wayareaoccgridmap_ = costmap;
  if (wayarea_ini == true)
  {
    wayareaoccgridmap_ = wayareaoccgridmap;
    ROS_INFO("wayarea/grid pub");
  }

  costmap_sensor_all = costmap;
  // costmap_all = costmap;
  // costmap_all_expand = costmap;

  // costmap_sensor_all = wayareaoccgridmap_;
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

      if (wayareacost_ > 0)
        continue;

      if (cost_ > 0 || lidcost > 0 || camcost > 0)
      {
        costmap_sensor_all.data[og_index] = 100;
        costmap_all.data[og_index] = 100;
        costmap_all_expand.data[og_index] = 100;
      }

      if (costmap_sensor_all.data[og_index] == 0)
      {
        for (int k = -int(expand_size/resolution) ; k < int(expand_size/resolution) ; k++)
        {
          int m = i+k;
          if (m < 0)
          {
            continue;
          }
          if (m >= height)
          {
            continue;
          }
          for (int l = -int(expand_size/resolution) ; l < int(expand_size/resolution) ; l++)
          {
            int n = j+l;
            if (n < 0)
            {
              continue;
            }
            if (n >= width)
            {
              continue;
            }

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
          {
            continue;
          }
          if (q >= height)
          {
            continue;
          }
          for (int p = -int(expand_size_0/resolution) ; p < int(expand_size_0/resolution) ; p++)
          {
            int r = j+p;
            if (r < 0)
            {
              continue;
            }
            if (r >= width)
            {
              continue;
            }

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
  ros::Subscriber pp_grid_sub = node.subscribe("PathPredictionOutput/grid", 1, pp_occgridCallback);
  ros::Subscriber basewaypoints30_sub = node.subscribe("nav_path_astar_base_30", 1, basewaypoints30Callback);
  ros::Subscriber current_pose_sub = node.subscribe("current_pose", 1, CurrentPoseCallback);

  occ_grid_sensor_all_pub = node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_sensor_all", 10, true);
  occ_grid_all_pub = node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_all", 10, true);
  occ_grid_all_expand_pub = node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_all_expand", 10, true);
  occ_grid_wayarea_pub = node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_wayarea", 10, true);
  path_vehicle_pub = node.advertise<nav_msgs::Path>("path_vehicle", 10, true);
  path_vehicle_left_pub = node.advertise<nav_msgs::Path>("path_vehicle_left", 10, true);
  path_vehicle_right_pub = node.advertise<nav_msgs::Path>("path_vehicle_right", 10, true);
  // ros::Rate loop_rate(0.0001);
  // while (ros::ok())
  // { 
  
  //   ros::spinOnce();
  //   loop_rate.sleep();   
  // }

  ros::spin();
  return 0;
};