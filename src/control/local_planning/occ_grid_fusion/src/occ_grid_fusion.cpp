#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>

ros::Publisher occ_grid_all_pub;
ros::Publisher occ_grid_all_expand_pub;
nav_msgs::OccupancyGrid costmap_;
nav_msgs::OccupancyGrid lidcostmap;
nav_msgs::OccupancyGrid camcostmap;
nav_msgs::OccupancyGrid costmap_all;
nav_msgs::OccupancyGrid costmap_all_expand;
bool lid_ini = false;
bool cam_ini = false;
double expand_size = 1.0;


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

void occgridCallback(const nav_msgs::OccupancyGrid& costmap)
{
  std::cout << "expand_size : " << expand_size << std::endl;
  costmap_ = costmap;
  static double resolution = costmap_.info.resolution;
  costmap_all = costmap;
  costmap_all_expand = costmap;
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
      int lidcost = lidcostmap_.data[og_index];
      int camcost = camcostmap_.data[og_index];

      if (cost_ > 0)
        continue;

      if (lidcost > 0 || camcost > 0)
      {
        costmap_all.data[og_index] = 100;
        costmap_all_expand.data[og_index] = 100;
      }

      if (costmap_all.data[og_index] == 0)
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
            if (costmap_all.data[og_index_1] > 0)
            {
              costmap_all_expand.data[og_index] = 50;
              k = int(expand_size/resolution);
              break;
            }
          }
        }
      }
    }
  }
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

  occ_grid_all_pub = node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_all", 10, true);
  occ_grid_all_expand_pub = node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_all_expand", 10, true);
  // ros::Rate loop_rate(0.0001);
  // while (ros::ok())
  // { 
  
  //   ros::spinOnce();
  //   loop_rate.sleep();   
  // }

  ros::spin();
  return 0;
};