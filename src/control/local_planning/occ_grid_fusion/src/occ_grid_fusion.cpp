#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>

ros::Publisher occ_grid_all_pub;
nav_msgs::OccupancyGrid costmap_;
nav_msgs::OccupancyGrid lidcostmap;
nav_msgs::OccupancyGrid camcostmap;
nav_msgs::OccupancyGrid costmap_all;
bool lid_ini = false;
bool cam_ini = false;

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
  costmap_ = costmap;
  costmap_all = costmap;
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
        costmap_all.data[og_index] = 100;
    }
  }
  occ_grid_all_pub.publish(costmap_all);
  ROS_INFO("Pub costmap_all");
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "occ_grid_fusion");
  ros::NodeHandle node;

  ros::Subscriber occ_grid_sub = node.subscribe("occupancy_grid", 1, occgridCallback);
  ros::Subscriber liddetect_grid_sub = node.subscribe("LidarDetection/grid", 1, lid_occgridCallback);
  ros::Subscriber cameradetect_grid_sub = node.subscribe("CameraDetection/grid", 1, cam_occgridCallback);

  occ_grid_all_pub = node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_all", 10, true);
  // ros::Rate loop_rate(0.0001);
  // while (ros::ok())
  // { 
  
  //   ros::spinOnce();
  //   loop_rate.sleep();   
  // }

  ros::spin();
  return 0;
};