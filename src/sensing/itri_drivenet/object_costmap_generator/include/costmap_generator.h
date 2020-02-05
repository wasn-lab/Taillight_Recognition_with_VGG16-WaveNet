#ifndef COSTMAP_GENERATOR_H_
#define COSTMAP_GENERATOR_H_

#include "ros/ros.h"
#include "std_msgs/String.h"
// ros - grid map
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_msgs/GridMap.h>
// msgs
#include <msgs/PointXYZ.h>
#include <msgs/ConvexPoint.h>
#include <msgs/DetectedObjectArray.h>

class CosmapGenerator
{
private:
  // grid_map
  double grid_min_value_ = 0.0;
  double grid_max_value_ = 0.0;
  double grid_length_x_ = 0.0;
  double grid_length_y_ = 0.0;
  double grid_resolution_ = 0.0;
  double grid_position_x_ = 0.0;
  double grid_position_y_ = 0.0;
  double maximum_lidar_height_thres_ = 0.0;
  double minimum_lidar_height_thres_ = 0.0;

  // functions
  msgs::ConvexPoint makePolygonFromObjectBox(const msgs::DetectedObject in_objects);
  grid_map::Index fetchGridIndexFromPoint(msgs::PointXYZ point);
  bool isValidInd(const grid_map::Index grid_ind);
  msgs::PointXYZ makeMidpoint(msgs::PointXYZ p1, msgs::PointXYZ p2);

public:
  // grid_map
  std::string layer_name_;

  // functions
  CosmapGenerator();
  ~CosmapGenerator();
  grid_map::GridMap initGridMap();
  grid_map::Matrix makeCostmapFromObjects(const grid_map::GridMap costmap, const std::string gridmap_layer_name,
                                          const double expand_polygon_size, const msgs::DetectedObjectArray in_objects,
                                          const bool use_objects_convex_hull);
  void objectMsgToGridMapMsg(const msgs::DetectedObjectArray& in_objects, ros::Publisher gridmap_publisher);
  nav_msgs::OccupancyGrid gridMapToOccupancyMsg(grid_map::GridMap& costmap, std_msgs::Header header);
  int gridMsgPublisher(grid_map::GridMap& costmap, ros::Publisher gridmap_publisher, std_msgs::Header header);
  int OccupancyMsgPublisher(grid_map::GridMap& costmap, ros::Publisher occupancy_grid_publisher,
                            std_msgs::Header header);
};
#endif /*COSTMAP_GENERATOR_H_*/