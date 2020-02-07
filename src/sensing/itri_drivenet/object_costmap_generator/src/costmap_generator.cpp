#include "costmap_generator.h"

// OBJECTS_COSTMAP_LAYER_ = "objects_costmap";
// BLURRED_OBJECTS_COSTMAP_LAYER_ = "blurred_objects_costmap";

CosmapGenerator::CosmapGenerator()
  : grid_min_value_(0.0)
  , grid_max_value_(1.0)
  , grid_length_x_(50)
  , grid_length_y_(30)
  , grid_resolution_(0.2)
  , grid_position_x_(10)
  , grid_position_y_(0)
  , maximum_lidar_height_thres_(5)
  , minimum_lidar_height_thres_(-5)
  , layer_name_("points_layer")
{
}

CosmapGenerator::~CosmapGenerator()
{
}

grid_map::GridMap CosmapGenerator::initGridMap()
{
  grid_map::GridMap costmap_;
  costmap_.setFrameId("base_link");
  costmap_.setGeometry(grid_map::Length(grid_length_x_, grid_length_y_), grid_resolution_,
                       grid_map::Position(grid_position_x_, grid_position_y_));
  // ROS_INFO("Created map with size %f x %f m (%i x %i cells).\n The center of the map is located at (%f, %f) in the %s
  // frame.",
  //           costmap_.getLength().x(), costmap_.getLength().y(),
  //           costmap_.getSize() (0), costmap_.getSize() (1),
  //           costmap_.getPosition().x(), costmap_.getPosition().y(), costmap_.getFrameId().c_str());
  // std::cout << "layer_name_  " << layer_name_ <<  std::endl;
  Eigen::Array2i size;
  size(0) = costmap_.getSize()(0);
  size(1) = costmap_.getSize()(1);

  costmap_.add(layer_name_, grid_map::Matrix::Constant(size(0), size(1), 0.0));
  return costmap_;
}
grid_map::Index CosmapGenerator::fetchGridIndexFromPoint(msgs::PointXYZ point)
{
  // calculate out_grid_map position
  const double origin_x_offset = grid_length_x_ / 2.0 - grid_position_x_;
  const double origin_y_offset = grid_length_y_ / 2.0 - grid_position_y_;
  // coordinate conversion for making index. Set bottom left to the origin of coordinate (0, 0) in gridmap area
  double mapped_x = (grid_length_x_ - origin_x_offset - point.x) / grid_resolution_;
  double mapped_y = (grid_length_y_ - origin_y_offset - point.y) / grid_resolution_;

  int mapped_x_ind = std::ceil(mapped_x);
  int mapped_y_ind = std::ceil(mapped_y);
  grid_map::Index index(mapped_x_ind, mapped_y_ind);
  return index;
}
bool CosmapGenerator::isValidInd(const grid_map::Index grid_ind)
{
  bool is_valid = false;
  int x_grid_ind = grid_ind.x();
  int y_grid_ind = grid_ind.y();
  if (x_grid_ind >= 0 && x_grid_ind < std::ceil(grid_length_x_ * (1 / grid_resolution_)) && y_grid_ind >= 0 &&
      y_grid_ind < std::ceil(grid_length_y_ * (1 / grid_resolution_)))
  {
    is_valid = true;
  }
  return is_valid;
}

// midpoint of a line
msgs::PointXYZ CosmapGenerator::makeMidpoint(msgs::PointXYZ p1, msgs::PointXYZ p2)
{
  msgs::PointXYZ midPoint_;
  midPoint_.x = (p1.x + p2.x) / 2;
  midPoint_.y = (p1.y + p2.y) / 2;
  midPoint_.z = (p1.z + p2.z) / 2;
  return midPoint_;
}

msgs::ConvexPoint CosmapGenerator::makePolygonFromObjectBox(const msgs::DetectedObject in_objects)
{
  msgs::ConvexPoint convexPoint_;
  msgs::PointXYZ mid_p0_p3, mid_p0_p4, mid_p3_p7, mid_p4_p7;
  // double expand_polygon_size = 16;  // default value
  std::vector<msgs::PointXYZ> lowerAreaPoints_;
  /// 3D bounding box
  ///   p5------p6
  ///   /|  2   /|
  /// p1-|----p2 |
  ///  |p4----|-p7
  ///  |/  1  | /
  /// p0-----P3

  mid_p0_p3 = makeMidpoint(in_objects.bPoint.p0, in_objects.bPoint.p3);
  mid_p0_p4 = makeMidpoint(in_objects.bPoint.p0, in_objects.bPoint.p4);
  mid_p3_p7 = makeMidpoint(in_objects.bPoint.p3, in_objects.bPoint.p7);
  mid_p4_p7 = makeMidpoint(in_objects.bPoint.p4, in_objects.bPoint.p7);
  lowerAreaPoints_.push_back(in_objects.bPoint.p0);
  lowerAreaPoints_.push_back(in_objects.bPoint.p3);
  lowerAreaPoints_.push_back(in_objects.bPoint.p4);
  lowerAreaPoints_.push_back(in_objects.bPoint.p7);
  lowerAreaPoints_.push_back(mid_p0_p3);
  lowerAreaPoints_.push_back(mid_p0_p4);
  lowerAreaPoints_.push_back(mid_p3_p7);
  lowerAreaPoints_.push_back(mid_p4_p7);

  // p0-----P3
  mid_p0_p3 = makeMidpoint(in_objects.bPoint.p0, mid_p0_p3);
  lowerAreaPoints_.push_back(mid_p0_p3);
  mid_p0_p3 = makeMidpoint(mid_p0_p3, in_objects.bPoint.p3);
  lowerAreaPoints_.push_back(mid_p0_p3);

  // p0-----P4
  mid_p0_p4 = makeMidpoint(in_objects.bPoint.p0, mid_p0_p4);
  lowerAreaPoints_.push_back(mid_p0_p4);
  mid_p0_p4 = makeMidpoint(mid_p0_p4, in_objects.bPoint.p4);
  lowerAreaPoints_.push_back(mid_p0_p4);

  // p3-----P7
  mid_p3_p7 = makeMidpoint(in_objects.bPoint.p3, mid_p3_p7);
  lowerAreaPoints_.push_back(mid_p3_p7);
  mid_p3_p7 = makeMidpoint(mid_p3_p7, in_objects.bPoint.p7);
  lowerAreaPoints_.push_back(mid_p3_p7);

  // p4-----P7
  mid_p4_p7 = makeMidpoint(in_objects.bPoint.p4, mid_p4_p7);
  lowerAreaPoints_.push_back(mid_p4_p7);
  mid_p4_p7 = makeMidpoint(mid_p4_p7, in_objects.bPoint.p7);
  lowerAreaPoints_.push_back(mid_p4_p7);

  convexPoint_.lowerAreaPoints = lowerAreaPoints_;
  convexPoint_.objectHigh = in_objects.bPoint.p1.z - in_objects.bPoint.p0.z;
  return convexPoint_;
}
grid_map::Matrix CosmapGenerator::makeCostmapFromSingleObject(const grid_map::GridMap costmap,
                                                         const std::string gridmap_layer_name,
                                                         const double expand_polygon_size,
                                                         const msgs::DetectedObject object,
                                                         const bool use_objects_convex_hull)
{
  grid_map::Matrix gridmap_data = costmap[gridmap_layer_name];
  msgs::ConvexPoint convexPoint_;
  if (use_objects_convex_hull) 
  {
    msgs::ConvexPoint polygonPoint_ = object.cPoint;
    if (polygonPoint_.lowerAreaPoints.size() > 0)
    {
      convexPoint_.lowerAreaPoints.insert(convexPoint_.lowerAreaPoints.end(), polygonPoint_.lowerAreaPoints.begin(), polygonPoint_.lowerAreaPoints.end());
    }
    else
    {
      return gridmap_data;
    }
  }
  else
  {
    if (object.distance >= 0) 
    {
      msgs::ConvexPoint boxConvexPoint_ = makePolygonFromObjectBox(object);
      convexPoint_.lowerAreaPoints.insert(convexPoint_.lowerAreaPoints.end(), boxConvexPoint_.lowerAreaPoints.begin(), boxConvexPoint_.lowerAreaPoints.end());
    }
    else
    {
      return gridmap_data;
    }
  }
  
  for (size_t i = 0; i < convexPoint_.lowerAreaPoints.size(); i++)
  {
    grid_map::Index grid_index_ = fetchGridIndexFromPoint(convexPoint_.lowerAreaPoints[i]);
    if (isValidInd(grid_index_))
    {
      gridmap_data(grid_index_.x(), grid_index_.y()) = grid_max_value_;
    }
  }
  return gridmap_data;
}
grid_map::Matrix CosmapGenerator::makeCostmapFromObjects(const grid_map::GridMap costmap,
                                                         const std::string gridmap_layer_name,
                                                         const double expand_polygon_size,
                                                         const msgs::DetectedObjectArray in_objects,
                                                         const bool use_objects_convex_hull)
{
  grid_map::Matrix gridmap_data = costmap[gridmap_layer_name];
  for (size_t i = 0; i < in_objects.objects.size(); i++)
  {
    if (in_objects.objects[i].distance >= 0)
    {
      msgs::ConvexPoint convexPoint_;
      if (use_objects_convex_hull) convexPoint_ = in_objects.objects[i].cPoint;
      else convexPoint_ = makePolygonFromObjectBox(in_objects.objects[i]);
      for (size_t i = 0; i < convexPoint_.lowerAreaPoints.size(); i++)
      {
        grid_map::Index grid_index_ = fetchGridIndexFromPoint(convexPoint_.lowerAreaPoints[i]);
        if (isValidInd(grid_index_))
        {
          gridmap_data(grid_index_.x(), grid_index_.y()) = grid_max_value_;
        }
      }
    }
  }
  return gridmap_data;
}

nav_msgs::OccupancyGrid CosmapGenerator::gridMapToOccupancyMsg(grid_map::GridMap& costmap, std_msgs::Header header)
{
  nav_msgs::OccupancyGrid occupancy_grid_msg_;
  grid_map::GridMapRosConverter::toOccupancyGrid(costmap, layer_name_, 0, 1, occupancy_grid_msg_);
  occupancy_grid_msg_.header = header;
  occupancy_grid_msg_.header.frame_id = "/base_link";
  return occupancy_grid_msg_;
}
int CosmapGenerator::gridMsgPublisher(grid_map::GridMap& costmap, ros::Publisher gridmap_publisher,
                                      std_msgs::Header header)
{
  grid_map_msgs::GridMap out_gridmap_msg;
  grid_map::GridMapRosConverter::toMessage(costmap, out_gridmap_msg);
  out_gridmap_msg.info.header = header;
  gridmap_publisher.publish(out_gridmap_msg);
  return true;
}
int CosmapGenerator::OccupancyMsgPublisher(grid_map::GridMap& costmap, ros::Publisher occupancy_grid_publisher,
                                           std_msgs::Header header)
{
  nav_msgs::OccupancyGrid occupancy_grid_msg_;
  occupancy_grid_msg_ = gridMapToOccupancyMsg(costmap, header);
  occupancy_grid_publisher.publish(occupancy_grid_msg_);
  return true;
}
