#include "points_to_costmap.h"

// Constructor
PointsToCostmap::PointsToCostmap()
{
}

PointsToCostmap::~PointsToCostmap()
{
}

void PointsToCostmap::initGridmapParam(const grid_map::GridMap& gridmap)
{
  grid_length_x_ = gridmap.getLength().x();
  grid_length_y_ = gridmap.getLength().y();
  grid_resolution_ = gridmap.getResolution();
  grid_position_x_ = gridmap.getPosition().x();
  grid_position_y_ = gridmap.getPosition().y();
}

bool PointsToCostmap::isValidInd(const grid_map::Index& grid_ind)
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

template <typename PointT>
grid_map::Index PointsToCostmap::fetchGridIndexFromPoint(const PointT& point)
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

template <typename PointT>
std::vector<std::vector<std::vector<double>>> PointsToCostmap::assignPoints2GridCell(
    const grid_map::GridMap& gridmap, const typename pcl::PointCloud<PointT>::Ptr& in_sensor_points)
{
  double y_cell_size = std::ceil(grid_length_y_ * (1 / grid_resolution_));
  double x_cell_size = std::ceil(grid_length_x_ * (1 / grid_resolution_));
  std::vector<double> z_vec;
  std::vector<std::vector<double>> vec_y_z(y_cell_size, z_vec);
  std::vector<std::vector<std::vector<double>>> vec_x_y_z(x_cell_size, vec_y_z);

  for (const auto& point : *in_sensor_points)
  {
    grid_map::Index grid_ind = fetchGridIndexFromPoint<PointT>(point);
    if (isValidInd(grid_ind))
    {
      vec_x_y_z[grid_ind.x()][grid_ind.y()].push_back(point.z);
    }
  }
  return vec_x_y_z;
}

grid_map::Matrix PointsToCostmap::calculateCostmap(const double maximum_height_thres,
                                                   const double minimum_lidar_height_thres, const double grid_min_value,
                                                   const double grid_max_value, const grid_map::GridMap& gridmap,
                                                   const std::string& gridmap_layer_name,
                                                   const std::vector<std::vector<std::vector<double>>> grid_vec)
{
  grid_map::Matrix gridmap_data = gridmap[gridmap_layer_name];
  for (size_t x_ind = 0; x_ind < grid_vec.size(); x_ind++)
  {
    for (size_t y_ind = 0; y_ind < grid_vec[0].size(); y_ind++)
    {
      if (grid_vec[x_ind][y_ind].size() == 0)
      {
        gridmap_data(x_ind, y_ind) = grid_min_value;
        continue;
      }
      for (const auto& z : grid_vec[x_ind][y_ind])
      {
        if (z > maximum_height_thres || z < minimum_lidar_height_thres)
        {
          continue;
        }
        gridmap_data(x_ind, y_ind) = grid_max_value;
        break;
      }
    }
  }
  return gridmap_data;
}

template <typename PointT>
grid_map::Matrix
PointsToCostmap::makeCostmapFromSensorPoints(const double maximum_height_thres, const double minimum_lidar_height_thres,
                                             const double grid_min_value, const double grid_max_value,
                                             const grid_map::GridMap& gridmap, const std::string& gridmap_layer_name,
                                             const typename pcl::PointCloud<PointT>::Ptr& in_sensor_points)
{
  initGridmapParam(gridmap);
  std::vector<std::vector<std::vector<double>>> grid_vec = assignPoints2GridCell<PointT>(gridmap, in_sensor_points);
  grid_map::Matrix costmap = calculateCostmap(maximum_height_thres, minimum_lidar_height_thres, grid_min_value,
                                              grid_max_value, gridmap, gridmap_layer_name, grid_vec);
  return costmap;
}

template <typename PointT>
grid_map::GridMap PointsToCostmap::makeGridMap(const typename pcl::PointCloud<PointT>::Ptr& input)
{
  grid_map::GridMap gridmap_;
  gridmap_.setFrameId("base_link");
  gridmap_.setGeometry(grid_map::Length(50, 30), 0.2, grid_map::Position(10, 0));
  // ROS_INFO("Created map with size %f x %f m (%i x %i cells).\n The center of the map is located at (%f, %f) in the %s
  // frame.",
  //          gridmap_.getLength().x(), gridmap_.getLength().y(),
  //          gridmap_.getSize() (0), gridmap_.getSize() (1),
  //          gridmap_.getPosition().x(), gridmap_.getPosition().y(), gridmap_.getFrameId().c_str());

  gridmap_.add("points_layer", grid_map::Matrix::Constant(gridmap_.getSize()(0), gridmap_.getSize()(1), 0.0));

  gridmap_["points_layer"] = makeCostmapFromSensorPoints<PointT>(5, -5, 0.0, 1.0, gridmap_, "points_layer", input);

  return gridmap_;
}

template grid_map::Index PointsToCostmap::fetchGridIndexFromPoint<PointXYZ>(const PointXYZ& point);

template grid_map::Index PointsToCostmap::fetchGridIndexFromPoint<PointXYZIL>(const PointXYZIL& point);

template std::vector<std::vector<std::vector<double>>> PointsToCostmap::assignPoints2GridCell<PointXYZ>(
    const grid_map::GridMap& gridmap, const pcl::PointCloud<PointXYZ>::Ptr& in_sensor_points);

template std::vector<std::vector<std::vector<double>>> PointsToCostmap::assignPoints2GridCell<PointXYZIL>(
    const grid_map::GridMap& gridmap, const pcl::PointCloud<PointXYZIL>::Ptr& in_sensor_points);

template grid_map::Matrix PointsToCostmap::makeCostmapFromSensorPoints<PointXYZ>(
    const double maximum_height_thres, const double minimum_lidar_height_thres, const double grid_min_value,
    const double grid_max_value, const grid_map::GridMap& gridmap, const std::string& gridmap_layer_name,
    const pcl::PointCloud<PointXYZ>::Ptr& in_sensor_points);

template grid_map::Matrix PointsToCostmap::makeCostmapFromSensorPoints<PointXYZIL>(
    const double maximum_height_thres, const double minimum_lidar_height_thres, const double grid_min_value,
    const double grid_max_value, const grid_map::GridMap& gridmap, const std::string& gridmap_layer_name,
    const pcl::PointCloud<PointXYZIL>::Ptr& in_sensor_points);

template grid_map::GridMap PointsToCostmap::makeGridMap<PointXYZ>(const pcl::PointCloud<PointXYZ>::Ptr& input);

template grid_map::GridMap PointsToCostmap::makeGridMap<PointXYZIL>(const pcl::PointCloud<PointXYZIL>::Ptr& input);
