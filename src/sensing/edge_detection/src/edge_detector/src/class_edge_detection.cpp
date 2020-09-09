#include "headers.h"
#include "class_edge_detection.h"
#include <iostream>
#include <fstream>
#include <chrono>

EdgeDetection::EdgeDetection()
  : theta_sample_(360)
  , max_radius_(50)
  , seq_(0)
  , grid_length_x_(50)
  , grid_length_y_(30)
  , grid_resolution_(0.2)
  , grid_position_x_(10)
  , grid_position_y_(0)
  , grid_min_value_(0.0)
  , grid_max_value_(1.0)
  , maximum_lidar_height_thres_(5)
  , minimum_lidar_height_thres_(-5)
  , layer_name_("points_layer")
{
  release_cloud_ = PointCloud<PointXYZI>::Ptr(new PointCloud<PointXYZI>);
  ring_edge_pointCloud_ = PointCloud<PointXYZI>::Ptr(new PointCloud<PointXYZI>);
  ;
  ground_pointCloud_ = PointCloud<PointXYZI>::Ptr(new PointCloud<PointXYZI>);
  ;
  non_ground_pointCloud_ = PointCloud<PointXYZI>::Ptr(new PointCloud<PointXYZI>);
  ;
}

EdgeDetection::~EdgeDetection()
{
}

void EdgeDetection::LogTotxt(const std::vector<float> contour_distance)
{
  ofstream myfile("contour.txt");

  if (myfile.is_open())
  {
    for (int i = 0; i < contour_distance.size(); i++)
    {
      myfile << contour_distance.at(i) * 10 << std::endl;
    }
    myfile.close();
  }
}

bool EdgeDetection::RegisterCallbacks(const ros::NodeHandle& n)
{
  // Create a local nodehandle to manage callback subscriptions.
  ros::NodeHandle nl(n);
  ground_pointcloud_publisher = nl.advertise<sensor_msgs::PointCloud2>("ground_points", 1, false);

  return true;
}

void EdgeDetection::setInputCloud(const PointCloud<PointXYZI>::ConstPtr input)
{
  *release_cloud_ = *input;
}

void EdgeDetection::startThread()
{
  mthread_1 = thread(&EdgeDetection::calculate, this);
}

void EdgeDetection::waitThread()
{
  if (mthread_1.joinable())
  {
    mthread_1.join();
  }
}

void EdgeDetection::cloudPosPreprocess(const PointCloud<PointXYZI>::ConstPtr& input_cloud,
                                       pcl::PointCloud<PointXYZI>::Ptr& output_cloud, float deg_x, float deg_y,
                                       float deg_z)
{
  float Rad_x = deg_x * (M_PI / 180);
  float Rad_y = deg_x * (M_PI / 180);
  float Rad_z = deg_x * (M_PI / 180);

  Eigen::Affine3f transformer = Eigen::Affine3f::Identity();
  //    transformer =   Eigen::AngleAxisf (Rad_x, Eigen::Vector3f::UnitX())*
  //                    Eigen::AngleAxisf (Rad_y, Eigen::Vector3f::UnitY())*
  //                    Eigen::AngleAxisf (Rad_z, Eigen::Vector3f::UnitZ());
  transformer = Eigen::AngleAxisf(Rad_x, Eigen::Vector3f::UnitX());
  //    Eigen::AngleAxisf (Rad_y, Eigen::Vector3f::UnitX())*
  //    Eigen::AngleAxisf (Rad_z, Eigen::Vector3f::UnitX());

  pcl::transformPointCloud(*input_cloud, *output_cloud, transformer);
}

// points2gridmap
bool EdgeDetection::isValidInd(const grid_map::Index& grid_ind)
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

grid_map::Index EdgeDetection::fetchGridIndexFromPoint(const pcl::PointXYZI& point)
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

grid_map::Matrix EdgeDetection::calculateCostmap(const double maximum_height_thres,
                                                 const double minimum_lidar_height_thres, const double grid_min_value,
                                                 const double grid_max_value, const grid_map::GridMap& gridmap,
                                                 const std::string& gridmap_layer_name,
                                                 const std::vector<std::vector<std::vector<double> > > grid_vec)
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

grid_map::Matrix EdgeDetection::makeCostmapFromSensorPoints(
    const double maximum_height_thres, const double minimum_lidar_height_thres, const double grid_min_value,
    const double grid_max_value, const grid_map::GridMap& gridmap, const std::string& gridmap_layer_name,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& in_sensor_points)
{
  initGridmapParam(gridmap);

  std::vector<std::vector<std::vector<double> > > grid_vec = assignPoints2GridCell(gridmap, in_sensor_points);

  grid_map::Matrix costmap = calculateCostmap(maximum_height_thres, minimum_lidar_height_thres, grid_min_value,
                                              grid_max_value, gridmap, gridmap_layer_name, grid_vec);

  return costmap;
}

void EdgeDetection::initGridmapParam(const grid_map::GridMap& gridmap)
{
  grid_length_x_ = gridmap.getLength().x();
  grid_length_y_ = gridmap.getLength().y();
  grid_resolution_ = gridmap.getResolution();
  grid_position_x_ = gridmap.getPosition().x();
  grid_position_y_ = gridmap.getPosition().y();
}

std::vector<std::vector<std::vector<double> > > EdgeDetection::assignPoints2GridCell(
    const grid_map::GridMap& gridmap, const pcl::PointCloud<pcl::PointXYZI>::Ptr& in_sensor_points)
{
  double y_cell_size = std::ceil(grid_length_y_ * (1 / grid_resolution_));

  double x_cell_size = std::ceil(grid_length_x_ * (1 / grid_resolution_));
  std::vector<double> z_vec;

  std::vector<std::vector<double> > vec_y_z(y_cell_size, z_vec);

  std::vector<std::vector<std::vector<double> > > vec_x_y_z(x_cell_size, vec_y_z);
  for (const auto& point : *in_sensor_points)
  {
    grid_map::Index grid_ind = fetchGridIndexFromPoint(point);

    if (isValidInd(grid_ind))
    {
      vec_x_y_z[grid_ind.x()][grid_ind.y()].push_back(point.z);
    }
  }

  return vec_x_y_z;
}

// grid_map grnerator
void EdgeDetection::initGridmap()
{
  gridmap_.setFrameId("base_link");
  gridmap_.setGeometry(grid_map::Length(grid_length_x_, grid_length_y_), grid_resolution_,
                       grid_map::Position(grid_position_x_, grid_position_y_));
  ROS_INFO("Created map with size %f x %f m (%i x %i cells).\n The center of the map is located at (%f, %f) in the %s "
           "frame.",
           gridmap_.getLength().x(), gridmap_.getLength().y(), gridmap_.getSize()(0), gridmap_.getSize()(1),
           gridmap_.getPosition().x(), gridmap_.getPosition().y(), gridmap_.getFrameId().c_str());
  std::cout << "layer_name_  " << layer_name_ << std::endl;
  Eigen::Array2i size;
  size(0) = gridmap_.getSize()(0);
  size(1) = gridmap_.getSize()(1);

  gridmap_.add(layer_name_, grid_map::Matrix::Constant(size(0), size(1), 0.0));
}

grid_map::Matrix EdgeDetection::generatePointsGridmap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& in_sensor_points)
{
  grid_map::Matrix sensor_points_costmap =
      makeCostmapFromSensorPoints(maximum_lidar_height_thres_, minimum_lidar_height_thres_, grid_min_value_,
                                  grid_max_value_, gridmap_, layer_name_, in_sensor_points);

  return sensor_points_costmap;
}

void EdgeDetection::calculate()

{
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

  boost::shared_ptr<pcl::PointCloud<PointT> > cloud_filtered(new pcl::PointCloud<PointT>);

  // Terrain detection, ProgressiveMorphologicalFilter ------------------------------------
  boost::shared_ptr<pcl::PointCloud<PointT> > cloud_terrain(new pcl::PointCloud<PointT>);
  boost::shared_ptr<pcl::PointCloud<PointT> > cloud_non_terrain(new pcl::PointCloud<PointT>);
  *cloud_filtered = *release_cloud_;
  *cloud_terrain = *cloud_filtered;
  *cloud_non_terrain = *cloud_filtered;
  pcl::PointIndicesPtr ground(new pcl::PointIndices);

  approximateProgressiveMorphological(cloud_filtered, ground);

  extractIndice(cloud_filtered, cloud_non_terrain, ground, true);
  extractIndice(cloud_filtered, cloud_terrain, ground, false);

//  // Radius Filter Here -------------
  // boost::shared_ptr<pcl::PointCloud<PointT> > cloud_non_terrain_non_noise(new pcl::PointCloud<PointT>);
  // radiusFilter(cloud_non_terrain, cloud_non_terrain_non_noise, 0.13, 1);
  // *non_ground_pointCloud_ = *cloud_non_terrain_non_noise;
//   // --------------------------------

  *non_ground_pointCloud_ = *cloud_non_terrain;
  
  *ground_pointCloud_ = *cloud_terrain;

  boost::shared_ptr<pcl::PointCloud<PointT> > cloud_terrain_filter_i(new pcl::PointCloud<PointT>);


  // Project to z plane ------------------------------------
  boost::shared_ptr<pcl::PointCloud<PointT> > cloud_terrain_2D(new pcl::PointCloud<PointT>);
  boost::shared_ptr<pcl::PointCloud<PointT> > cloud_non_terrain_2D(new pcl::PointCloud<PointT>);
  *cloud_terrain_2D = *cloud_terrain;
  *cloud_non_terrain_2D = *cloud_non_terrain;

  std::thread tools_thread_0(projectToZPlane, std::ref(cloud_terrain), std::ref(cloud_terrain_2D));
  std::thread tools_thread_1(projectToZPlane, std::ref(cloud_non_terrain), std::ref(cloud_non_terrain_2D));

  tools_thread_0.join();
  tools_thread_1.join();
  //    **voxel downsample************************

  boost::shared_ptr<pcl::PointCloud<PointT> > sampled_cloud_terrain_2D(new pcl::PointCloud<PointT>);
  boost::shared_ptr<pcl::PointCloud<PointT> > sampled_cloud_non_terrain_2D(new pcl::PointCloud<PointT>);

  *sampled_cloud_terrain_2D = *cloud_terrain_2D;
  *sampled_cloud_non_terrain_2D = *cloud_non_terrain_2D;

  boost::shared_ptr<pcl::PointCloud<PointT> > sampled_cloud_boundary(new pcl::PointCloud<PointT>);
  boost::shared_ptr<pcl::PointCloud<PointT> > sampled_cloud_innerboundary(new pcl::PointCloud<PointT>);

  std::vector<float> contour_distance;

  getContourV2(sampled_cloud_non_terrain_2D, sampled_cloud_terrain_2D, ring_edge_pointCloud_,
               sampled_cloud_innerboundary, contour_distance, theta_sample_, max_radius_, false);

  std::vector<int> indices_contour;
  pcl::removeNaNFromPointCloud(*ring_edge_pointCloud_, *ring_edge_pointCloud_, indices_contour);

  gridmap_[layer_name_] = generatePointsGridmap(cloud_non_terrain);
}
