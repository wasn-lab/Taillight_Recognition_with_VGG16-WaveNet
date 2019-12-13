#ifndef CLASS_EDGE_DETECTION_H_
#define CLASS_EDGE_DETECTION_H_
#include "headers.h"
#include "edge_detection_tools.h"

class EdgeDetection
{
private:
pcl::PointCloud<PointXYZI>::Ptr release_cloud_;
pcl::PointCloud<PointXYZI>::Ptr ring_edge_pointCloud_;
pcl::PointCloud<PointXYZI>::Ptr ground_pointCloud_;
pcl::PointCloud<PointXYZI>::Ptr non_ground_pointCloud_;

thread mthread_1;

double theta_sample_;
float max_radius_;
std::vector<float> contour_distance_;

uint32_t seq_;

grid_map::GridMap gridmap_;

//grid_map
double grid_length_x_;
double grid_length_y_;
double grid_resolution_;
double grid_position_x_;
double grid_position_y_;
double grid_min_value_;
double grid_max_value_;
double maximum_lidar_height_thres_;
double minimum_lidar_height_thres_;
std::string layer_name_;

public:
//ros
ros::Publisher ground_pointcloud_publisher;
ros::Publisher pub_occupancy_grid;
ros::Publisher pub_occupancy_dense_grid;

ros::Publisher pub_costmap;

EdgeDetection ();
~EdgeDetection();

inline void
setThetaSample (float theta_sample)
{
        theta_sample_ = theta_sample;
};

inline void
setMaxRadious (float max_radius)
{
        max_radius_ = max_radius;
};

inline void
setGridMinValue (float grid_min_value)
{
        grid_min_value_ = grid_min_value;
};

inline void
setGridMaxValue (float grid_max_value)
{
        grid_max_value_ = grid_max_value;
};

inline void
setGridLengthOfX (float grid_length_x)
{
        grid_length_x_ = grid_length_x;
};

inline void
setGridLengthOfY (float grid_length_y)
{
        grid_length_y_ = grid_length_y;
};

inline void
setGridResolution (float grid_resolution)
{
        grid_resolution_ = grid_resolution;
};

inline void
setMaxLidarHeightThres (float maximum_lidar_height_thres)
{
        maximum_lidar_height_thres_ = maximum_lidar_height_thres;
};

inline void
setMinLidarHeightThres (float minimum_lidar_height_thres)
{
        minimum_lidar_height_thres_ = minimum_lidar_height_thres;
};

inline void
setLayerName (std::string layer_name)
{
        layer_name_ = layer_name;
};


void LogTotxt(const std::vector<float> contour_distance);



bool
RegisterCallbacks(const ros::NodeHandle& n);



void
setInputCloud (const PointCloud<PointXYZI>::ConstPtr input);

void
cloudPosPreprocess (const PointCloud<PointXYZI>::ConstPtr &input_cloud,
                    pcl::PointCloud<PointXYZI>::Ptr &output_cloud,
                    float deg_x, float deg_y, float deg_z);


void
startThread ();

void
waitThread ();

//points2grid_map
bool isValidInd(const grid_map::Index& grid_ind);

grid_map::Index fetchGridIndexFromPoint(const pcl::PointXYZI& point);

grid_map::Matrix calculateCostmap(const double maximum_height_thres,
                                  const double minimum_lidar_height_thres, const double grid_min_value,
                                  const double grid_max_value, const grid_map::GridMap& gridmap,
                                  const std::string& gridmap_layer_name,
                                  const std::vector<std::vector<std::vector<double> > > grid_vec);

grid_map::Matrix makeCostmapFromSensorPoints(
        const double maximum_height_thres, const double minimum_lidar_height_thres, const double grid_min_value,
        const double grid_max_value, const grid_map::GridMap& gridmap, const std::string& gridmap_layer_name,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& in_sensor_points);

void initGridmapParam(const grid_map::GridMap& gridmap);

std::vector<std::vector<std::vector<double> > > assignPoints2GridCell(
        const grid_map::GridMap& gridmap, const pcl::PointCloud<pcl::PointXYZI>::Ptr& in_sensor_points);
//grid_map_generator
void
initGridmap();

grid_map::Matrix
generatePointsGridmap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& in_sensor_points);

inline pcl::PointCloud<pcl::PointXYZI>::Ptr
getRingEdgePointCloud ()
{
        return ring_edge_pointCloud_;
};

inline grid_map::GridMap
getGridMap ()
{
        return gridmap_;
};

inline pcl::PointCloud<pcl::PointXYZI>::Ptr
getGroundPointCloud ()
{
        return ground_pointCloud_;
};

inline pcl::PointCloud<pcl::PointXYZI>::Ptr
getNonGroundPointCloud ()
{
        return non_ground_pointCloud_;
};

void
calculate ();


};

#endif /*CLASS_EDGE_DETECTION_H_*/
