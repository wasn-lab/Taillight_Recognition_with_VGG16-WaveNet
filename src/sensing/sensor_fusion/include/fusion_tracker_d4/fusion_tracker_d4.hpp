#ifndef FUSION_TRACKER_D4
#define FUSION_TRACKER_D4
/*
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <msgs/LidRoi.h>
#include <msgs/PointCloud.h>
#include <msgs/PointXYZI.h>
#include <Eigen/Dense>
#include <stdio.h>
*/
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>

// Fov segmentation
#include <pcl/filters/frustum_culling.h>
// plane segmentation
#include <pcl/segmentation/approximate_progressive_morphological_filter.h>
// sac plane segmentation
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
// object clustering
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
// bbox orientation
#include <pcl/features/moment_of_inertia_estimation.h>



using namespace std;
using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

static const int NUM_CAM_OBJ_CALLBACK   = 1;
static const int NUM_LIDAR_OBJ_CALLBACK = 1;
static const int NUM_CAM_GRABBER        = 3;
static const int NUM_LIDAR_CALLBACK     = 5;
static const int TOTAL_CB = NUM_CAM_OBJ_CALLBACK +
                            NUM_LIDAR_CALLBACK +
                            NUM_CAM_GRABBER +
                            NUM_LIDAR_OBJ_CALLBACK;
pthread_t thrd_viewer;                          
void* radClustering(void* arg);
void Segmentation(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_plane, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_objects);

uint32_t dbgPCView;
pthread_mutex_t mut_dbgPCView;
pthread_cond_t cnd_dbgPCView;
pthread_t thrd_dbgPCView;

void* dbg_drawPointCloud(void* arg);

uint32_t syncCount;
pthread_mutex_t callback_mutex;
pthread_cond_t callback_cond;
void sync_callbackThreads();

pthread_mutex_t mtx_laneCloud;
pthread_cond_t cnd_laneCloud;

// typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
void lidFrontLeftCb(const msgs::PointCloud::ConstPtr& pc);
void lidFrontRightCb(const msgs::PointCloud::ConstPtr& pc);
void lidFrontTopCb(const msgs::PointCloud::ConstPtr& pc);
void lidRearLeftCb(const msgs::PointCloud::ConstPtr& pc);
void lidRearRightCb(const msgs::PointCloud::ConstPtr& pc);

std::fstream fLidFrontLeft;
std::fstream fLidFrontRight;
std::fstream fLidFrontTop;
std::fstream fLidRearLeft;
std::fstream fLidRearRight;

void publish_msg();
#endif
