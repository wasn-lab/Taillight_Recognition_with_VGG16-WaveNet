/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */

#ifndef __PARKNET_PCD_MANAGER_H__
#define __PARKNET_PCD_MANAGER_H__

#include "parknet_camera.h"
#include <mutex>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;

namespace parknet
{
int read_pcd_file(const std::string filename, PointCloud& out_pcd);
};  // namespace

class ParknetPCDManager
{
private:
  PointCloudPtr pcd_ptr_[parknet::camera::num_cams_e];
  // vars for threading
  std::mutex pcd_mutex_[parknet::camera::num_cams_e];

public:
  ParknetPCDManager();
  ~ParknetPCDManager();
  PointCloudPtr get_pcd_ptr(const int cam_id);
  PointCloud& get_pcd(const int cam_id);
  int set_pcd(PointCloud&, const int cam_id);
};

#endif  // __PARKNET_PCD_MANAGER_H__
