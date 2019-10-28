/*
   CREATER: ICL U300
   DATE: June, 2019
 */
#include "glog/logging.h"
#include "parknet_pcd_manager.h"
#include "parknet_logging.h"
#include <mutex>
#include <pcl/io/pcd_io.h>

namespace parknet
{
int read_pcd_file(const std::string filename, PointCloud& out_pcd)
{
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, out_pcd) == -1)
  {
    LOG(ERROR) << "Couldn't read file " << filename;
    return -1;
  }
  return 0;
}
};  // namespace

// ------
ParknetPCDManager::ParknetPCDManager()
{
  for (int i = 0; i < parknet::camera::num_cams_e; i++)
  {
    const int cam_id = i;
    pcd_ptr_[cam_id].reset(new PointCloud);
  }
}

ParknetPCDManager::~ParknetPCDManager()
{
}

PointCloudPtr ParknetPCDManager::get_pcd_ptr(const int cam_id)
{
  std::lock_guard<std::mutex> lock(pcd_mutex_[cam_id]);
  return pcd_ptr_[cam_id];
}

PointCloud& ParknetPCDManager::get_pcd(const int cam_id)
{
  std::lock_guard<std::mutex> lock(pcd_mutex_[cam_id]);
  return *pcd_ptr_[cam_id];
}

int ParknetPCDManager::set_pcd(PointCloud& in_pcd, const int cam_id)
{
  std::lock_guard<std::mutex> lock(pcd_mutex_[cam_id]);
  *pcd_ptr_[cam_id] = in_pcd;
  return 0;
}
