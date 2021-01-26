#ifndef ROI_FUSION_H_
#define ROI_FUSION_H_

#include <vector>
#include <sensor_msgs/RegionOfInterest.h>
#include <msgs/DetectedObjectArray.h>
#include "drivenet/image_preprocessing.h"
#include "roi_cluster_fusion/roi_cluster_fusion_node.h"

class RoiFusion
{
public:
  RoiFusion() = default;
  ~RoiFusion() = default;
  std::vector<sensor_msgs::RegionOfInterest> getLidar2DROI(const std::vector<DriveNet::MinMax2D>& cam_pixels_obj);
  std::vector<sensor_msgs::RegionOfInterest> getCam2DROI(const msgs::DetectedObjectArray& objects_array);
  std::vector<std::pair<int,int>> getRoiFusionResult(const std::vector<sensor_msgs::RegionOfInterest>& object_camera_roi, const std::vector<sensor_msgs::RegionOfInterest>& object_lidar_roi);
  void getFusionCamObj(const msgs::DetectedObjectArray& objects_array, const std::vector<std::pair<int,int>> fusion_index, std::vector<DriveNet::MinMax2D>& cam_pixels_obj);
private:
  roi_cluster_fusion::RoiClusterFusionNodelet roi_fusion_nodelet;
};

#endif
