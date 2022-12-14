#ifndef CLOUD_CLUSTER_H_
#define CLOUD_CLUSTER_H_

#include "VoxelGrid_CUDA.h"
#include "DBSCAN_CUDA.h"

class CloudCluster
{
public:
  CloudCluster();
  ~CloudCluster() = default;

  std::vector<pcl::PointCloud<pcl::PointXYZI>> getClusters(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input,
                                                           bool do_downsampling);

private:
  DBSCAN_CUDA dbscan_;
};

#endif /* CLOUD_CLUSTER_H_ */