#ifndef S1CLUSTER_H_
#define S1CLUSTER_H_

#include "all_header.h"

#include "UseApproxMVBB.h"
#include "NoiseFilter.h"
#include "VoxelGrid_CUDA.h"
#include "DBSCAN_CUDA.h"

#define TAG_RAW    -2
#define TAG_DROP   -1
#define TAG_UNKNOW  0
#define TAG_PERSON  1
#define TAG_BICYCLE 2
#define TAG_CAR     3


class S1Cluster
{
  public:
    S1Cluster ();
    S1Cluster (boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer,
               int *input_viewID);
    virtual
    ~S1Cluster ();

    CLUSTER_INFO*
    getClusters (bool is_debug,
                 const PointCloud<PointXYZIL>::ConstPtr inputIL,
                 int *cluster_number);

  private:
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    int *viewID;

    DBSCAN_CUDA dbscan;

};

#endif /* S1CLUSTER_H_ */
