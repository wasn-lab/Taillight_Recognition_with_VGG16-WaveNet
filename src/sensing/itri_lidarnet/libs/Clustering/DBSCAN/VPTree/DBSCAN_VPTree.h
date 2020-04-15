#ifndef DBSCAN_VPTREE_H_
#define DBSCAN_VPTREE_H_

#include <fstream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

using namespace pcl;

#include "../dataset.hpp"
#include "dbscan_vp.h"

class DBSCAN_VPTree
{
public:
  DBSCAN_VPTree();
  virtual ~DBSCAN_VPTree();

  void setInputCloud(const PointCloud<PointXYZ>::ConstPtr input);
  void setEpsilon(const double Epsilon);
  void setMinpts(const unsigned int MinPts);
  void segment(IndicesClusters& clusters);

private:
  DBSCAN_VP::Ptr dbs;
  PointCloud<PointXYZ>::ConstPtr input;
  double epsilon;
  unsigned int minpts;
};

#endif /* DBSCAN_VPTREE_H_ */
