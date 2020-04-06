#ifndef DBSCAN_H_
#define DBSCAN_H_

#include <omp.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

using namespace pcl;

class DBSCAN
{
public:
  DBSCAN();
  virtual ~DBSCAN();

  void setInputCloud(const PointCloud<PointXYZ>::ConstPtr input);
  void setEpsilon(const double Epsilon);
  void setMinpts(const unsigned int MinPts);
  void segment(IndicesClusters& clusters);

private:
  PointCloud<PointXYZ>::ConstPtr input;
  double epsilon;
  unsigned int minpts;
};

#endif /* DBSCAN_H_ */
