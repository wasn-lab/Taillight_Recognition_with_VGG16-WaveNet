#ifndef S1CLUSTER_H_
#define S1CLUSTER_H_

#include "../all_header.h"
#include "../GlobalVariable.h"
#include "PCA.h"
#include "DBSCAN_CUDA.h"

class S1Cluster
{
public:
  S1Cluster();
  S1Cluster(boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer, int* input_viewID);
  virtual ~S1Cluster();

  void initial(boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer, int* input_viewID);

  void setPlaneParameter(pcl::ModelCoefficients inputCoef);

  CLUSTER_INFO* getClusters(bool debug, PointCloud<PointXYZ>::ConstPtr input, int* cluster_number);

private:
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  int* viewID;

  pcl::ModelCoefficients plane_coef;

  DBSCAN_CUDA dbscan;
};

#endif /* S1CLUSTER_H_ */
