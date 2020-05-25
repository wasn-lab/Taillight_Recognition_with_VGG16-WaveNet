#ifndef DBSCAN_CUDA_H_
#define DBSCAN_CUDA_H_

#include "g_dbscan.h"
#include "../../../UserDefine.h"

class DBSCAN_CUDA
{
public:
  DBSCAN_CUDA();
  virtual ~DBSCAN_CUDA();

  template <typename PointT>
  void setInputCloud(const typename pcl::PointCloud<PointT>::ConstPtr input);
  void setEpsilon(const double Epsilon, 
                                   const double EpsilonCar, 
                                   const double EpsilonPed, 
                                   const double EpsilonBike,
                                   const double EpsilonRule);
  void setMinpts(const unsigned int MinPts, 
                                  const unsigned int MinPtsCar, 
                                  const unsigned int MinPtsPed, 
                                  const unsigned int MinPtsBike,
                                  const unsigned int MinPtsRule);
  void segment(pcl::IndicesClusters& clusters);

private:
  static bool hasInitialCUDA;
  static int maxThreadsNumber;

  float* epsilon;
  size_t* minpts;
  Dataset::Ptr dset;
  GDBSCAN::Ptr dbs;
};

#endif /* DBSCAN_CUDA_H_ */
