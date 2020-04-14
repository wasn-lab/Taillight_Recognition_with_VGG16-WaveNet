#ifndef NOISE_FILTER_H_
#define NOISE_FILTER_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include "../../UserDefine.h"

using namespace std;
using namespace pcl;

class NoiseFilter
{
public:
  NoiseFilter();
  ~NoiseFilter();

  template <typename PointT>
  PointCloud<PointT> runUniformSampling(const typename PointCloud<PointT>::Ptr input, float model_ss);

  PointCloud<PointXYZ> runRandomSampling(PointCloud<PointXYZ>::Ptr input, float model_ss);

  PointCloud<PointXYZ> runStatisticalOutlierRemoval(PointCloud<PointXYZ>::Ptr input, int MeanK, double StddevMulThresh);

  PointCloud<PointXYZ> runRadiusOutlierRemoval(PointCloud<PointXYZ>::Ptr input, double radius, int min_pts);
};

#endif
