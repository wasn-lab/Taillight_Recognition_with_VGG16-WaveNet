#ifndef EXTRACTINDICES_H_
#define EXTRACTINDICES_H_

#include "../UserDefine.h"

#include <pcl/filters/project_inliers.h>
#include <pcl/filters/extract_indices.h>

using namespace pcl;

template <typename PointT>
void
extract_Indices (const typename PointCloud<PointT>::ConstPtr input_cloud,
                 const pcl::PointIndicesConstPtr input_Indices,
                 PointCloud<PointT> &output_inlier,
                 PointCloud<PointT> &output_outlier);




PointCloud<PointXYZ>
project_inlier (PointCloud<PointXYZ>::Ptr input,
                float a,
                float b,
                float c,
                float d);

#endif
