#include "extract_Indices.h"


template <typename PointT>
void
extract_Indices (const typename PointCloud<PointT>::ConstPtr input_cloud,
                 const pcl::PointIndicesConstPtr input_Indices,
                 PointCloud<PointT> &output_inlier,
                 PointCloud<PointT> &output_outlier)
{
  PointCloud<PointT> out_cloud;

  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud (input_cloud);
  extract.setIndices (input_Indices);
  extract.setNegative (false);
  extract.filter (output_inlier);
  extract.setNegative (true);
  extract.filter (output_outlier);
}

template
void
extract_Indices <PointXYZ>(const typename PointCloud<PointXYZ>::ConstPtr input_cloud,
                 const pcl::PointIndicesConstPtr input_Indices,
                 PointCloud<PointXYZ> &output_inlier,
                 PointCloud<PointXYZ> &output_outlier);

template
void
extract_Indices <PointXYZI>(const typename PointCloud<PointXYZI>::ConstPtr input_cloud,
                 const pcl::PointIndicesConstPtr input_Indices,
                 PointCloud<PointXYZI> &output_inlier,
                 PointCloud<PointXYZI> &output_outlier);

template
void
extract_Indices <PointXYZIL>(const typename PointCloud<PointXYZIL>::ConstPtr input_cloud,
                 const pcl::PointIndicesConstPtr input_Indices,
                 PointCloud<PointXYZIL> &output_inlier,
                 PointCloud<PointXYZIL> &output_outlier);

PointCloud<PointXYZ>
project_inlier (PointCloud<PointXYZ>::Ptr input,
                float a,
                float b,
                float c,
                float d)
{
  PointCloud<PointXYZ> out_cloud;

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  coefficients->values.resize (4);  //ax+by+cz+d=0
  coefficients->values[0] = a;  //a
  coefficients->values[1] = b;  //b
  coefficients->values[2] = c;  //c
  coefficients->values[3] = d;  //d

  pcl::ProjectInliers<PointXYZ> proj;
  proj.setModelType (pcl::SACMODEL_PLANE);  //use a plane model
  proj.setInputCloud (input);
  proj.setModelCoefficients (coefficients);
  proj.filter (out_cloud);

  return out_cloud;
}

