#include "NoiseFilter.h"

NoiseFilter::NoiseFilter ()
{

}

NoiseFilter::~NoiseFilter ()
{

}

template <typename PointT>
PointCloud<PointT>
NoiseFilter::runUniformSampling (const typename PointCloud<PointT>::Ptr input,
                  float model_ss)
{
  PointCloud<PointT> out_cloud;

  pcl::UniformSampling<PointT> uniform_sampling;
  uniform_sampling.setInputCloud (input);
  uniform_sampling.setRadiusSearch (model_ss);
  uniform_sampling.filter (out_cloud);

  return out_cloud;
}

template
PointCloud<PointXYZ>
NoiseFilter::runUniformSampling<PointXYZ> (const typename PointCloud<PointXYZ>::Ptr input,
                  float model_ss);

template
PointCloud<PointXYZI>
NoiseFilter::runUniformSampling<PointXYZI> (const typename PointCloud<PointXYZI>::Ptr input,
                  float model_ss);

PointCloud<PointXYZ>
NoiseFilter::runRandomSampling (PointCloud<PointXYZ>::Ptr input,
                 float model_ss_)
{
  PointCloud<PointXYZ> out_cloud;

  pcl::RandomSample<pcl::PointXYZ> random;

  random.setInputCloud (input);
  random.setKeepOrganized (false);  //设置为true，则输出点云大小和输入点云大小一致
  random.setUserFilterValue (std::numeric_limits<float>::quiet_NaN ());  //当setKeepOrganized设置为true时，设置删除点的值
  random.setNegative (false);  //当设置为false时，输出点云大小为setSamlple的值，否则为点云大小减去setSamlple的值
  random.setSample (input->size ());  //设置采样后，输出的点云大小
  random.filter (out_cloud);

  /*  PointMatcher<float>::DataPoints::Labels buff_label;

   MatrixXf buff_points(4,input->size());

   for (size_t i=0;i<input->size();++i)
   {
   buff_points(0,i) = input->points[i].x;
   buff_points(1,i) = input->points[i].y;
   buff_points(2,i) = input->points[i].z;
   buff_points(3,i) = 1;
   }

   PointMatcher<float>::DataPoints inputCloud(buff_points,buff_label);

   PointMatcher<float>::DataPointsFilter* randomSample(PointMatcher<float>::get().DataPointsFilterRegistrar.create( "RandomSamplingDataPointsFilter",map_list_of("prob", toParam(model_ss_))));

   PointMatcher<float>::DataPoints outputCloud;
   outputCloud = randomSample->filter(inputCloud);


   out_cloud.width = outputCloud.getNbPoints();
   out_cloud.height = 1;
   out_cloud.is_dense = false;
   out_cloud.points.resize (outputCloud.getNbPoints());

   #pragma omp parallel for
   for (size_t i=0;i<outputCloud.getNbPoints();++i)
   {
   out_cloud.points[i].x = outputCloud.features(0,i)/outputCloud.features(3,i);
   out_cloud.points[i].y = outputCloud.features(1,i)/outputCloud.features(3,i);
   out_cloud.points[i].z = outputCloud.features(2,i)/outputCloud.features(3,i);
   }*/

  return out_cloud;
}

PointCloud<PointXYZ>
NoiseFilter::runStatisticalOutlierRemoval (PointCloud<PointXYZ>::Ptr input,
                             int MeanK,
                             double StddevMulThresh)
{
  PointCloud<PointXYZ> out_cloud;

  pcl::StatisticalOutlierRemoval<PointXYZ> sor;
  sor.setInputCloud (input);
  sor.setMeanK (MeanK);
  sor.setStddevMulThresh (StddevMulThresh);
  sor.setNegative (false);
  sor.filter (out_cloud);

  return out_cloud;
}

PointCloud<PointXYZ>
NoiseFilter::runRadiusOutlierRemoval (PointCloud<PointXYZ>::Ptr input,
                        double radius,
                        int min_pts)
{
  PointCloud<PointXYZ> out_cloud;

  if (input->size () > 0)
  {
    pcl::RadiusOutlierRemoval<PointXYZ> outrem;
    outrem.setInputCloud (input);
    outrem.setRadiusSearch (radius);  //unit:m
    outrem.setMinNeighborsInRadius (min_pts);
    outrem.filter (out_cloud);  // apply filter
  }
  else
  {
    out_cloud = *input;
  }
  return out_cloud;

}
