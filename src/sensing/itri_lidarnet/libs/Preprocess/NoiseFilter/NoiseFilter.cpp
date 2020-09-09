#include "NoiseFilter.h"

NoiseFilter::NoiseFilter()
{
}

NoiseFilter::~NoiseFilter()
{
}

//-------- Uniform Sampling <PointT>
template <typename PointT>
PointCloud<PointT> NoiseFilter::runUniformSampling(const typename PointCloud<PointT>::Ptr input, float model_ss)
{
  PointCloud<PointT> out_cloud;

  pcl::UniformSampling<PointT> uniform_sampling;
  uniform_sampling.setInputCloud(input);
  uniform_sampling.setRadiusSearch(model_ss);
  uniform_sampling.filter(out_cloud);

  return out_cloud;
}

template PointCloud<PointXYZ> NoiseFilter::runUniformSampling<PointXYZ>(const typename PointCloud<PointXYZ>::Ptr input,
                                                                        float model_ss);

template PointCloud<PointXYZI>
NoiseFilter::runUniformSampling<PointXYZI>(const typename PointCloud<PointXYZI>::Ptr input, float model_ss);

//-------- Random Sample <PointXYZ>
PointCloud<PointXYZ> NoiseFilter::runRandomSampling(PointCloud<PointXYZ>::Ptr input, float model_ss_)
{
  PointCloud<PointXYZ> out_cloud;

  pcl::RandomSample<pcl::PointXYZ> random;

  random.setInputCloud(input);
  random.setKeepOrganized(false);  //设置为true，则输出点云大小和输入点云大小一致
  random.setUserFilterValue(std::numeric_limits<float>::quiet_NaN());  //当setKeepOrganized设置为true时，设置删除点的值
  random.setNegative(false);  //当设置为false时，输出点云大小为setSamlple的值，否则为点云大小减去setSamlple的值
  random.setSample(input->size());  //设置采样后，输出的点云大小
  random.filter(out_cloud);

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

   PointMatcher<float>::DataPointsFilter* randomSample(PointMatcher<float>::get().DataPointsFilterRegistrar.create(
   "RandomSamplingDataPointsFilter",map_list_of("prob", toParam(model_ss_))));

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

//-------- Statistical Filter <PointXYZ>
template <typename PointT>
PointCloud<PointT> NoiseFilter::runStatisticalOutlierRemoval(typename PointCloud<PointT>::Ptr input, int MeanK,
                                                             double StddevMulThresh)
{
  PointCloud<PointT> out_cloud;

  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(input);
  sor.setMeanK(MeanK);
  sor.setStddevMulThresh(StddevMulThresh);
  sor.setNegative(false);
  sor.filter(out_cloud);

  return out_cloud;
}
template PointCloud<PointXYZI> NoiseFilter::runStatisticalOutlierRemoval<PointXYZI>(
    typename PointCloud<PointXYZI>::Ptr input, int MeanK, double StddevMulThresh);

//-------- RadiusFilter <PointT>
template <typename PointT>
PointCloud<PointT> NoiseFilter::runRadiusOutlierRemoval(typename PointCloud<PointT>::Ptr input, double radius,
                                                        int min_pts)
{
  PointCloud<PointT> out_cloud;

  if (input->size() > 0)
  {
    pcl::RadiusOutlierRemoval<PointT> outrem;
    outrem.setInputCloud(input);
    outrem.setRadiusSearch(radius);  // unit:m
    outrem.setMinNeighborsInRadius(min_pts);
    outrem.filter(out_cloud);  // apply filter
  }
  else
  {
    out_cloud = *input;
  }
  return out_cloud;
}

template PointCloud<PointXYZ> 
NoiseFilter::runRadiusOutlierRemoval<PointXYZ>(typename PointCloud<PointXYZ>::Ptr input,
                                                                             double radius, int min_pts);
template PointCloud<PointXYZI>
NoiseFilter::runRadiusOutlierRemoval<PointXYZI>(typename PointCloud<PointXYZI>::Ptr input, double radius, int min_pts);


PointCloud<PointXYZIR> NoiseFilter::runRingOutlierRemoval(PointCloud<PointXYZIR>::Ptr input, int ring_num, float threshold)
{ 
  pcl::PointCloud<pcl::PointXYZIR>::Ptr pcl_input(new pcl::PointCloud<pcl::PointXYZIR>);
  pcl::copyPointCloud(*input, *pcl_input);
  if (pcl_input->points.empty()) 
  {
    return *pcl_input;
  }

  std::vector<pcl::PointCloud<pcl::PointXYZIR>> pcl_input_ring_array;
  pcl_input_ring_array.resize(ring_num);  // TODO

  for (const auto& p : pcl_input->points) 
  {
    pcl_input_ring_array.at(p.ring).push_back(p);
  }

  pcl::PointCloud<pcl::PointXYZIR>::Ptr pcl_output(new pcl::PointCloud<pcl::PointXYZIR>);
  pcl_output->points.reserve(pcl_input->points.size());

  pcl::PointXYZIR p;
  pcl::PointXYZIR p_forward;
  pcl::PointXYZIR p_forward_forward;

  for (int i=0; i < pcl_input_ring_array.size(); i++)
  {
    pcl_input_ring_array[i].points.push_back(pcl_input_ring_array[i].points.at(0));
    pcl_input_ring_array[i].points.push_back(pcl_input_ring_array[i].points.at(1));
  }

  for (const auto & ring_pointcloud : pcl_input_ring_array)
  {
    if (ring_pointcloud.points.size() < 2) 
    {
      continue;
    }

    for (auto iter = std::begin(ring_pointcloud.points); iter != std::end(ring_pointcloud.points) - 1; ++iter) 
    {
      p.x = iter->x;
      p.y = iter->y;
      p.z = iter->z;
      p.intensity = iter->intensity;
      p.ring = iter->ring;
      if (p_forward.x && p_forward_forward.x)
      {
        float diff_point_1 = std::sqrt(
                        std::pow(p_forward.x - p.x, 2.0) +
                        std::pow(p_forward.y - p.y, 2.0) +
                        std::pow(p_forward.z - p.z, 2.0) );
        
        float diff_point_2 = std::sqrt(
                        std::pow(p_forward.x - p_forward_forward.x, 2.0) +
                        std::pow(p_forward.y - p_forward_forward.y, 2.0) +
                        std::pow(p_forward.z - p_forward_forward.z, 2.0) );
        
        if( diff_point_1 <= threshold || diff_point_2 <= threshold )
        {
          pcl_output->points.push_back(p_forward);
        }    
      }
      p_forward_forward = p_forward;
      p_forward = p;    
    }
  }
  pcl_output->header = input->header;
  return *pcl_output;
}


