#include "PlaneGroundFilter.h"

PlaneGroundFilter::PlaneGroundFilter()
{
}

PlaneGroundFilter::~PlaneGroundFilter()
{
}

pcl::ModelCoefficients PlaneGroundFilter::getCoefficientsSAC(const PointCloud<PointXYZ>::ConstPtr input_cloud,
                                                             float high)
{
  pcl::ModelCoefficients coefficients;

  if (input_cloud->size() < 20)
  {
    coefficients.values.resize(4);
    coefficients.values[3] = high;
    return coefficients;
  }
  else
  {
    pcl::PointIndices output;
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(false);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.1);
    seg.setProbability(0.9);
    seg.setInputCloud(input_cloud);
    seg.segment(output, coefficients);
  }

  return coefficients;
}

pcl::ModelCoefficients PlaneGroundFilter::getCoefficientsRANSAC(const PointCloud<PointXYZ>::ConstPtr input_cloud,
                                                                float high)
{
  pcl::ModelCoefficients coefficients;

  vector<int> inliers;
  Eigen::VectorXf vxf;
  PointCloud<PointXYZ>::Ptr outliers(new PointCloud<PointXYZ>);

  pcl::SampleConsensusModelPlane<PointXYZ>::Ptr model_1(new pcl::SampleConsensusModelPlane<PointXYZ>(input_cloud));
  pcl::RandomSampleConsensus<PointXYZ> ransac(model_1);
  ransac.setDistanceThreshold(0.5);
  ransac.setProbability(0.9);
  ransac.computeModel();
  ransac.getInliers(inliers);
  ransac.getModelCoefficients(vxf);

  pcl::copyPointCloud<PointXYZ>(*input_cloud, inliers, *outliers);

  coefficients.values.resize(vxf.size());  // ax+by+cz+d=0
  for (int i = 0; i < vxf.size(); i++)
  {
    coefficients.values[i] = vxf[i];
    cout << "," << vxf[i];
  }
  cout << endl;

  return coefficients;
}

PointCloud<PointXYZ> PlaneGroundFilter::runCoefficients(PointCloud<PointXYZ>::Ptr input, float a, float b, float c,
                                                        float d)
{
  PointCloud<PointXYZ> out_cloud;

  for (size_t i = 0; i < input->size(); i++)
  {
    if (pointToPlaneDistance(input->points[i], a, b, c, d) > 0.2)
    {
      out_cloud.points.push_back(input->points[i]);
    }
  }
  return out_cloud;
}

pcl::PointIndices PlaneGroundFilter::runSAC(const PointCloud<PointXYZ>::ConstPtr input_cloud)
{
  pcl::PointIndices output;

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.1);
  seg.setProbability(0.9);
  seg.setInputCloud(input_cloud);
  seg.segment(output, *coefficients);
  return output;
}

pcl::PointIndices PlaneGroundFilter::runSampleConsensusModel(const PointCloud<PointXYZ>::ConstPtr input_cloud)
{
  pcl::PointIndices output;
  pcl::SampleConsensusModelPlane<PointXYZ>::Ptr model_1(new pcl::SampleConsensusModelPlane<PointXYZ>(input_cloud));
  pcl::RandomSampleConsensus<PointXYZ> ransac(model_1);
  ransac.setMaxIterations(1000);
  ransac.setDistanceThreshold(0.1);
  ransac.setProbability(0.99);
  ransac.computeModel();
  ransac.getInliers(output.indices);

  Eigen::VectorXf vxf;
  ransac.getModelCoefficients(vxf);
  // TODO
  return output;
}

template <typename PointT>
pcl::PointIndices PlaneGroundFilter::runMorphological(const typename PointCloud<PointT>::ConstPtr input,
                                                      float setCellSize, float setBase, int setMaxWindowSize,
                                                      float setSlope, float setInitialDistance, float setMaxDistance)
{
  pcl::PointIndices output;
  if (input->size() == 0)
    return output;

  pcl::ApproximateProgressiveMorphologicalFilter<PointT> apmf;
  apmf.setInputCloud(input);
  apmf.setCellSize(setCellSize);  // 0.3
  apmf.setBase(setBase);          // 2
  apmf.setExponential(false);
  apmf.setMaxWindowSize(setMaxWindowSize);      // 1
  apmf.setSlope(setSlope);                      // 0.9
  apmf.setInitialDistance(setInitialDistance);  // 0.32
  apmf.setMaxDistance(setMaxDistance);          // 0.33
  apmf.setNumberOfThreads(2);
  apmf.extract(output.indices);
  return output;
}

template pcl::PointIndices PlaneGroundFilter::runMorphological<PointXYZ>(
    const typename PointCloud<PointXYZ>::ConstPtr input, float setCellSize, float setBase, int setMaxWindowSize,
    float setSlope, float setInitialDistance, float setMaxDistance);

template pcl::PointIndices PlaneGroundFilter::runMorphological<PointXYZI>(
    const typename PointCloud<PointXYZI>::ConstPtr input, float setCellSize, float setBase, int setMaxWindowSize,
    float setSlope, float setInitialDistance, float setMaxDistance);
