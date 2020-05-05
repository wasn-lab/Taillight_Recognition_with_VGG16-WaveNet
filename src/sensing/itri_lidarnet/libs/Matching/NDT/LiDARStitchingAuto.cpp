#include "LiDARStitchingAuto.h"

LiDARStitchingAuto::LiDARStitchingAuto()  // : initialized_(false)
{
  src_.reset(new pcl::PointCloud<pcl::PointXYZI>);
  trg_.reset(new pcl::PointCloud<pcl::PointXYZI>);
  icp_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>);
  cloud_source_aligned_.reset(new pcl::PointCloud<pcl::PointXYZI>);
  x_ = 0;
  y_ = 0;
  z_ = 0;
  rx_ = 0;
  ry_ = 0;
  rz_ = 0;
  icp_epsilon_ = 0;
  icp_Distance_ = 0;
  icp_Euclidean_Distance_ = 0;
  icp_iterations_ = 0;
}

LiDARStitchingAuto::~LiDARStitchingAuto()
{
}

void LiDARStitchingAuto::coFilter(const boost::shared_ptr<pcl::PointCloud<PointT> >& input_cloud,
                                  boost::shared_ptr<pcl::PointCloud<PointT> >& cloud_filtered, const std::string& co,
                                  float min_value, float max_value, bool negative)
{
  pcl::PassThrough<PointT> pass;
  pass.setInputCloud(input_cloud);
  pass.setFilterFieldName(co), pass.setFilterLimits(min_value, max_value);
  pass.setFilterLimitsNegative(negative);
  pass.filter(*cloud_filtered);
}

void LiDARStitchingAuto::pairOmpNDT(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_src,
                                    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_trg,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_source_aligned_,
                                    Eigen::Matrix4f& final_transform_, int iterations)
{
  std::cout << "start pairing ... " << std::endl;

  Eigen::Matrix4f move_T = Eigen::Matrix4f::Identity();
  pcl::PointCloud<PointT>::Ptr cloud_source_aligned(new pcl::PointCloud<PointT>);

  pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>::Ptr ndt_omp(
      new pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  ndt_omp->setResolution(0.01);
  ndt_omp->setInputSource(cloud_src);
  ndt_omp->setInputTarget(cloud_trg);
  ndt_omp->setMaximumIterations(300);
  ndt_omp->setStepSize(0.001);
  ndt_omp->setTransformationEpsilon(0.0001);
  ndt_omp->setNumThreads(8);
  ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);

  std::cout << "start pairing ... " << std::endl;
  ndt_omp->align(*cloud_source_aligned, init_guess_);

  std::cout << "aligned " << std::endl;
  std::cout << "has converged first time:" << ndt_omp->getFinalNumIteration()
            << " score: " << ndt_omp->getFitnessScore() << std::endl;
  std::cout << "has getTransformationEpsilon:" << ndt_omp->getFinalTransformation() << std::endl;

  *cloud_source_aligned_ = *cloud_source_aligned;

  move_T = ndt_omp->getFinalTransformation() * move_T;
  final_transform_ = move_T;
  // final_transform = move_T.inverse();
  std::cout << "final_transform" << std::endl << final_transform_ << std::endl;
}

bool LiDARStitchingAuto::updateEstimation(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_src_cloud,
                                          const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_trg_cloud)
{
  if (input_src_cloud->size() > 2 && input_trg_cloud->size() > 2)
  {
    std::cout << "input_src_cloud = " << input_src_cloud->size() << std::endl;
    std::cout << "input_trg_cloud = " << input_trg_cloud->size() << std::endl;

    pcl::PointCloud<PointT>::Ptr cloud_filtered_src(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered_trg(new pcl::PointCloud<PointT>);

    copyPointCloud(*input_src_cloud, *src_);
    copyPointCloud(*input_trg_cloud, *trg_);
    //        std::cout << "cloud_filtered_src = " << cloud_filtered_src->size()<< std::endl;
    pairOmpNDT(src_, trg_, cloud_source_aligned_, final_transform_, 10);

    return true;
  }
  std::cout << "no valid src or trg = " << input_trg_cloud->size() << std::endl;

  return false;
}
