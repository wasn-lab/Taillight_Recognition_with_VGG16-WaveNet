#ifndef LIDARSTITCHINGAUTO_H_
#define LIDARSTITCHINGAUTO_H_

#include "headers.h"

class LiDARStitchingAuto
{
public:
  LiDARStitchingAuto();
  ~LiDARStitchingAuto();

  // void Initialize(const ros::NodeHandle& n);
  bool updateEstimation(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_src_cloud,
                        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_trg_cloud);

  void coFilter(const boost::shared_ptr<pcl::PointCloud<PointT> >& input_cloud,
                boost::shared_ptr<pcl::PointCloud<PointT> >& cloud_filtered, const std::string& co, float min_value,
                float max_value, bool negative);

  inline void getFinalTransform(Eigen::Matrix4f& final_transform)
  {
    final_transform = final_transform_;
  };

  inline void setInitTransform(Eigen::Matrix4f init_guess)
  {
    init_guess_ = init_guess;
  };

  inline void setInitTransform(float x, float y, float z, float rx, float ry, float rz)
  {
    Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
    Eigen::Translation3f init_translation(x, y, z);
    Eigen::AngleAxisf init_rotation_x(rx, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf init_rotation_y(ry, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf init_rotation_z(rz, Eigen::Vector3f::UnitZ());
    Eigen::Matrix4f init_guess = (init_translation * init_rotation_z * init_rotation_y * init_rotation_x) * tf;
    init_guess_ = init_guess;
  };

  inline void getAlignNormalClouds(pcl::PointCloud<pcl::PointXYZI>::Ptr& src, pcl::PointCloud<pcl::PointXYZI>::Ptr& trg,
                                   boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > aligned_cloud)
  {
    *src = *src_;
    *trg = *trg_;
    aligned_cloud = normalcloud_source_aligned_;
  };

  inline void getAlignClouds(pcl::PointCloud<pcl::PointXYZI>::Ptr& src, pcl::PointCloud<pcl::PointXYZI>::Ptr& trg,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr& aligned_cloud)
  {
    *src = *src_;
    *trg = *trg_;
    *aligned_cloud = *cloud_source_aligned_;
  };

  inline void getAlignMatrix(Eigen::Matrix4f& final_transform)
  {
    final_transform = final_transform_;
  };

private:
  void pairOmpNDT(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_src,
                  const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_trg,
                  pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_source_aligned_, Eigen::Matrix4f& final_transform,
                  int iterations);

  int numCPU_ = sysconf(_SC_NPROCESSORS_ONLN);

  pcl::console::TicToc tt;
  bool initialized_;
  // trg, src point clouds
  pcl::PointCloud<pcl::PointXYZI>::Ptr src_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr trg_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr icp_cloud_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_source_aligned_;
  boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > normalcloud_source_aligned_;

  double icp_epsilon_;
  double icp_Distance_;
  double icp_Euclidean_Distance_;
  float x_, y_, z_, rx_, ry_, rz_;

  unsigned int icp_iterations_;
  Eigen::Matrix4f final_transform_;
  Eigen::Matrix4f init_guess_;
  Eigen::Matrix3f R_;
};

#endif
