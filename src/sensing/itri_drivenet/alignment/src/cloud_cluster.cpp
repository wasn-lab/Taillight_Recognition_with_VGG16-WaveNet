#include "cloud_cluster.h"

CloudCluster::CloudCluster()
{
  dbscan.setEpsilon(0.6);
  dbscan.setMinpts(5);
}

CloudCluster::~CloudCluster()
{
}

std::vector<pcl::PointCloud<pcl::PointXYZI>> CloudCluster::getClusters(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input, bool do_downsampling)
{
  std::vector<pcl::PointIndices> vectorCluster;
  pcl::PointCloud<pcl::PointXYZ>::Ptr ptr_cur_cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if(do_downsampling && input->points.size() >= 50)
  {
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_upsample(new pcl::PointCloud<pcl::PointXYZI>);
    *input_upsample = *input;
    *input_upsample = VoxelGrid_CUDA().compute<pcl::PointXYZI>(input_upsample, 0.5);

    pcl::copyPointCloud(*input_upsample, *ptr_cur_cloud);
  }
  else
  {
    pcl::copyPointCloud(*input, *ptr_cur_cloud);
  }

  dbscan.setInputCloud<pcl::PointXYZ>(ptr_cur_cloud);
  dbscan.segment(vectorCluster);

  std::vector<pcl::PointCloud<pcl::PointXYZI>> vector_raw_cloud(vectorCluster.size());

  for (size_t i = 0; i < vectorCluster.size(); i++)
  {
    pcl::PointCloud<pcl::PointXYZI> raw_cloud;

    raw_cloud.resize(vectorCluster.at(i).indices.size());

    for (size_t j = 0; j < vectorCluster.at(i).indices.size(); j++)
    {
      raw_cloud.points[j] = input->points[vectorCluster.at(i).indices.at(j)];
    }
    vector_raw_cloud[i] = raw_cloud;
  }

  return vector_raw_cloud;
}