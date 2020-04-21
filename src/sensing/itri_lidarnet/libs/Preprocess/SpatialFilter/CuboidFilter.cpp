#include "CuboidFilter.h"

CuboidFilter::CuboidFilter()
{
}

CuboidFilter::~CuboidFilter()
{
}

template <typename PointT>
PointCloud<PointT> CuboidFilter::pass_through_soild(typename PointCloud<PointT>::Ptr input, float Xmin, float Xmax,
                                                    float Ymin, float Ymax, float Zmin, float Zmax)
{
  PointCloud<PointT> buff = *input;

  int k = 0;

  for (size_t i = 0; i < buff.size(); i++)
  {
    if ((buff.points[i].x > Xmin && buff.points[i].x < Xmax && buff.points[i].y > Ymin && buff.points[i].y < Ymax &&
         buff.points[i].z > Zmin && buff.points[i].z < Zmax))
    {
      buff.points[k] = buff.points[i];
      k++;
    }
  }
  buff.resize(k);

  return buff;
}

template PointCloud<PointXYZ> CuboidFilter::pass_through_soild<PointXYZ>(typename PointCloud<PointXYZ>::Ptr input,
                                                                         float Xmin, float Xmax, float Ymin, float Ymax,
                                                                         float Zmin, float Zmax);

template PointCloud<PointXYZI> CuboidFilter::pass_through_soild<PointXYZI>(typename PointCloud<PointXYZI>::Ptr input,
                                                                           float Xmin, float Xmax, float Ymin,
                                                                           float Ymax, float Zmin, float Zmax);

template <typename PointT>
PointCloud<PointT> CuboidFilter::hollow_removal(typename PointCloud<PointT>::Ptr input, double Xmin, double Xmax,
                                                double Ymin, double Ymax, double Zmin, double Zmax)
{
  PointCloud<PointT> buff = *input;

  int k = 0;
  for (size_t i = 0; i < buff.size(); i++)
  {
    if (!(buff.points[i].x > Xmin && buff.points[i].x < Xmax && buff.points[i].y > Ymin && buff.points[i].y < Ymax &&
          buff.points[i].z > Zmin && buff.points[i].z < Zmax))
    {
      buff.points[k] = buff.points[i];
      k++;
    }
  }
  buff.resize(k);

  return buff;
}

template PointCloud<PointXYZ> CuboidFilter::hollow_removal<PointXYZ>(typename PointCloud<PointXYZ>::Ptr input,
                                                                     double Xmin, double Xmax, double Ymin, double Ymax,
                                                                     double Zmin, double Zmax);

template PointCloud<PointXYZI> CuboidFilter::hollow_removal<PointXYZI>(typename PointCloud<PointXYZI>::Ptr input,
                                                                       double Xmin, double Xmax, double Ymin,
                                                                       double Ymax, double Zmin, double Zmax);

template <typename PointT>
vector<PointCloud<PointT>> CuboidFilter::separate_cloud(typename PointCloud<PointT>::Ptr input, float Xmin, float Xmax,
                                                        float Ymin, float Ymax, float Zmin, float Zmax)
{
  vector<PointCloud<PointT>> output;
  output.resize(2);

  PointCloud<PointT> cloud1;
  PointCloud<PointT> cloud2;
  cloud1.resize(input->size());
  cloud2.resize(input->size());

  int cnt1 = 0, cnt2 = 0;
  for (size_t i = 0; i < input->size(); ++i)
  {
    if (input->points[i].x > Xmin && input->points[i].x < Xmax && input->points[i].y > Ymin &&
        input->points[i].y < Ymax && input->points[i].z > Zmin && input->points[i].z < Zmax)
    {
      cloud1.points[cnt1] = input->points[i];
      cnt1++;
    }
    else
    {
      cloud2.points[cnt2] = input->points[i];
      cnt2++;
    }
  }
  cloud1.resize(cnt1);
  cloud2.resize(cnt2);

  output.at(0) = cloud1;  // inside
  output.at(1) = cloud2;  // outside
  return output;
}

template vector<PointCloud<PointXYZ>> CuboidFilter::separate_cloud(typename PointCloud<PointXYZ>::Ptr input, float Xmin,
                                                                   float Xmax, float Ymin, float Ymax, float Zmin,
                                                                   float Zmax);

template vector<PointCloud<PointXYZI>> CuboidFilter::separate_cloud(typename PointCloud<PointXYZI>::Ptr input,
                                                                    float Xmin, float Xmax, float Ymin, float Ymax,
                                                                    float Zmin, float Zmax);

/*PointCloud<PointXYZ>
CuboidFilter::background_removal (const PointCloud<PointXYZ>::ConstPtr input,
                    const PointCloud<PointXYZ>::ConstPtr base)
{
  // Instantiate octree-based point cloud change detection class
  pcl::octree::OctreePointCloudChangeDetector<PointXYZ> opccd (0.5f);  // Octree resolution - side length of octree
voxels

  // Add points from base cloud to octree
  opccd.setInputCloud (base);
  opccd.addPointsFromInputCloud ();

  // Switch octree buffers: This resets octree but keeps previous tree structure in memory.
  opccd.switchBuffers ();

  // Add points from input to octree
  opccd.setInputCloud (input);
  opccd.addPointsFromInputCloud ();

  vector<int> newPointIdxVector;

  // Get vector of point indices from octree voxels which did not exist in previous buffer
  opccd.getPointIndicesFromNewVoxels (newPointIdxVector);

  // substracted cloud
  PointCloud<PointXYZ> out_cloud;
  out_cloud.width = newPointIdxVector.size ();
  out_cloud.height = 1;
  out_cloud.is_dense = false;
  out_cloud.points.resize (out_cloud.width * out_cloud.height);

#pragma omp parallel for
  for (size_t i = 0; i < newPointIdxVector.size (); ++i)
  {
    out_cloud.points[i].x = input->points[newPointIdxVector[i]].x;
    out_cloud.points[i].y = input->points[newPointIdxVector[i]].y;
    out_cloud.points[i].z = input->points[newPointIdxVector[i]].z;
  }

  return out_cloud;
}*/
