#include "DBSCAN_VPTree.h"

DBSCAN_VPTree::DBSCAN_VPTree()
{
  epsilon = 1;
  minpts = 1;
}

DBSCAN_VPTree::~DBSCAN_VPTree()
{
}

void DBSCAN_VPTree::setInputCloud(const PointCloud<PointXYZ>::ConstPtr Input)
{
  input = Input;

  Dataset::Ptr dset = Dataset::create();
  dset->load_pcl(input);
  dbs = boost::make_shared<DBSCAN_VP>(dset);
}
void DBSCAN_VPTree::setEpsilon(const double Epsilon)
{
  epsilon = Epsilon;
}
void DBSCAN_VPTree::setMinpts(const unsigned int MinPts)
{
  minpts = MinPts;
}

void DBSCAN_VPTree::segment(IndicesClusters& index)
{
  dbs->fit();
  dbs->predict(epsilon, minpts);

  if (dbs->get_totoal_cluster_number() > 0)
  {
    PointIndices buff[dbs->get_totoal_cluster_number()];

#pragma omp parallel for
    for (size_t i = 0; i < dbs->get_labels().size(); i++)  // scan all points
    {
      if (dbs->get_labels().at(i) >= 0)
#pragma omp critical
      {
        buff[dbs->get_labels().at(i)].indices.push_back(i);
      }
    }

#pragma omp parallel for
    for (int k = 0; k < dbs->get_totoal_cluster_number(); k++)
    {
#pragma omp critical
      {
        index.push_back(buff[k]);
      }
    }
  }
}
