#ifndef G_DBSCAN
#define G_DBSCAN

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/min_cut_segmentation.h>

#include "../dataset.hpp"

enum ClusteringMode
{
  NORMAL_MODE = 0,
  LABEL_MODE
};

void vertdegree(int N, int colsize, float* eps, float* d_data, int* d_Va, int* d_label, int maxThreadsNumber,
                int label_mode);

void adjlistsind(int N, int* Va0, int* Va1);

void asmadjlist(int N, int colsize, float* eps, float* d_data, int* d_Va1, int* d_Ea, int* d_label, int label_mode);

void breadth_first_search_kern(int N, int* d_Ea, int* d_Va0, int* d_Va1, int* d_Fa, int* d_Xa);

class GDBSCAN : private boost::noncopyable
{
public:
  typedef std::vector<int32_t> Labels;
  typedef boost::shared_ptr<GDBSCAN> Ptr;

public:
  GDBSCAN(const Dataset::Ptr& dset);
  ~GDBSCAN();

  void fit(float* eps, const size_t* min_elems, int maxThreadsNumber);
  void predict(pcl::IndicesClusters& index);

private:
  const Dataset::Ptr m_dset;
  float* d_data;
  int* d_label;
  const size_t vA_size;
  int* d_Va0;
  int* d_Va1;
  std::vector<int> h_Va0;
  std::vector<int> h_Va1;
  int* d_Ea;
  int* d_Fa;
  int* d_Xa;
  float* d_eps;
  std::vector<bool> core;

  Labels labels;
  int32_t cluster_id;

  void breadth_first_search(int i, int32_t cluster, std::vector<bool>& visited);

  void ErrorHandle(cudaError_t r, const std::string& Msg);
};

#endif
