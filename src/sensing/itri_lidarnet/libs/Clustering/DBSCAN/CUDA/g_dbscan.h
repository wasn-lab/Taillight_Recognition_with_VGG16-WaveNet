#ifndef G_DBSCAN
#define G_DBSCAN

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/min_cut_segmentation.h>

using namespace pcl;

#include "../dataset.hpp"

void
vertdegree (int N,
            int colsize,
            float eps,
            float* d_data,
            int* d_Va,
            int maxThreadsNumber);

void
adjlistsind (int N,
             int* Va0,
             int* Va1);

void
asmadjlist (int N,
            int colsize,
            float eps,
            float* d_data,
            int* d_Va1,
            int* d_Ea);

void
breadth_first_search_kern (int N,
                           int* d_Ea,
                           int* d_Va0,
                           int* d_Va1,
                           int* d_Fa,
                           int* d_Xa);

class GDBSCAN : private boost::noncopyable
{
  public:
    typedef std::vector<int32_t> Labels;
    typedef boost::shared_ptr<GDBSCAN> Ptr;

  public:
    GDBSCAN (const Dataset::Ptr dset);
    ~GDBSCAN ();

    void
    fit (float eps,
         size_t min_elems, int maxThreadsNumber);
    void
    predict (IndicesClusters &index);

  private:

    const Dataset::Ptr m_dset;
    float* d_data;
    const size_t vA_size;
    int* d_Va0;
    int* d_Va1;
    std::vector<int> h_Va0;
    std::vector<int> h_Va1;
    int* d_Ea;
    int* d_Fa;
    int* d_Xa;
    std::vector<bool> core;

    Labels labels;
    int32_t cluster_id;

    void
    breadth_first_search (int i,
                          int32_t cluster,
                          std::vector<bool>& visited);

    void
    ErrorHandle (cudaError_t r,
                 std::string Msg);

};

#endif
