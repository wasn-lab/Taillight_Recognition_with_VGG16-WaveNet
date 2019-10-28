#ifndef DBSCAN_CUDA_H_
#define DBSCAN_CUDA_H_

#include "g_dbscan.h"

class DBSCAN_CUDA
{
  public:
    DBSCAN_CUDA ();
    virtual
    ~DBSCAN_CUDA ();

    void
    setInputCloud (const PointCloud<PointXYZ>::ConstPtr input);
    void
    setEpsilon (const double Epsilon);
    void
    setMinpts (const unsigned int MinPts);
    void
    segment (IndicesClusters &clusters);

  private:
    double epsilon;
    unsigned int minpts;
    Dataset::Ptr dset;
    GDBSCAN::Ptr dbs;
};

#endif /* DBSCAN_CUDA_H_ */
