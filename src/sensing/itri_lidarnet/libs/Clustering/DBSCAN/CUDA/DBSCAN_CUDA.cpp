#include "DBSCAN_CUDA.h"

DBSCAN_CUDA::DBSCAN_CUDA ()
{
  epsilon = 1;
  minpts = 1;
  dset = Dataset::create ();

  cudaError_t err = ::cudaSuccess;
  err = cudaSetDevice (0);
  if (err != ::cudaSuccess){
    return;
  }
}

DBSCAN_CUDA::~DBSCAN_CUDA ()
{
}

void
DBSCAN_CUDA::setInputCloud (const PointCloud<PointXYZ>::ConstPtr Input)
{

  dset->load_pcl (Input);

  dbs = boost::make_shared<GDBSCAN> (dset);// very slow, need -O2

}
void
DBSCAN_CUDA::setEpsilon (const double Epsilon)
{
  epsilon = Epsilon;
}
void
DBSCAN_CUDA::setMinpts (const unsigned int MinPts)
{
  minpts = MinPts;
}

void
DBSCAN_CUDA::segment (IndicesClusters &index)
{
  try
  {
    dbs->fit (epsilon, minpts);
    dbs->predict (index);
  }
  catch (const std::runtime_error& re)
  {
    std::cout << "[DBSCAN] no memory" << std::endl;
  }
}
