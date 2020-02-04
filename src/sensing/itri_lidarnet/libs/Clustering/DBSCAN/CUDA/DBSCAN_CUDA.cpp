#include "DBSCAN_CUDA.h"

bool DBSCAN_CUDA::hasInitialCUDA = false;
int DBSCAN_CUDA::maxThreadsNumber = 0;

DBSCAN_CUDA::DBSCAN_CUDA ()
{
  epsilon = 1;
  minpts = 1;
  dset = Dataset::create ();

  if (!hasInitialCUDA)
  {
    cudaError_t err = ::cudaSuccess;
    err = cudaSetDevice (0);
    if (err != ::cudaSuccess){
      return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties (&prop, 0);

    if (prop.major == 2)
    {
      maxThreadsNumber = prop.maxThreadsPerBlock / 2;
    }
    else if (prop.major > 2)
    {
      maxThreadsNumber = prop.maxThreadsPerBlock;
    }
    else
    {
      maxThreadsNumber = 0;
    }

    hasInitialCUDA = true;
  }
}

DBSCAN_CUDA::~DBSCAN_CUDA ()
{
}

template <typename PointT>
void
DBSCAN_CUDA::setInputCloud (const typename PointCloud<PointT>::ConstPtr Input)
{
  dset->load_pcl (Input);
  dbs = boost::make_shared<GDBSCAN> (dset);
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
    //const double start = omp_get_wtime ();

    dbs->fit (epsilon, minpts ,maxThreadsNumber);
    dbs->predict (index);

    //std::cout << "[DBSCNA] CUDA 2 " << dset->rows()<< " " << (omp_get_wtime () - start) <<std::endl;
  }
  catch (const std::runtime_error& re)
  {
    std::cout << "[DBSCAN] no memory" << std::endl;
  }
}

template
void
DBSCAN_CUDA::setInputCloud<PointXYZ> (const PointCloud<PointXYZ>::ConstPtr Input);

template
void
DBSCAN_CUDA::setInputCloud<PointXYZIL> (const PointCloud<PointXYZIL>::ConstPtr Input);
