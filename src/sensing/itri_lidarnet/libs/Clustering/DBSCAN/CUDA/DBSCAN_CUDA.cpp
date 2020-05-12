#include "DBSCAN_CUDA.h"

bool DBSCAN_CUDA::hasInitialCUDA = false;
int DBSCAN_CUDA::maxThreadsNumber = 0;

DBSCAN_CUDA::DBSCAN_CUDA()
{
  epsilon = new float[4];
  minpts = new size_t[4];
  dset = Dataset::create();

  if (!hasInitialCUDA)
  {
    cudaError_t err = ::cudaSuccess;
    err = cudaSetDevice(0);
    if (err != ::cudaSuccess)
    {
      return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

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

DBSCAN_CUDA::~DBSCAN_CUDA()
{
  dbs = NULL;
}

template <typename PointT>
void DBSCAN_CUDA::setInputCloud(const typename PointCloud<PointT>::ConstPtr Input)
{
  dset->load_pcl(Input);
  dbs = boost::make_shared<GDBSCAN>(dset);
}
void DBSCAN_CUDA::setEpsilon(const double Epsilon, const double EpsilonCar, const double EpsilonPed, const double EpsilonBike)
{
  epsilon[0] = Epsilon;
  epsilon[1] = EpsilonCar;
  epsilon[2] = EpsilonPed;
  epsilon[3] = EpsilonBike;
}
void DBSCAN_CUDA::setMinpts(const unsigned int MinPts, const unsigned int MinPtsCar, const unsigned int MinPtsPed, const unsigned int MinPtsBike)
{
  minpts[0] = MinPts;
  minpts[1] = MinPtsCar;
  minpts[2] = MinPtsPed;
  minpts[3] = MinPtsBike;
}

void DBSCAN_CUDA::segment(IndicesClusters& index)
{
  try
  {
    // const double start = omp_get_wtime ();

    dbs->fit(epsilon, minpts, maxThreadsNumber);
    dbs->predict(index);
    //dbs = NULL;

    // std::cout << "[DBSCNA] CUDA 2 " << dset->rows()<< " " << (omp_get_wtime () - start) <<std::endl;
  }
  catch (const std::runtime_error& re)
  {
    std::cout <<  re.what() << std::endl;
    std::cout << "[DBSCAN] no memory" << std::endl;
  }
}

template void DBSCAN_CUDA::setInputCloud<PointXYZ>(const PointCloud<PointXYZ>::ConstPtr Input);

template void DBSCAN_CUDA::setInputCloud<PointXYZIL>(const PointCloud<PointXYZIL>::ConstPtr Input);
