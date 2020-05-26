#include "DBSCAN_CUDA.h"

using namespace pcl;

bool DBSCAN_CUDA::hasInitialCUDA_ = false;
int DBSCAN_CUDA::maxThreadsNumber_ = 0;

DBSCAN_CUDA::DBSCAN_CUDA()
{
  epsilon = new float[4];
  minpts = new size_t[4];
  dset = Dataset::create();

  if (!hasInitialCUDA_)
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
      maxThreadsNumber_ = prop.maxThreadsPerBlock / 2;
    }
    else if (prop.major > 2)
    {
      maxThreadsNumber_ = prop.maxThreadsPerBlock;
    }
    else
    {
      maxThreadsNumber_ = 0;
    }

    hasInitialCUDA_ = true;
  }
}

DBSCAN_CUDA::~DBSCAN_CUDA()
{
  dbs = nullptr;
  delete[] epsilon;
  delete[] minpts;
}

template <typename PointT>
void DBSCAN_CUDA::setInputCloud(const typename PointCloud<PointT>::ConstPtr input)
{
  dset->load_pcl(Input);
  dbs = boost::make_shared<GDBSCAN>(dset);
}
void DBSCAN_CUDA::setEpsilon(const double Epsilon, const double EpsilonCar, const double EpsilonPed,
                             const double EpsilonBike, const double EpsilonRule)
{
  epsilon[0] = Epsilon;
  epsilon[1] = EpsilonCar;
  epsilon[2] = EpsilonPed;
  epsilon[3] = EpsilonBike;
  epsilon[4] = EpsilonRule;
}
void DBSCAN_CUDA::setMinpts(const unsigned int MinPts, const unsigned int MinPtsCar, const unsigned int MinPtsPed,
                            const unsigned int MinPtsBike, const unsigned int MinPtsRule)
{
  minpts[0] = MinPts;
  minpts[1] = MinPtsCar;
  minpts[2] = MinPtsPed;
  minpts[3] = MinPtsBike;
  minpts[4] = MinPtsRule;
}

void DBSCAN_CUDA::segment(pcl::IndicesClusters& clusters)
{
  try
  {
    // const double start = omp_get_wtime ();

    dbs->fit(epsilon, minpts, maxThreadsNumber_);
    dbs->predict(clusters);
    // dbs = NULL;

    // std::cout << "[DBSCNA] CUDA 2 " << dset->rows()<< " " << (omp_get_wtime () - start) <<std::endl;
  }
  catch (const std::runtime_error& re)
  {
    std::cout << re.what() << std::endl;
  }
}

template void DBSCAN_CUDA::setInputCloud<PointXYZ>(const PointCloud<PointXYZ>::ConstPtr Input);

template void DBSCAN_CUDA::setInputCloud<PointXYZIL>(const PointCloud<PointXYZIL>::ConstPtr Input);
