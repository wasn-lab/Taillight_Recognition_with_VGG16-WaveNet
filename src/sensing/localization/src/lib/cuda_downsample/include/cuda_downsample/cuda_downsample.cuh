#ifndef __CUDA_DOWNSAMPLE_CUH__
#define __CUDA_DOWNSAMPLE_CUH__

#include "cuda_downsample.h"

struct compareHashElements
{
        __host__ __device__
        bool operator()(hashElement l, hashElement r)
        {
                return l.index_of_bucket < r.index_of_bucket;
        }
};

struct compareX
{
        __host__ __device__
        bool operator()(pcl::PointXYZI lp, pcl::PointXYZI rp)
        {
                return lp.x < rp.x;
        }
};

struct compareY
{
        __host__ __device__
        bool operator()(pcl::PointXYZI lp, pcl::PointXYZI rp)
        {
                return lp.y < rp.y;
        }
};

struct compareZ
{
        __host__ __device__
        bool operator()(pcl::PointXYZI lp, pcl::PointXYZI rp)
        {
                return lp.z < rp.z;
        }
};

#endif
