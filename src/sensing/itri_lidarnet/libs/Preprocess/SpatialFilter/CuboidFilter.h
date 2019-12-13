#ifndef __CUBOIDFILTER_CUDA__
#define __CUBOIDFILTER_CUDA__

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
//#include <pcl/octree/octree.h>

using namespace std;
using namespace pcl;

class CuboidFilter
{
  public:
    CuboidFilter ();
    ~CuboidFilter ();

    template <typename PointT>
    PointCloud<PointT>
    pass_through_soild (typename PointCloud<PointT>::Ptr input,
                        float x_min,
                        float x_max,
                        float y_min,
                        float y_max,
                        float z_min,
                        float z_max);

    template <typename PointT>
    PointCloud<PointT>
    hollow_removal (typename PointCloud<PointT>::Ptr input,
                    double Xmin,
                    double Xmax,
                    double Ymin,
                    double Ymax,
                    double Zmin,
                    double Zmax);

    template <typename PointT>
    vector<PointCloud<PointT>>
    separate_cloud (typename PointCloud<PointT>::Ptr input,
                    float Xmin,
                    float Xmax,
                    float Ymin,
                    float Ymax,
                    float Zmin,
                    float Zmax);

/*
 *  // conflict with QT UI : boost::Q_FOREACH’ has not been declared
    PointCloud<PointXYZ>
    background_removal (const PointCloud<PointXYZ>::ConstPtr input,
                        const PointCloud<PointXYZ>::ConstPtr base);
*/
};

#endif
