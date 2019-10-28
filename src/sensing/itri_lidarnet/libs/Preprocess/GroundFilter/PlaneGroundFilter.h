#ifndef PLANE_GROUND_FILTER_H_
#define PLANE_GROUND_FILTER_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/approximate_progressive_morphological_filter.h>

using namespace std;
using namespace pcl;

class PlaneGroundFilter
{
  public:
    PlaneGroundFilter ();
    ~PlaneGroundFilter ();

    ModelCoefficients
    getCoefficientsSAC (const PointCloud<PointXYZ>::ConstPtr input_cloud,
                                float high);

    ModelCoefficients
    getCoefficientsRANSAC (const PointCloud<PointXYZ>::ConstPtr input_cloud,
                                float high);

    PointCloud<PointXYZ>
    runCoefficients (PointCloud<PointXYZ>::Ptr input,
                   float a,
                   float b,
                   float c,
                   float d);

    PointIndices
    runSAC (const PointCloud<PointXYZ>::ConstPtr input_cloud);

    PointIndices
    runSampleConsensusModel (const PointCloud<PointXYZ>::ConstPtr input_cloud);

    template <typename PointT>
    pcl::PointIndices
    runMorphological (const typename PointCloud<PointT>::ConstPtr input,
                                                         float setCellSize,
                                                         float setBase,
                                                         int setMaxWindowSize,
                                                         float setSlope,
                                                         float setInitialDistance,
                                                         float setMaxDistance
                                                         );

};

#endif  // PLANE_GROUND_FILTER_H_
