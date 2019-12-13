//http://codextechnicanum.blogspot.com/2015/04/find-minimum-oriented-bounding-box-of.html
//http://www.pcl-users.org/Finding-oriented-bounding-box-of-a-cloud-td4024616.html

#ifndef PCA_H_
#define PCA_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h> //getMinMax3D
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <Eigen/Eigenvalues>

class PCA
{
  public:
    PCA ();
    virtual
    ~PCA ();

    void
    setInputCloud (const pcl::PointCloud<pcl::PointXYZ> input);
    void
    compute (std::vector<pcl::PointXYZ> &out_cube,
             pcl::PointXYZ &out_centroid,
             Eigen::Matrix3f &covariance,
             pcl::PointXYZ &out_minPoint,
             pcl::PointXYZ &out_maxPoint);
    void
    compute2 (std::vector<pcl::PointXYZ> &out_cube,
              pcl::PointXYZ &out_centroid);

  private:
    pcl::PointCloud<pcl::PointXYZ> cloud_3d;
    pcl::PointCloud<pcl::PointXYZ> cloud_2d;
};

#endif /* PCA_H_ */
