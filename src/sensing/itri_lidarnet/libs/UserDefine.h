#ifndef __USERDEFINE__
#define __USERDEFINE__

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <vector>

#define PCL_NO_PRECOMPILE

#define D2R (M_PI/180.0)
#define R2D (180.0/M_PI)

namespace pcl
{

// (x,y,z,intensity,distance,label)
  struct PointXYZIDL
  {
      PCL_ADD_POINT4D      // point types, 4 elements
      float intensity;     // lidar intensity
      float d;             // distance or range (paper)
      int label;EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // ensure proper alignment (Eigen)
  }EIGEN_ALIGN16;
  // point structure : euclidean xyz coordinates, and the intensity value.

// (x,y,z,intensity,label,object)
  struct PointXYZLO
  {
      PCL_ADD_POINT4D; // This adds the members x,y,z which can also be accessed using the point
      int label;
      int object;EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
  }EIGEN_ALIGN16;
  // enforce SSE padding for correct memory alignment

  struct PointXYZIL
  {
      PCL_ADD_POINT4D; // preferred way of adding a XYZ+padding
      float intensity;
      int label;EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
  }EIGEN_ALIGN16;
  // enforce SSE padding for correct memory alignment

  struct PointXYZID
  {
      PCL_ADD_POINT4D                 // point types, 4 elements
      float intensity;                // lidar intensity
      float d;                        // depth or range (paper)
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // ensure proper alignment (Eigen)
  }EIGEN_ALIGN16;
// point structure : euclidean xyz coordinates, and the intensity value.

}// namespace pcl

// here we assume a XYZ + "test" (as fields)
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIDL, (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, d, d)(int, label, label))

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZLO, (float, x, x)(float, y, y)(float, z, z)(int, label, label)(int, object, object))

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIL, (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(int, label, label))

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZID, (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, d, d))

PCL_INSTANTIATE_PointCloud(pcl::PointXYZIDL);
PCL_INSTANTIATE_PointCloud(pcl::PointXYZLO);
PCL_INSTANTIATE_PointCloud(pcl::PointXYZIL);
PCL_INSTANTIATE_PointCloud(pcl::PointXYZID);

typedef pcl::PointXYZI VPoint;
typedef pcl::PointCloud<VPoint> VPointCloud;
typedef pcl::PointXYZIDL VPointXYZIDL;
typedef pcl::PointCloud<VPointXYZIDL> VPointCloudXYZIDL;
typedef pcl::PointXYZLO VPointXYZLO;
typedef pcl::PointCloud<VPointXYZLO> VPointCloudXYZLO;
typedef pcl::PointXYZIL VPointXYZIL;
typedef pcl::PointCloud<VPointXYZIL> VPointCloudXYZIL;
typedef pcl::PointXYZID VPointXYZID;
typedef pcl::PointCloud<VPointXYZID> VPointCloudXYZID;

struct CLUSTER_INFO
{
    // Cloud
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::PointCloud<pcl::PointXYZIL> cloud_IL;
    pcl::PointXYZ cld_center;

    // Bounding Box
    std::vector<pcl::PointXYZ> abb_vertex;
    pcl::PointXYZ center;
    pcl::PointXYZ min;
    pcl::PointXYZ max;
    float dx;
    float dy;
    float dz;

    std::vector<pcl::PointXYZ> obb_vertex;
    pcl::PointXYZ obb_center;
    float obb_dx;
    float obb_dy;
    float obb_dz;
    float obb_orient;  // when z = 0

    float dis_max_min;       // size of vehicle
    float dis_center_origin;
    float dis_abbc_obbc;
    float dis_abb_cldc_min;
    float dis_obb_cldc_min;
    float angle_from_x_axis;  // when z = 0

    // Convex Hull
    pcl::PointCloud<pcl::PointXYZ> convex_hull;

    // Tracking
    bool found_num;
    int tracking_id;
    pcl::PointXYZ track_last_center;
    pcl::PointXYZ predict_next_center;
    pcl::PointXYZ velocity;  // use point to represent velocity x,y,z

    // Classification
    int cluster_tag;  // tag =0 (not key object) tag =1 (unknown object) tag =2 (pedestrian) tag =3 (motorcycle)  tag =4 (car)  tag =5 (bus)
    int confidence;
    float hull_vol;
    float GRSD21[21];
    Eigen::Matrix3f covariance;

    // Output
    pcl::PointXY to_2d_PointWL[2];  // to_2d_PointWL[0] --> 2d image coordinate x,y     to_2d_PointWL[1] --> x = width y = height
    pcl::PointXY to_2d_points[4];   // to_2d_points[0]~[4] --> 2d image coordinate x,y

};
typedef struct CLUSTER_INFO CLUSTER_INFO;

struct POSE_MAP
{
    double tx;
    double ty;
    double tz;
    double rx;
    double ry;
    double rz;
    double qx;
    double qy;
    double qz;
    double qw;
};
typedef struct POSE_MAP POSE_MAP;

#endif
