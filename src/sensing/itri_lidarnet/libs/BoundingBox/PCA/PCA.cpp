#include "PCA.h"

PCA::PCA ()
{
}

PCA::~PCA ()
{
}

void
PCA::setInputCloud (const pcl::PointCloud<pcl::PointXYZ> input)
{
  cloud_3d = input;
  cloud_2d = input;

#pragma omp parallel for
  for (size_t i = 0; i < input.size (); ++i)
  {
    cloud_2d.points[i].z = 0;
  }
}

void
PCA::compute (std::vector<pcl::PointXYZ> &out_cube,
              pcl::PointXYZ &out_centroid,
              Eigen::Matrix3f &out_covariance,
              pcl::PointXYZ &out_minPoint,
              pcl::PointXYZ &out_maxPoint)
{
  // 1) compute the centroid (c0, c1, c2) and the normalized covariance
  Eigen::Vector4f Centroid3D, Centroid2D;
  pcl::compute3DCentroid (cloud_3d, Centroid3D);
  out_centroid.x = Centroid3D[0];
  out_centroid.y = Centroid3D[1];
  out_centroid.z = Centroid3D[2];
  Centroid2D[0] = Centroid3D[0];
  Centroid2D[1] = Centroid3D[1];
  Centroid2D[2] = 0;
  computeCovarianceMatrixNormalized (cloud_2d, Centroid2D, out_covariance);

  // 2) compute principal directions, get the eigenvectors e0, e1, e2. The reference system will be (e0, e1, e0 X e1) --- note: e0 X e1 = +/- e2
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver (out_covariance, Eigen::ComputeEigenvectors);
  Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors ();
  // proper orientation in some cases. The numbers come out the same without it, but signs are different and box doesn't get correctly oriented in some cases.
  eigenVectorsPCA.col (2) = eigenVectorsPCA.col (0).cross (eigenVectorsPCA.col (1));
  //std::cout << "eigenVectorsPCA:\n" << eigenVectorsPCA << std::endl;
  //std::cout << "eigenValuesPCA:\n" << eigen_solver.eigenvalues() << std::endl;
  /*
   /// Note that getting the eigenvectors can also be obtained via the PCL PCA interface with something like:
   pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPCAprojection (new pcl::PointCloud<pcl::PointXYZ>);
   pcl::PCA<pcl::PointXYZ> pca;
   pca.setInputCloud(inputCloud);
   pca.project(*inputCloud, *cloudPCAprojection);
   std::cerr << std::endl << "EigenVectors:\n" << pca.getEigenVectors() << std::endl;
   std::cerr << std::endl << "EigenValues\n" << pca.getEigenValues() << std::endl;
   //In this case, pca.getEigenVectors returns something similar to eigenVectorsPCA.
   */

  // 3) move the points in that RF --- note: the transformation given by the rotation matrix (e0, e1, e0 X e1) & (c0, c1, c2) must be inverted
  // Transform the original cloud to the origin where the principal components correspond to the axes.
  Eigen::Matrix4f projectionTransform (Eigen::Matrix4f::Identity ());
  projectionTransform.block<3, 3> (0, 0) = eigenVectorsPCA.transpose ();
  projectionTransform.block<3, 1> (0, 3) = -1.f * (projectionTransform.block<3, 3> (0, 0) * Centroid2D.head<3> ());
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointsProjected (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud (cloud_2d, *cloudPointsProjected, projectionTransform);
  //std::cout << "Projection transform:\n" << projectionTransform << std::endl;

  // 4) compute the max, the min and the center of the diagonal
  //Get the minimum and maximum points of the transformed cloud.
  pcl::PointXYZ minPoint, maxPoint;
  pcl::getMinMax3D (*cloudPointsProjected, minPoint, maxPoint);
  const Eigen::Vector3f meanDiagonal = 0.5f * (maxPoint.getVector3fMap () + minPoint.getVector3fMap ());

  //5) given a box centered at the origin with size (max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z) the transformation you have to apply is Rotation = (e0, e1, e0 X e1) & Translation = Rotation * center_diag + (c0, c1, c2)

  //Final transform
  const Eigen::Quaternionf bboxQuaternion (eigenVectorsPCA);  //Quaternions are a way to do rotations https://www.youtube.com/watch?v=mHVwd8gYLnI
  Eigen::Matrix3f rotationMatrix = bboxQuaternion.matrix ();
  Eigen::Vector3f ea = rotationMatrix.eulerAngles (0, 1, 2);

  const Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + Centroid2D.head<3> ();
  /*
   | pt5 _______pt6(max)
   |     |\      \
   |     | \      \
   |     |  \______\
   | pt4 \  |pt1   |pt2
   |      \ |      |
   |       \|______|
   | pt0(min)    pt3
   |------------------------>X
   */

  //Get vertex of cube
  pcl::PointCloud<pcl::PointXYZ> point8;
  point8.resize (8);
  point8.points.at (0) = (pcl::PointXYZ (minPoint.x, minPoint.y, minPoint.z));
  point8.points.at (1) = (pcl::PointXYZ (minPoint.x, minPoint.y, maxPoint.z));
  point8.points.at (2) = (pcl::PointXYZ (maxPoint.x, minPoint.y, maxPoint.z));
  point8.points.at (3) = (pcl::PointXYZ (maxPoint.x, minPoint.y, minPoint.z));
  point8.points.at (4) = (pcl::PointXYZ (minPoint.x, maxPoint.y, minPoint.z));
  point8.points.at (5) = (pcl::PointXYZ (minPoint.x, maxPoint.y, maxPoint.z));
  point8.points.at (6) = (pcl::PointXYZ (maxPoint.x, maxPoint.y, maxPoint.z));
  point8.points.at (7) = (pcl::PointXYZ (maxPoint.x, maxPoint.y, minPoint.z));

  Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity ();
  transform_2.translation () << bboxTransform[0], bboxTransform[1], bboxTransform[2];
  transform_2.rotate (Eigen::AngleAxisf (ea[0], Eigen::Vector3f::UnitX ()));
  transform_2.rotate (Eigen::AngleAxisf (ea[1], Eigen::Vector3f::UnitY ()));
  transform_2.rotate (Eigen::AngleAxisf (ea[2], Eigen::Vector3f::UnitZ ()));

  pcl::transformPointCloud (point8, point8, transform_2);

  pcl::getMinMax3D (cloud_3d, out_minPoint, out_maxPoint);

  point8.points[0].z = out_minPoint.z;
  point8.points[1].z = out_minPoint.z;
  point8.points[4].z = out_minPoint.z;
  point8.points[5].z = out_minPoint.z;
  point8.points[2].z = out_maxPoint.z;
  point8.points[3].z = out_maxPoint.z;
  point8.points[6].z = out_maxPoint.z;
  point8.points[7].z = out_maxPoint.z;

  out_cube.resize (8);
#pragma omp parallel for
  for (size_t i = 0; i < point8.size (); ++i)
  {
    out_cube.at (i) = point8.points[i];
  }

  //double width, height, depth;
  Eigen::Matrix3f EigenVectorsPCA;
  Eigen::Quaternionf BboxQuaternion;
  Eigen::Vector3f BboxTransform;

  EigenVectorsPCA = eigenVectorsPCA;
  BboxQuaternion = bboxQuaternion;
  BboxTransform = bboxTransform;
  //width = maxPoint.x - minPoint.x;
  //height = maxPoint.y - minPoint.y;
  //depth = maxPoint.z - minPoint.z;
}

void
PCA::compute2 (std::vector<pcl::PointXYZ> &out_cube,
               pcl::PointXYZ &out_centroid)
{

  std::vector<float> moment_of_inertia;
  std::vector<float> eccentricity;
  pcl::PointXYZ min_point_AABB, max_point_AABB;
  pcl::PointXYZ min_point_OBB, max_point_OBB, position_OBB;
  Eigen::Matrix3f rotational_matrix_OBB;
  float major_value, middle_value, minor_value;
  Eigen::Vector3f major_vector, middle_vector, minor_vector;
  Eigen::Vector3f mass_center;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2d_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  *cloud_2d_ptr = cloud_2d;

  // Find Principle Axis And Bounding Box For Every Cluster
  pcl::MomentOfInertiaEstimation<pcl::PointXYZ> MOIE;
  MOIE.setInputCloud (cloud_2d_ptr);
  MOIE.compute ();
  MOIE.getMomentOfInertia (moment_of_inertia);
  MOIE.getEccentricity (eccentricity);
  MOIE.getAABB (min_point_AABB, max_point_AABB);
  MOIE.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
  MOIE.getEigenValues (major_value, middle_value, minor_value);
  MOIE.getEigenVectors (major_vector, middle_vector, minor_vector);
  MOIE.getMassCenter (mass_center);

}
