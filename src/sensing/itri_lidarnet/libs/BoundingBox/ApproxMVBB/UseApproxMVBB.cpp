#include "UseApproxMVBB.h"

UseApproxMVBB::UseApproxMVBB()
{
}

UseApproxMVBB::~UseApproxMVBB()
{
}

void UseApproxMVBB::setInputCloud(const pcl::PointCloud<pcl::PointXYZ> input)
{
  cloud_3d = input;
}

void UseApproxMVBB::Compute(std::vector<pcl::PointXYZ>& out_cube, pcl::PointXYZ& out_centroid,
                            pcl::PointXYZ& out_minPoint, pcl::PointXYZ& out_maxPoint,
                            pcl::PointCloud<pcl::PointXYZ>& out_ch)
{
  //    Eigen::Vector4f Centroid3D;
  //    pcl::compute3DCentroid (cloud_3d, Centroid3D);
  //    out_centroid.x = Centroid3D[0];
  //    out_centroid.y = Centroid3D[1];
  //    out_centroid.z = Centroid3D[2];

  /* ABB order
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
  pcl::getMinMax3D(cloud_3d, out_minPoint, out_maxPoint);

  ApproxMVBB::Matrix3Dyn points(3, cloud_3d.size());

  for (size_t i = 0; i < cloud_3d.size(); i++)
  {
    points(0, i) = cloud_3d.points[i].x;
    points(1, i) = cloud_3d.points[i].y;
    points(2, i) = 0;
  }

  ApproxMVBB::OOBB oobb = ApproxMVBB::approximateMVBB(points,
                                                      0.2,  // 0.1
                                                      40,   // 50
                                                      1,    // 1 increasing the grid size decreases speed
                                                      0,    // 0
                                                      0);   // 0

  ApproxMVBB::Vector3List buff;
  buff = oobb.getCornerPoints();

  out_cube.resize(8);

  out_cube.at(1) = pcl::PointXYZ(buff.at(7).x(), buff.at(7).y(), out_maxPoint.z);
  out_cube.at(2) = pcl::PointXYZ(buff.at(6).x(), buff.at(6).y(), out_maxPoint.z);
  out_cube.at(6) = pcl::PointXYZ(buff.at(0).x(), buff.at(0).y(), out_maxPoint.z);
  out_cube.at(5) = pcl::PointXYZ(buff.at(1).x(), buff.at(1).y(), out_maxPoint.z);

  if (out_cube.at(1).x == out_cube.at(2).x && out_cube.at(1).y == out_cube.at(2).y)
  {
    out_cube.at(2) = pcl::PointXYZ(buff.at(2).x(), buff.at(2).y(), out_maxPoint.z);
    out_cube.at(5) = pcl::PointXYZ(buff.at(4).x(), buff.at(4).y(), out_maxPoint.z);
  }

  out_cube.at(0) = pcl::PointXYZ(out_cube.at(1).x, out_cube.at(1).y, out_minPoint.z);
  out_cube.at(3) = pcl::PointXYZ(out_cube.at(2).x, out_cube.at(2).y, out_minPoint.z);
  out_cube.at(7) = pcl::PointXYZ(out_cube.at(6).x, out_cube.at(6).y, out_minPoint.z);
  out_cube.at(4) = pcl::PointXYZ(out_cube.at(5).x, out_cube.at(5).y, out_minPoint.z);

  /* ConvexHull order
  Y
  |        ______
  |       |pt2   |pt1
  |       |      |
  |       |______|
  |       pt3    pt0
  |------------------------>X
  */
  ApproxMVBB::Matrix2Dyn M2Dpoints(2, cloud_3d.size());

  for (size_t i = 0; i < cloud_3d.size(); i++)
  {
    M2Dpoints(0, i) = cloud_3d.points[i].x;
    M2Dpoints(1, i) = cloud_3d.points[i].y;
  }

  ApproxMVBB::ConvexHull2D ch2D(M2Dpoints);
  std::vector<unsigned int> chIndices;

  ch2D.compute();
  chIndices = ch2D.getIndices();

  out_ch.resize(chIndices.size());
  for (size_t i = 0; i < chIndices.size(); i++)
  {
    out_ch.at(i).x = M2Dpoints(0, chIndices[i]);
    out_ch.at(i).y = M2Dpoints(1, chIndices[i]);
    out_ch.at(i).z = out_minPoint.z;
  }
}
