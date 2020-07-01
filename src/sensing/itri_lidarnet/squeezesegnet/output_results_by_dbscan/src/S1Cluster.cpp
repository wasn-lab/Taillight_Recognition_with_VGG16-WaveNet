#include "S1Cluster.h"

S1Cluster::S1Cluster() : viewID(nullptr)
{
}

S1Cluster::S1Cluster(boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer, int* input_viewID)
{
  viewer = input_viewer;
  viewID = input_viewID;

  dbscan.setEpsilon (0.6, 0.6, 0.28, 0.6, 0.28);
  dbscan.setMinpts (5, 5, 5, 5, 5);
}

S1Cluster::~S1Cluster()
{
}

CLUSTER_INFO* S1Cluster::getClusters(bool debug, const PointCloud<PointXYZIL>::ConstPtr input, int* cluster_number)
{
  pcl::StopWatch timer;

  vector<pcl::PointIndices> vector_cluster;

  PointCloud<PointXYZIL>::Ptr input_il(new PointCloud<PointXYZIL>);
  *input_il = *input;
  *input_il = VoxelGrid_CUDA().compute<PointXYZIL>(input_il, 0.2);

  PointCloud<PointXYZIL>::Ptr ptr_cur_cloud_il(new PointCloud<PointXYZIL>);
  PointCloud<PointXYZ>::Ptr ptr_cur_cloud(new PointCloud<PointXYZ>);

  for (size_t i = 0; i < input_il->size(); ++i)
  {
    if (input_il->points[i].x != 0 && input_il->points[i].y != 0 && input_il->points[i].z != 0)
    {
      ptr_cur_cloud_il->push_back(input_il->points[i]);
      ptr_cur_cloud->push_back(PointXYZ(input_il->points[i].x, input_il->points[i].y, input_il->points[i].z));
    }
  }

  //*ptr_cur_cloud = VoxelGrid_CUDA ().compute <PointXYZ> (ptr_cur_cloud, 0.2);

#if ENABLE_DEBUG_MODE == true
  cout << "-------------------------------Part 0 : get cluster_vector " << timer.getTimeSeconds() << ","
       << ptr_cur_cloud->size() << endl;
#endif

#if ENABLE_LABEL_MODE == true
  dbscan.setInputCloud<PointXYZIL>(ptr_cur_cloud_il);
#else
  dbscan.setInputCloud<PointXYZ> (ptr_cur_cloud);
#endif

  dbscan.segment (vector_cluster);

  

#if ENABLE_DEBUG_MODE == true
  cout << "-------------------------------Part 1 : get cluster_vector " << timer.getTimeSeconds() << ","
       << ptr_cur_cloud->size() << endl;
#endif

  vector<CLUSTER_INFO> cluster_vector;  // initialize an vector of cluster Information
  cluster_vector.resize(vector_cluster.size());

#pragma omp parallel for
  for (size_t i = 0; i < vector_cluster.size(); i++)
  {
    PointCloud<PointXYZ> raw_cloud;
    PointCloud<PointXYZIL> raw_cloud_il;

    raw_cloud.resize(vector_cluster.at(i).indices.size());
    raw_cloud_il.resize(vector_cluster.at(i).indices.size());

    for (size_t j = 0; j < vector_cluster.at(i).indices.size(); j++)
    {
      raw_cloud.points[j] = ptr_cur_cloud->points[vector_cluster.at(i).indices.at(j)];
      raw_cloud_il.points[j] = ptr_cur_cloud_il->points[vector_cluster.at(i).indices.at(j)];
    }

    CLUSTER_INFO cluster_raw;
    cluster_raw.cloud = raw_cloud;
    cluster_raw.cloud_IL = raw_cloud_il;

    pcl::getMinMax3D(cluster_raw.cloud, cluster_raw.min, cluster_raw.max);
    cluster_raw.cluster_tag = 1;

    cluster_vector.at(i) = cluster_raw;
  }

#if ENABLE_DEBUG_MODE == true
  cout << "-------------------------------Part 2 hierarchical feature " << timer.getTimeSeconds() << endl;
#endif
/*
  for (size_t i = 0; i < vector_cluster.size (); i++)
  {
    if (cluster_vector.at (i).cluster_tag == 1)
    {
      for (size_t j = 0; j < cluster_vector.size (); j++)
      {
        if (j != i && cluster_vector.at (j).cluster_tag == 1)
        {
          if (cluster_vector.at (j).max.x < cluster_vector.at (i).max.x && cluster_vector.at (j).max.y <
  cluster_vector.at (i).max.y
              && cluster_vector.at (j).max.z < cluster_vector.at (i).max.z && cluster_vector.at (j).min.x >
  cluster_vector.at (i).min.x
              && cluster_vector.at (j).min.y > cluster_vector.at (i).min.y && cluster_vector.at (j).min.z >
  cluster_vector.at (i).min.z)
          {
            cluster_vector.at (i).cloud += cluster_vector.at (j).cloud;

            cluster_vector.at (i).cloud_IL += cluster_vector.at (j).cloud_IL;

            cluster_vector.at (j).cluster_tag = 0;
          }
        }
      }
    }
  }
*/
#if ENABLE_DEBUG_MODE == true
  cout << "-------------------------------Part 3 hierarchical clustering " << timer.getTimeSeconds() << endl;
#endif

#pragma omp parallel for
  for (size_t i = 0; i < vector_cluster.size(); i++)
  {
    if (cluster_vector.at(i).cluster_tag == 1)
    {
      // std::vector<pcl::PointXY> convexhull;
      // pcl::PointCloud<pcl::PointXYZ> convexhull;
      // TODO get min and max to reduce noise
      // pcl::getMinMax3D (cloud_3d, out_minPoint, out_maxPoint);

      /*       PCA bbox;
       bbox.setInputCloud (cluster_vector.at (i).cloud);
       bbox.compute (cluster_vector.at (i).obb_vertex, cluster_vector.at (i).center, cluster_vector.at (i).covariance,
       cluster_vector.at (i).min, cluster_vector.at (i).max);*/



      cluster_vector.at(i).dx = fabs(cluster_vector.at(i).max.x - cluster_vector.at(i).min.x);
      cluster_vector.at(i).dy = fabs(cluster_vector.at(i).max.y - cluster_vector.at(i).min.y);
      cluster_vector.at(i).dz = fabs(cluster_vector.at(i).max.z - cluster_vector.at(i).min.z);
      cluster_vector.at(i).center = PointXYZ((cluster_vector.at(i).max.x + cluster_vector.at(i).min.x) / 2,
                                             (cluster_vector.at(i).max.y + cluster_vector.at(i).min.y) / 2, 0);

      cluster_vector.at(i).dis_center_origin = geometry::distance(cluster_vector.at(i).center, PointXYZ(0, 0, 0));

      cluster_vector[i].abb_vertex.resize(8);
      cluster_vector[i].abb_vertex.at(0) =
          PointXYZ(cluster_vector[i].min.x, cluster_vector[i].min.y, cluster_vector[i].min.z);
      cluster_vector[i].abb_vertex.at(1) =
          PointXYZ(cluster_vector[i].min.x, cluster_vector[i].min.y, cluster_vector[i].max.z);
      cluster_vector[i].abb_vertex.at(2) =
          PointXYZ(cluster_vector[i].max.x, cluster_vector[i].min.y, cluster_vector[i].max.z);
      cluster_vector[i].abb_vertex.at(3) =
          PointXYZ(cluster_vector[i].max.x, cluster_vector[i].min.y, cluster_vector[i].min.z);
      cluster_vector[i].abb_vertex.at(4) =
          PointXYZ(cluster_vector[i].min.x, cluster_vector[i].max.y, cluster_vector[i].min.z);
      cluster_vector[i].abb_vertex.at(5) =
          PointXYZ(cluster_vector[i].min.x, cluster_vector[i].max.y, cluster_vector[i].max.z);
      cluster_vector[i].abb_vertex.at(6) =
          PointXYZ(cluster_vector[i].max.x, cluster_vector[i].max.y, cluster_vector[i].max.z);
      cluster_vector[i].abb_vertex.at(7) =
          PointXYZ(cluster_vector[i].max.x, cluster_vector[i].max.y, cluster_vector[i].min.z);

      /*
       cluster_vector.at (i).obb_center = PointXYZ (
       (cluster_vector[i].obb_vertex.at (0).x + cluster_vector[i].obb_vertex.at (4).x + cluster_vector[i].obb_vertex.at
       (7).x
       + cluster_vector[i].obb_vertex.at (3).x) / 4,
       (cluster_vector[i].obb_vertex.at (0).y + cluster_vector[i].obb_vertex.at (4).y + cluster_vector[i].obb_vertex.at
       (7).y
       + cluster_vector[i].obb_vertex.at (3).y) / 4,
       0);

       cluster_vector.at (i).dis_abbc_obbc = geometry::distance (cluster_vector.at (i).obb_center, cluster_vector.at
       (i).center);

       cluster_vector.at (i).dis_abb_cldc_min = 0;
       cluster_vector.at (i).dis_obb_cldc_min = 0;

       for (size_t j = 0; j < 8; j++)
       {
       float dist_ac = geometry::distance (cluster_vector.at (i).cld_center, cluster_vector[i].abb_vertex.at (j));
       float dist_oc = geometry::distance (cluster_vector.at (i).cld_center, cluster_vector[i].obb_vertex.at (j));
       if (dist_ac < cluster_vector.at (i).dis_abb_cldc_min || j == 0)
       cluster_vector.at (i).dis_abb_cldc_min = dist_ac;
       if (dist_oc < cluster_vector.at (i).dis_obb_cldc_min || j == 0)
       cluster_vector.at (i).dis_obb_cldc_min = dist_oc;
       }
       */
    }
  }

#if ENABLE_DEBUG_MODE == true
  cout << "-------------------------------Part 4 all feature " << timer.getTimeSeconds() << endl;
#endif

#pragma omp parallel for
  for (size_t i = 0; i < vector_cluster.size(); i++)
  {
    if (cluster_vector.at(i).cluster_tag == 1)
    {
      // if (cluster_vector.at (i).dx > 20 || cluster_vector.at (i).dy > 20 || cluster_vector.at (i).dz > 4)
      //   cluster_vector.at (i).cluster_tag = 0;

      if (cluster_vector.at(i).dis_center_origin < 40)
      {
        //         if (cluster_vector.at (i).center.y > 4.5 || cluster_vector.at (i).center.y < -3.5)
        //         {

        //           if (cluster_vector.at (i).dz < 0.3)
        //             cluster_vector.at (i).cluster_tag = 0;

        // //          if (cluster_vector.at (i).min.z > -2.4 && cluster_vector.at (i).center.x > 0 &&
        // (cluster_vector.at (i).dx > 2 || cluster_vector.at (i).dy > 2))   //-1.3
        // //            cluster_vector.at (i).cluster_tag = 0;

        //           // if (cluster_vector.at (i).min.z > -1.5)
        //           //   cluster_vector.at (i).cluster_tag = 0;

        //   if (cluster_vector.at(i).max.z < -2.0)
        //   {
        //     cluster_vector.at(i).cluster_tag = 0;
        //   }
        // }
        // else
        // {
        //   // if (cluster_vector.at (i).max.z < -1.9 && cluster_vector.at (i).dy > 0.4)
        //   //   cluster_vector.at (i).cluster_tag = 0;
        // }

        // ============== label counting for providing cluster_tag with class types ==================
        if (cluster_vector.at(i).cluster_tag == 1)
        {
          size_t cnt_person = 0;
          size_t cnt_motor = 0;
          size_t cnt_car = 0;
          size_t cnt_rule = 0;

          for (size_t j = 0; j < cluster_vector.at(i).cloud_IL.size(); j++)
          {
            switch (cluster_vector.at(i).cloud_IL.points.at(j).label)
            {
              case nnClassID::Person:
                cnt_person++;
                break;
              case nnClassID::Motobike:
                cnt_motor++;
                break;
              case nnClassID::Car:
                cnt_car++;
                break;
              default:
                cnt_rule++;
            }
            if (j > 100)
            {
              break;
            }
          }

          size_t cnt_max = max(max(cnt_person, cnt_motor), max(cnt_car, cnt_rule));

          if (cnt_max == cnt_person)
          {
            cluster_vector.at(i).cluster_tag = nnClassID::Person;
          }
          else if (cnt_max == cnt_motor)
          {
            cluster_vector.at(i).cluster_tag = nnClassID::Motobike;
          }
          else if (cnt_max == cnt_car)
          {
            cluster_vector.at(i).cluster_tag = nnClassID::Car;
          }
          else
          {
            if (cnt_person == 0 && cnt_motor == 0 && cnt_car == 0)
            {
              cluster_vector.at(i).cluster_tag = nnClassID::Rule;
            }
            else
            {
              size_t cnt_2nd_max = max(max(cnt_person, cnt_motor), cnt_car);

              if (cnt_2nd_max == cnt_person)
              {
                cluster_vector.at(i).cluster_tag = nnClassID::Person;
              }
              else if (cnt_2nd_max == cnt_motor)
              {
                cluster_vector.at(i).cluster_tag = nnClassID::Motobike;
              }
              else if (cnt_2nd_max == cnt_car)
              {
                cluster_vector.at(i).cluster_tag = nnClassID::Car;
              }
            }
          }
          if(cluster_vector.at(i).cluster_tag == nnClassID::Car)
          {
/*
            UseApproxMVBB bbox2;
            bbox2.setInputCloud(cluster_vector.at(i).cloud);
            bbox2.Compute(cluster_vector.at(i).obb_vertex, cluster_vector.at(i).cld_center,
                          cluster_vector.at(i).min, cluster_vector.at(i).max, 
                          cluster_vector.at(i).convex_hull);
*/
            estimator_.getShapeAndPose(cluster_vector.at(i));
          }
        }
      }
    }
  }

#if ENABLE_DEBUG_MODE == true
  cout << "-------------------------------Part 5 reduce noise " << timer.getTimeSeconds() << endl;
#endif

#if ENABLE_DEBUG_MODE == true

  viewer->removeAllShapes(0);
  for (size_t i = 0; i < vector_cluster.size(); i++)
  {
    if (cluster_vector.at(i).cluster_tag > 0)
    {
      PointXYZ pt0;
      PointXYZ pt1;
      PointXYZ pt2;
      PointXYZ pt3;
      PointXYZ pt4;
      PointXYZ pt5;
      PointXYZ pt6;
      PointXYZ pt7;

      if (cluster_vector.at(i).dis_abb_cldc_min <= cluster_vector.at(i).dis_obb_cldc_min ||
          cluster_vector.at(i).dis_abbc_obbc > 0.8)
      {
        pt0 = PointXYZ(cluster_vector[i].min.x, cluster_vector[i].min.y, cluster_vector[i].min.z);
        pt1 = PointXYZ(cluster_vector[i].min.x, cluster_vector[i].min.y, cluster_vector[i].max.z);
        pt2 = PointXYZ(cluster_vector[i].max.x, cluster_vector[i].min.y, cluster_vector[i].max.z);
        pt3 = PointXYZ(cluster_vector[i].max.x, cluster_vector[i].min.y, cluster_vector[i].min.z);
        pt4 = PointXYZ(cluster_vector[i].min.x, cluster_vector[i].max.y, cluster_vector[i].min.z);
        pt5 = PointXYZ(cluster_vector[i].min.x, cluster_vector[i].max.y, cluster_vector[i].max.z);
        pt6 = PointXYZ(cluster_vector[i].max.x, cluster_vector[i].max.y, cluster_vector[i].max.z);
        pt7 = PointXYZ(cluster_vector[i].max.x, cluster_vector[i].max.y, cluster_vector[i].min.z);
      }
      else
      {
        pt0 = (cluster_vector[i].obb_vertex.at(0));
        pt1 = (cluster_vector[i].obb_vertex.at(1));
        pt2 = (cluster_vector[i].obb_vertex.at(2));
        pt3 = (cluster_vector[i].obb_vertex.at(3));
        pt4 = (cluster_vector[i].obb_vertex.at(4));
        pt5 = (cluster_vector[i].obb_vertex.at(5));
        pt6 = (cluster_vector[i].obb_vertex.at(6));
        pt7 = (cluster_vector[i].obb_vertex.at(7));
      }

#if 0
      viewer->addLine (pt0, pt1, 255, 255, 255, to_string (*viewID));
      ++*viewID;
      viewer->addLine (pt0, pt3, 255, 255, 255, to_string (*viewID));
      ++*viewID;
      viewer->addLine (pt0, pt4, 255, 255, 255, to_string (*viewID));
      ++*viewID;
      viewer->addLine (pt4, pt5, 255, 255, 255, to_string (*viewID));
      ++*viewID;
      viewer->addLine (pt4, pt7, 255, 255, 255, to_string (*viewID));
      ++*viewID;
      viewer->addLine (pt1, pt5, 255, 255, 255, to_string (*viewID));
      ++*viewID;
      viewer->addLine (pt5, pt6, 255, 255, 255, to_string (*viewID));
      ++*viewID;
      viewer->addLine (pt6, pt7, 255, 255, 255, to_string (*viewID));
      ++*viewID;
      viewer->addLine (pt1, pt2, 255, 255, 255, to_string (*viewID));
      ++*viewID;
      viewer->addLine (pt3, pt7, 255, 255, 255, to_string (*viewID));
      ++*viewID;
      viewer->addLine (pt2, pt3, 255, 255, 255, to_string (*viewID));
      ++*viewID;
      viewer->addLine (pt2, pt6, 255, 255, 255, to_string (*viewID));
      ++*viewID;
#endif

      /*      int intPart = cluster_vector.at (i).dis_center_origin;
       int floatPart = (cluster_vector.at (i).dis_center_origin - (float) intPart) * 100;
       viewer->addText3D (to_string (intPart) + "." + to_string (floatPart), cluster_vector.at (i).max, 0.7, 255, 255,
       255, to_string (*viewID));
       ++*viewID;*/

      /*      viewer->addText3D (to_string (cluster_vector.at (i).cluster_tag), cluster_vector.at (i).max, 0.7, 255,
         255, 255, to_string (*viewID));
            ++*viewID;*/

      /*
       viewer->addText3D (to_string (cluster_vector.at (i).cloud.size()/(cluster_vector.at (i).dx*cluster_vector.at
       (i).dy)), cluster_vector.at (i).max, 0.7, 255, 255, 255, to_string (*viewID));
       ++*viewID;
       */

      /*
       viewer->addText3D (to_string (cluster_vector.at (i).cluster_tag) + "", cluster_vector.at (i).max, 0.7, 255, 255,
       255, to_string (*viewID));
       ++*viewID;
       */

#if ENABLE_DEBUG_MODE == true

      for (size_t j = 0; j < cluster_vector.at(i).convex_hull.size(); j++)
      {
        PointCloud<PointXYZ>::Ptr UIcloud(new PointCloud<PointXYZ>);
        *UIcloud = cluster_vector.at(i).convex_hull;
        viewer->addPolygon<PointXYZ>(UIcloud, 0, 255, 0, to_string(*viewID));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, to_string(*viewID));

        ++*viewID;
      }

#endif
    }
  }
  cout << "-------------------------------Part 6 UI " << timer.getTimeSeconds() << endl << endl;

#endif

  int k = 0;
  for (size_t i = 0; i < cluster_vector.size(); i++)
  {
    if (cluster_vector.at(i).cluster_tag > 0)
    {
      cluster_vector.at(k) = cluster_vector.at(i);
      k++;
    }
  }
  cluster_vector.resize(k);

  *cluster_number = cluster_vector.size();
  auto* cluster_modify =
      new CLUSTER_INFO[cluster_vector.size()];  // initialize an vector of cluster point cloud

  for (size_t i = 0; i < cluster_vector.size(); i++)
  {
    cluster_modify[i] = cluster_vector.at(i);
  }

  return cluster_modify;
}
