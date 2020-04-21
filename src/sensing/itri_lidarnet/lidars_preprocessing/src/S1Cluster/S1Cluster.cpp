#include "S1Cluster.h"

S1Cluster::S1Cluster () :
    viewID (NULL)
{
  initial (NULL, NULL);
}

S1Cluster::S1Cluster (boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer,
                      int *input_viewID)
{
  initial (input_viewer, input_viewID);
}

S1Cluster::~S1Cluster ()
{
}

void
S1Cluster::initial (boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer,
                    int *input_viewID)
{
  viewer = input_viewer;
  viewID = input_viewID;

  plane_coef.values.resize (4);  //ax+by+cz+d=0
  plane_coef.values[0] = 0;
  plane_coef.values[1] = 0;
  plane_coef.values[2] = 2.3;
  plane_coef.values[3] = 0;

  dbscan.setEpsilon (0.80);
  dbscan.setMinpts (3);
}

void
S1Cluster::setPlaneParameter (pcl::ModelCoefficients inputCoef)
{
  plane_coef = inputCoef;
}

CLUSTER_INFO*
S1Cluster::getClusters (bool debug,
                        PointCloud<PointXYZ>::ConstPtr input,
                        int *cluster_number)
{
  pcl::StopWatch stopWatch;

  // ========================================================================================================== Part I : get raw_cluster

  vector<pcl::PointIndices> raw_cluster;

  dbscan.setInputCloud<PointXYZ> (input);
  dbscan.segment (raw_cluster);

  if (debug)
  {
    cout << "-------------------------------get raw_cluster " << stopWatch.getTimeSeconds () << endl;
  }

  // ========================================================================================================== Part II : get cluster_vector

  vector<CLUSTER_INFO> cluster_vector;  // initialize an vector of cluster Information
  cluster_vector.resize (raw_cluster.size ());

#pragma omp parallel for
  for (size_t i = 0; i < raw_cluster.size (); i++)
  {
    PointCloud<PointXYZ> raw_cloud;
    raw_cloud.resize (raw_cluster.at (i).indices.size ());
#pragma omp parallel for
    for (size_t j = 0; j < raw_cluster.at (i).indices.size (); j++)
    {
      raw_cloud.points[j] = input->points[raw_cluster.at (i).indices.at (j)];
    }
    CLUSTER_INFO cluster_raw;
    cluster_raw.cloud = raw_cloud;
    cluster_raw.cloud.width = cluster_raw.cloud.size ();
    cluster_raw.cloud.height = 1;
    cluster_raw.cloud.is_dense = false;
    cluster_raw.cloud.points.resize (cluster_raw.cloud.width);

    pcl::getMinMax3D (cluster_raw.cloud, cluster_raw.min, cluster_raw.max);
    cluster_raw.cluster_tag = 1;

    cluster_vector.at (i) = cluster_raw;
  }

  if (debug)
  {
    cout << "-------------------------------get cluster_vector " << stopWatch.getTimeSeconds () << endl;
  }

  // ========================================================================================================== Part III hierarchical clustering

  for (size_t i = 0; i < raw_cluster.size (); i++)
  {
    if (cluster_vector.at (i).cluster_tag == 1)
    {
      for (size_t j = 0; j < cluster_vector.size (); j++)
      {
        if (j != i && cluster_vector.at (j).cluster_tag == 1)
        {
          if (cluster_vector.at (j).max.x < cluster_vector.at (i).max.x && cluster_vector.at (j).max.y < cluster_vector.at (i).max.y
              && cluster_vector.at (j).max.z < cluster_vector.at (i).max.z && cluster_vector.at (j).min.x > cluster_vector.at (i).min.x
              && cluster_vector.at (j).min.y > cluster_vector.at (i).min.y && cluster_vector.at (j).min.z > cluster_vector.at (i).min.z)
          {
            cluster_vector.at (i).cloud += cluster_vector.at (j).cloud;
            cluster_vector.at (i).cloud.width = cluster_vector.at (i).cloud.size ();
            cluster_vector.at (i).cloud.height = 1;
            cluster_vector.at (i).cloud.is_dense = false;
            cluster_vector.at (i).cloud.points.resize (cluster_vector.at (i).cloud.width);
            cluster_vector.at (j).cluster_tag = 0;
          }
        }
      }
    }
  }

  if (debug)
  {
    cout << "-------------------------------hierarchical " << stopWatch.getTimeSeconds () << endl;
  }

  // ========================================================================================================== Part IV get partial feature

#pragma omp parallel for
  for (size_t i = 0; i < raw_cluster.size (); i++)
  {
    if (cluster_vector.at (i).cluster_tag == 1)
    {
      PCA bbox;
      bbox.setInputCloud (cluster_vector.at (i).cloud);
      bbox.compute (cluster_vector.at (i).obb_vertex, cluster_vector.at (i).center, cluster_vector.at (i).covariance, cluster_vector.at (i).min,
                    cluster_vector.at (i).max);

      cluster_vector.at (i).dis_max_min = geometry::distance (cluster_vector.at (i).max, cluster_vector.at (i).min);
      cluster_vector.at (i).dx = fabs (cluster_vector.at (i).max.x - cluster_vector.at (i).min.x);
      cluster_vector.at (i).dy = fabs (cluster_vector.at (i).max.y - cluster_vector.at (i).min.y);
      cluster_vector.at (i).dz = fabs (cluster_vector.at (i).max.z - cluster_vector.at (i).min.z);
      cluster_vector.at (i).dis_center_origin = geometry::distance (cluster_vector.at (i).center, PointXYZ (0, 0, 0));
      cluster_vector.at (i).angle_from_x_axis = pcl::getAngle3D (Eigen::Vector3f (cluster_vector.at (i).center.x, cluster_vector.at (i).center.y, 0),
                                                                 Eigen::Vector3f (1, 0, 0), true);
      cluster_vector.at (i).found_num = false;
      cluster_vector.at (i).tracking_id = 0;
      cluster_vector.at (i).track_last_center = PointXYZ (99999, 99999, 99999);
      cluster_vector.at (i).predict_next_center = cluster_vector.at (i).center;
      cluster_vector.at (i).velocity = PointXYZ (0, 0, 0);
      cluster_vector.at (i).confidence = 0;
    }
  }

  if (debug)
  {
    cout << "-------------------------------partial " << stopWatch.getTimeSeconds () << endl;
  }
  /*
   for (size_t i = 0; i < raw_cluster.size (); i++)
   {
   if (cluster_vector.at (i).cluster_tag == 1 && cluster_vector.at (i).cloud.size () > 3)
   {
   PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new PointCloud<pcl::PointXYZ>);
   *cloud_hull = cluster_vector.at (i).cloud;
   pcl::ConvexHull<pcl::PointXYZ> chull;
   chull.setInputCloud (cloud_hull);
   chull.setDimension (3);
   chull.setComputeAreaVolume (true);
   chull.reconstruct (*cloud_hull);
   cluster_vector.at (i).hull_vol = chull.getTotalVolume ();

   cout << "-------" << chull.getTotalVolume ();

   //     gpu::PseudoConvexHull3D pch(1e5);
   //     gpu::PseudoConvexHull3D::Cloud cloud_device;
   //     gpu::PseudoConvexHull3D::Cloud convex_device;
   //     cloud_device.upload(cluster_vector.at (i).cloud.points);
   //     pch.reconstruct(cloud_device, convex_device);
   //
   //     pcl::PolygonMesh mesh;
   //
   //     pcl::PointCloud<pcl::PointXYZ>::Ptr convex_ptr;
   //     convex_ptr.reset(new pcl::PointCloud<pcl::PointXYZ>((int)convex_device.size(), 1));
   //     convex_device.download(convex_ptr->points);
   //
   //     cout << "-------" << chull.getTotalVolume ()<<endl;

   }
   }*/

  // ========================================================================================================== Part V reduce noise
#pragma omp parallel for
  for (size_t i = 0; i < raw_cluster.size (); i++)
  {
    if (cluster_vector.at (i).cluster_tag == 1)
    {
      /*
       float max2plane = pointToPlaneDistance (cluster_vector.at (i).max, plane_coef.values[0], plane_coef.values[1], plane_coef.values[2],
       plane_coef.values[3]);
       float min2plane = pointToPlaneDistance (cluster_vector.at (i).min, plane_coef.values[0], plane_coef.values[1], plane_coef.values[2],
       plane_coef.values[3]);*/
      //float dxdy = fabs (cluster_vector.at (i).dx - cluster_vector.at (i).dy);
      //constrain : too small or thin objects
      if (cluster_vector.at(i).dz < 0.8)
      {
            cluster_vector.at(i).cluster_tag = 0;
      }

      //constrain : too big
      if (cluster_vector.at(i).dis_max_min > 15 || cluster_vector.at(i).dx > 15 || cluster_vector.at(i).dy > 15 ||
          cluster_vector.at(i).dz > 3)
      {
            cluster_vector.at(i).cluster_tag = 0;
      }

      //      //constrain : small size noise
      //      if (cluster_vector.at (i).cloud.size () < 4)
      //        cluster_vector.at (i).cluster_tag = 0;
      //
      //      //constrain : air object
      //      if (min2plane > 1.8 )
      //        cluster_vector.at (i).cluster_tag = 0;
      //
      //      //curb
      //      if (max2plane < 0.8 && (cluster_vector.at (i).dx > 2 || cluster_vector.at (i).dy > 2 || cluster_vector.at
      //      (i).dz <0.3))
      //         cluster_vector.at (i).cluster_tag = 0;

      //      if( cluster_vector.at (i).center.y < -1.3)
      //      {
      //       if (max2plane < 0.7)
      //          cluster_vector.at (i).cluster_tag = 0;
      //
      //       if (min2plane > 0.6 )
      //         cluster_vector.at (i).cluster_tag = 0;
      //
      //        if (cluster_vector.at (i).dz > 2.4 && cluster_vector.at (i).dz<3 && cluster_vector.at (i).dis_max_min
      //        >7)
      //          cluster_vector.at (i).cluster_tag = 0;
      //
      //        if (cluster_vector.at (i).dz > 2 && cluster_vector.at (i).cloud.size() < 100)
      //          cluster_vector.at (i).cluster_tag = 0;
      //
      //        if (cluster_vector.at (i).dz < 1)
      //          cluster_vector.at (i).cluster_tag = 0;
      //
      //        if ((cluster_vector.at (i).cloud.size()/cluster_vector.at (i).hull_vol) <20 )
      //          cluster_vector.at (i).cluster_tag = 0;
      //
      //        if (cluster_vector.at (i).dz > 2)
      //           cluster_vector.at (i).cluster_tag = 0;
      //
      //        //tree
      //        if (max2plane > 2 && ((cluster_vector.at (i).dx <2  && cluster_vector.at (i).dy < 2) ||
      //        (cluster_vector.at (i).cloud.size()/cluster_vector.at (i).hull_vol) >50))
      //           cluster_vector.at (i).cluster_tag = 0;
      //      }

    }
  }

  if (debug)
  {
    cout << "-------------------------------reduce noise " << stopWatch.getTimeSeconds () << endl;
  }

  // ========================================================================================================== Part VI get all feature

  if (debug)
  {
    cout << "-------------------------------ConvexHull " << stopWatch.getTimeSeconds () << endl;
  }

  /*#pragma omp parallel for
   for (size_t i = 0; i < raw_cluster.size (); i++)
   {
   if (cluster_vector.at (i).cluster_tag == 1)
   {
   pcl::gpu::Feature::PointCloud cloud_d (cluster_vector.at (i).cloud.size ());
   cloud_d.upload (cluster_vector.at (i).cloud.points);

   pcl::gpu::Feature::Normals normals_d (cluster_vector.at (i).cloud.size ());
   pcl::gpu::NormalEstimation ne_d;
   ne_d.setInputCloud (cloud_d);
   ne_d.setViewPoint (0, 0, 0);
   ne_d.setRadiusSearch (0.1, 100);
   ne_d.compute (normals_d);


   pcl::gpu::DeviceArray2D<FPFHSignature33> fpfh33_features;

   pcl::gpu::FPFHEstimation fpfh_gpu;
   fpfh_gpu.setInputCloud (cloud_d);
   fpfh_gpu.setInputNormals (normals_d);
   fpfh_gpu.setRadiusSearch (1.2, 5);
   fpfh_gpu.compute (fpfh33_features);

   vector<FPFHSignature33> descriptor;
   int host_step;
   fpfh33_features.download (descriptor, host_step);
   cout << "---" << descriptor.size () << endl;
   cluster_vector.at (i).fpfh33[0] = descriptor.size ();



   pcl::gpu::DeviceArray<VFHSignature308> vfh_features;

   gpu::VFHEstimation pc_gpu;
   pc_gpu.setSearchSurface (cloud_d);
   pc_gpu.setInputNormals (normals_d);
   pc_gpu.setRadiusSearch (1.2, 50);
   pc_gpu.compute (vfh_features);

   vector<VFHSignature308> downloaded;
   vfh_features.download (downloaded);
   cout << "---" << downloaded.size () << endl;



   #pragma omp parallel for
   for (int k = 0; k < 33; ++k)
   {
   cluster_vector.at (i).fpfh33[k] = descriptor[0].histogram[k];
   }


   vector<PointXYZ> downloaded;
   normals_d.download (downloaded);

   pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
   normals->resize (downloaded.size ());
   #pragma omp parallel for
   for (int j = 0; j < downloaded.size (); ++j)
   {
   normals->points[j].normal_x = downloaded[j].x;
   normals->points[j].normal_y = downloaded[j].y;
   normals->points[j].normal_z = downloaded[j].z;
   }

   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
   *cloud = cluster_vector.at (i).cloud;

   pcl::PointCloud<pcl::GRSDSignature21>::Ptr descriptors (new pcl::PointCloud<pcl::GRSDSignature21> ());
   pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree (new pcl::search::KdTree<pcl::PointXYZ>);
   pcl::GRSDEstimation<pcl::PointXYZ, pcl::Normal, pcl::GRSDSignature21> grsd;
   grsd.setInputCloud (cloud);
   grsd.setInputNormals (normals);
   grsd.setSearchMethod (kdtree);
   grsd.setRadiusSearch (0.2);
   grsd.compute (*descriptors);

   for (int k = 0; k < 21; ++k)
   {
   cluster_vector.at (i).GRSD21[k] = descriptors->points[0].histogram[k];
   }
   }
   }*/

  if (debug)
  {
    cout << "-------------------------------GRSDSignature21 " << stopWatch.getTimeSeconds () << endl;
    viewer->removeAllShapes (0);
    for (size_t i = 0; i < raw_cluster.size (); i++)
    {
      if (cluster_vector.at (i).cluster_tag > 0)
      {
        /*
         //               Z
         //               | pt5  _____________  pt6(max)
         //       Y       |     |\             \
         //        \      |     | \             \
         //         \     |     |  \_____________\
         //          \    | pt4 \  |pt1          | pt2
         //           \   |      \ |             |
         //            \  |       \|_____________|
         //             \ | pt0(min)          pt3
         //              \|----------------------------------->X
         */

#if 1
        PointXYZ pt0 (cluster_vector[i].min.x, cluster_vector[i].min.y, cluster_vector[i].min.z);
        PointXYZ pt1 (cluster_vector[i].min.x, cluster_vector[i].min.y, cluster_vector[i].max.z);
        PointXYZ pt2 (cluster_vector[i].max.x, cluster_vector[i].min.y, cluster_vector[i].max.z);
        PointXYZ pt3 (cluster_vector[i].max.x, cluster_vector[i].min.y, cluster_vector[i].min.z);
        PointXYZ pt4 (cluster_vector[i].min.x, cluster_vector[i].max.y, cluster_vector[i].min.z);
        PointXYZ pt5 (cluster_vector[i].min.x, cluster_vector[i].max.y, cluster_vector[i].max.z);
        PointXYZ pt6 (cluster_vector[i].max.x, cluster_vector[i].max.y, cluster_vector[i].max.z);
        PointXYZ pt7 (cluster_vector[i].max.x, cluster_vector[i].max.y, cluster_vector[i].min.z);
#else
        PointXYZ pt0 (cluster_vector[i].obb_vertex.at (0));
        PointXYZ pt1 (cluster_vector[i].obb_vertex.at (1));
        PointXYZ pt2 (cluster_vector[i].obb_vertex.at (2));
        PointXYZ pt3 (cluster_vector[i].obb_vertex.at (3));
        PointXYZ pt4 (cluster_vector[i].obb_vertex.at (4));
        PointXYZ pt5 (cluster_vector[i].obb_vertex.at (5));
        PointXYZ pt6 (cluster_vector[i].obb_vertex.at (6));
        PointXYZ pt7 (cluster_vector[i].obb_vertex.at (7));
#endif

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

        //viewer->addText3D (to_string ((int)(cluster_vector.at (i).cloud.size()/cluster_vector.at (i).hull_vol)) , cluster_vector.at (i).max, 0.7, 255, 255, 255, to_string (*viewID));

        //viewer->addText3D (to_string (int (cluster_vector.at (i).GRSD21[10])) + "", cluster_vector.at (i).max, 0.7, 255, 255, 255, to_string (*viewID));

        // see distance
        //viewer->addText3D (to_string ((cluster_vector.at (i).dis_center_origin)) + "m", cluster_vector.at (i).max, 0.7, 255, 255, 255,to_string (*viewID));
        //viewer->addText3D (to_string ((int) (cluster_vector.at (i).min.x)) + "x," + to_string ((int) (cluster_vector.at (i).min.y)) + "y",cluster_vector.at (i).max, 0.7, 255, 255, 255, to_string (*viewID));
        //viewer->addText3D (to_string ((int)(cluster_vector.at (i).dz)) + "dz," + to_string ((int) (cluster_vector.at (i).dis_max_min)), cluster_vector.at (i).max, 0.7, 255, 255, 255,to_string (*viewID));

        // see confidence
        //viewer->addText3D (to_string (cluster_vector.at (i).tag_confidence), cluster_vector.at (i).max, 0.5, 255, 255, 255, to_string (*viewID), 0);

        // see size
        //viewer->addText3D (to_string (cluster_vector.at (i).cloud.size()),cluster_vector.at (i).max, 0.5, 255, 255, 255, to_string (*viewID), 0);

        // see distance and cloud size
        /*
         viewer->addText3D ("D" + to_string ((int) (cluster_vector.at (i).dis_center_origin)) + "S" + to_string ((int) cluster_vector.at (i).cloud.size ()),
         cluster_vector.at (i).max, 0.5, 255, 255, 255, to_string (*viewID), 0);
         */

        // see legth, width, high, dis_max_min
        /*
         viewer->addText3D (
         to_string ((int) cluster_vector.at (i).dx) + " " + to_string ((int) cluster_vector.at (i).dy) + " " + to_string ((int) cluster_vector.at (i).dz) + " " + to_string (cluster_vector.at (i).dis_max_min),
         cluster_vector.at (i).max  , 0.5, 255, 255, 255, to_string (*viewID), 0);
         */

        // see Polygon Area
        /*
         viewer->addText3D (to_string (pcl::calculatePolygonArea (cluster_vector.at(i).cloud)), cluster_vector.at (i).max, 1, 255, 255, 255, to_string (*viewID), 0);
         viewer->setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, to_string (*viewID));
         */

        // see different color of clusters
        /*      PointCloud<PointXYZ>::Ptr ptr_cur_cloud (new PointCloud<PointXYZ>);
         *ptr_cur_cloud = cluster_vector.at (i).cloud;
         viewer->addPointCloud (ptr_cur_cloud, to_string (*viewID));
         viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, ((double) rand () / (RAND_MAX)), ((double) rand () / (RAND_MAX)),
         ((double) rand () / (RAND_MAX)), to_string (*viewID));
         viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, to_string (*viewID));*/

        ++*viewID;
      }
    }
    cout << "-------------------------------UI " << stopWatch.getTimeSeconds () << endl;
  }

  // ========================================================================================================== Part VII

  int k = 0;
  for (size_t i = 0; i < cluster_vector.size (); i++)
  {
    if (cluster_vector.at (i).cluster_tag > 0)
    {
      cluster_vector.at (k) = cluster_vector.at (i);
      k++;
    }
  }
  cluster_vector.resize (k);

  /*  for (vector<CLUSTER_INFO>::const_iterator it = cluster_vector.begin (); it != cluster_vector.end ();)
   {
   if (it->cluster_tag == 0)
   it = cluster_vector.erase (it);
   else
   ++it;
   }*/

  // ========================================================================================================== Part VIII
  *cluster_number = cluster_vector.size ();
  CLUSTER_INFO* cluster_modify = new CLUSTER_INFO[cluster_vector.size ()];  // initialize an vector of cluster point cloud

#pragma omp parallel for
  for (size_t i = 0; i < cluster_vector.size (); i++)
  {
    cluster_modify[i] = cluster_vector.at (i);
  }

  return cluster_modify;
}
