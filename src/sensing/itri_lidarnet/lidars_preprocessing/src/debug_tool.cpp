#include "debug_tool.h"

void show_cloud(PointCloud<PointXYZI>::ConstPtr cloud)
{
  static bool flag = true;
  static int ID = 0;
  static boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D "
                                                                                                           "Viewer"));

  if (flag)
  {
    pcl::visualization::Camera cam;
    cam = CompressFunction().CamPara(-15.847, 3.736, 69.862, 0.899, 0.026, 0.437, 12.458, 1.537, 11.760, 48.793,
                                     105.344, 0.857, 0.857, 0.000, 0.000, 358.000, 1028.000);

    viewer->initCameraParameters();
    viewer->setCameraParameters(cam, 0);
    viewer->addCoordinateSystem(3.0);  // red:x green:y
    viewer->setBackgroundColor(0, 0, 0);
    viewer->setShowFPS(false);

    flag = false;
  }

  viewer->removeAllPointClouds(0);

  viewer->addPointCloud<PointXYZI>(cloud, to_string(ID));
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, to_string(ID));
  viewer->spinOnce();

  ++ID;

  /*  while (!viewer->wasStopped ())
   {
   viewer->spinOnce (100);
   boost::this_thread::sleep (boost::posix_time::microseconds (100000));
   }*/
}

void show_cloud(PointCloud<PointXYZ>::ConstPtr cloud)
{
  static bool flag = true;
  static int ID = 0;
  static boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D "
                                                                                                           "Viewer"));

  if (flag)
  {
    pcl::visualization::Camera cam;
    cam = CompressFunction().CamPara(-15.847, 3.736, 69.862, 0.899, 0.026, 0.437, 12.458, 1.537, 11.760, 48.793,
                                     105.344, 0.857, 0.857, 0.000, 0.000, 358.000, 1028.000);

    viewer->initCameraParameters();
    viewer->setCameraParameters(cam, 0);
    viewer->addCoordinateSystem(3.0);  // red:x green:y
    viewer->setBackgroundColor(0, 0, 0);
    viewer->setShowFPS(false);

    flag = false;
  }

  viewer->removeAllPointClouds(0);

  viewer->addPointCloud<PointXYZ>(cloud, to_string(ID));
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, to_string(ID));
  viewer->spinOnce();

  ++ID;

  /*  while (!viewer->wasStopped ())
   {
   viewer->spinOnce (100);
   boost::this_thread::sleep (boost::posix_time::microseconds (100000));
   }*/
}

void show_cloud(PointCloud<PointNormal>::ConstPtr cloud)
{
  static bool flag = true;
  static int ID = 0;
  static boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D "
                                                                                                           "Viewer"));

  if (flag)
  {
    pcl::visualization::Camera cam;
    cam = CompressFunction().CamPara(1.001, -12.307, 28.940, -0.016, 0.739, 0.673, 1.414, 8.011, 6.644, 0.112, 111.772,
                                     0.857, 0.857, 0.000, 0.000, 1000.000, 500.000);

    viewer->initCameraParameters();
    viewer->addCoordinateSystem(3.0);  // red:x green:y
    viewer->setBackgroundColor(0, 0, 0);
    viewer->setCameraParameters(cam, 0);
    viewer->setShowFPS(false);

    flag = false;
  }

  viewer->removeAllPointClouds(0);

  /* addPointCloudNormals
   * [3] 60 refers i think to only every 10th normal is displayed
   * [4] the normals are 4.5 cm in your coordinate system (should be adjusted if you use large scale data)*/

  viewer->addPointCloudNormals<PointNormal>(cloud, 60, 4.5, to_string(ID));
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, to_string(ID));
  viewer->spinOnce();

  ++ID;
}

void show_cloud(PointCloud<PointXYZRGB>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);

  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud, centroid);

  viewer->setBackgroundColor(0, 0, 0);
  viewer->initCameraParameters();
  pcl::visualization::Camera cam;

  cam =
      CompressFunction().CamPara(1.85068e+09, -2.31115e+10, 1.96841e+10, -0.00150261, 0.648329, 0.761359, 1.00316,
                                 -1.04158, 1.44827, 3.01101e+10, 3.08705e+10, 1.74533e-10, 1.74533e-10, 0, 0, 960, 540);

  viewer->setCameraParameters(cam, 0);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem(1.0);

  while (!viewer->wasStopped())
  {
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> type_XYZRGB_Normal(PointCloud<PointXYZRGB>::ConstPtr cloud,
                                                                        PointCloud<pcl::Normal>::ConstPtr normals)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->initCameraParameters();
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
  viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 10, 0.05, "normals");
  viewer->addCoordinateSystem(1.0);
  viewer->setBackgroundColor(0, 0, 0);
  viewer->setCameraPosition(0.0, 0.0, 50.0, 1.0, 0.0, 0.0, 0);
  viewer->setCameraClipDistances(0.0, 50.0);
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Debug Windows");

  return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> type_XYZ_Normal(PointCloud<PointXYZ>::ConstPtr cloud,
                                                                     PointCloud<pcl::Normal>::ConstPtr normals)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addPointCloud<PointXYZ>(cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addPointCloudNormals<PointXYZ, pcl::Normal>(cloud, normals, 10, 0.05, "normals");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();

  return (viewer);
}

// log_3D (cur_cluster, cur_cluster_num);
void log_3D(CLUSTER_INFO* cluster_info, int cluster_size)
{
  time_t t = time(0);  // get time now
  struct tm* now = localtime(&t);

  char buf[30];
  strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", now);

  stringstream stream;
  stream << fixed << setprecision(2);
  // stream << buf << ",";
  stream << GlobalVariable::ROS_TIMESTAMP << ",";
  for (int i = 0; i < cluster_size; i++)
  {
    if (cluster_info[i].cluster_tag > 1)
    {
      stream << cluster_info[i].obb_vertex[0].x << ",";
      stream << cluster_info[i].obb_vertex[0].y << ",";
      stream << cluster_info[i].obb_vertex[0].z << ",";
      stream << cluster_info[i].obb_vertex[1].x << ",";
      stream << cluster_info[i].obb_vertex[1].y << ",";
      stream << cluster_info[i].obb_vertex[1].z << ",";
      stream << cluster_info[i].obb_vertex[2].x << ",";
      stream << cluster_info[i].obb_vertex[2].y << ",";
      stream << cluster_info[i].obb_vertex[2].z << ",";
      stream << cluster_info[i].obb_vertex[3].x << ",";
      stream << cluster_info[i].obb_vertex[3].y << ",";
      stream << cluster_info[i].obb_vertex[3].z << ",";
      stream << cluster_info[i].obb_vertex[4].x << ",";
      stream << cluster_info[i].obb_vertex[4].y << ",";
      stream << cluster_info[i].obb_vertex[4].z << ",";
      stream << cluster_info[i].obb_vertex[5].x << ",";
      stream << cluster_info[i].obb_vertex[5].y << ",";
      stream << cluster_info[i].obb_vertex[5].z << ",";
      stream << cluster_info[i].obb_vertex[6].x << ",";
      stream << cluster_info[i].obb_vertex[6].y << ",";
      stream << cluster_info[i].obb_vertex[6].z << ",";
      stream << cluster_info[i].obb_vertex[7].x << ",";
      stream << cluster_info[i].obb_vertex[7].y << ",";
      stream << cluster_info[i].obb_vertex[7].z << ",";
      stream << cluster_info[i].cluster_tag << ",";
      stream << cluster_info[i].confidence << ",";
    }
  }
  stream << endl;
  ofstream file("log.txt", std::ios_base::app | std::ios_base::out);

  if (file.is_open())
  {
    file << stream.str();
    file.close();
  }
}

// log_2D (cur_cluster, cur_cluster_num);
void log_2D(CLUSTER_INFO* cluster_info, int cluster_size)
{
  timeval curTime;
  gettimeofday(&curTime, NULL);

  char buf[30] = "";
  strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S-", localtime(&curTime.tv_sec));

  char currentTime[30] = "";
  sprintf(currentTime, "%s%d", buf, (int)(curTime.tv_usec / 1000));

  stringstream stream;
  stream << fixed << setprecision(2);
  for (int i = 0; i < cluster_size; i++)
  {
    if (cluster_info[i].cluster_tag > 0)
    {
      switch (cluster_info[i].cluster_tag)
      {
        case 1:
          stream << -1 << " ";
          break;
        case 2:
          stream << 0 << " ";
          break;
        case 3:
          stream << 3 << " ";
          break;
        case 4:
          stream << 2 << " ";
          break;
        case 5:
          stream << 5 << " ";
          break;
      }

      stream << cluster_info[i].to_2d_PointWL[0].x << ",";
      stream << cluster_info[i].to_2d_PointWL[0].y << ",";
      stream << cluster_info[i].to_2d_PointWL[1].y << ",";
      stream << cluster_info[i].to_2d_PointWL[1].y << ",";
      stream << endl;
    }
  }
  string str(currentTime);
  ofstream file("log/" + str + ".txt", std::ios_base::app | std::ios_base::out);

  if (file.is_open())
  {
    file << stream.str();
    file.close();
  }
}
