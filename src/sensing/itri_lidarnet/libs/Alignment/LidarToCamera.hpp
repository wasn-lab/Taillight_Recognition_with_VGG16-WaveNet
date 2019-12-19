#ifndef LIDARTOCAMERA_HPP_
#define LIDARTOCAMERA_HPP_

#include <pcl/common/common.h> //getMinMax3D
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "../UserDefine.h"

using namespace pcl;

class LidarToCamera
{
  public:
    LidarToCamera ()
    {
      viewer = NULL;
      viewID = NULL;
      image_width = 1280;
      image_height = 720;
    }
    LidarToCamera (boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer,
                   int *input_viewID,
                   float focal,
                   float theta_x,
                   float theta_y,
                   float theta_z,
                   float trans_x,
                   float trans_y,
                   float trans_z,
                   float image_w,
                   float image_h)
    {
      viewer = input_viewer;
      viewID = input_viewID;
      transform = Eigen::MatrixXf (3, 4);
      image_width = image_w;
      image_height = image_h;

      Eigen::Matrix3f foc, rx, ry, rz;
      foc << focal, 0, image_width / 2, 0, -focal, image_height / 2, 0, 0, 1;
      rx << 1, 0, 0, 0, cosf (theta_x), -sinf (theta_x), 0, sinf (theta_x), cosf (theta_x);
      ry << cosf (theta_y), 0, sinf (theta_y), 0, 1, 0, -sinf (theta_y), 0, cosf (theta_y);
      rz << cosf (theta_z), -sinf (theta_z), 0, sinf (theta_z), cosf (theta_z), 0, 0, 0, 1;

      Eigen::Matrix3f rot_tmp = foc * ( (rx * ry) * rz);
      Eigen::Vector3f txyz (trans_x, trans_z, trans_y);

      transform << rot_tmp, txyz;
    }

    void
    update (bool is_debug,
            CLUSTER_INFO* cluster_info,
            int cluster_size)
    {

#pragma omp parallel for
      for (int i = 0; i < cluster_size; ++i)
      {

        Eigen::Vector3f pt[8];
        pt[0] = transform * Eigen::Vector4f (cluster_info[i].min.x, cluster_info[i].min.z, cluster_info[i].min.y, 1);
        pt[1] = transform * Eigen::Vector4f (cluster_info[i].min.x, cluster_info[i].max.z, cluster_info[i].min.y, 1);
        pt[2] = transform * Eigen::Vector4f (cluster_info[i].max.x, cluster_info[i].max.z, cluster_info[i].min.y, 1);
        pt[3] = transform * Eigen::Vector4f (cluster_info[i].max.x, cluster_info[i].min.z, cluster_info[i].min.y, 1);
        pt[4] = transform * Eigen::Vector4f (cluster_info[i].min.x, cluster_info[i].min.z, cluster_info[i].max.y, 1);
        pt[5] = transform * Eigen::Vector4f (cluster_info[i].min.x, cluster_info[i].max.z, cluster_info[i].max.y, 1);
        pt[6] = transform * Eigen::Vector4f (cluster_info[i].max.x, cluster_info[i].max.z, cluster_info[i].max.y, 1);
        pt[7] = transform * Eigen::Vector4f (cluster_info[i].max.x, cluster_info[i].min.z, cluster_info[i].max.y, 1);

        PointCloud<PointXYZ> pt_cloud;

        for (int j = 0; j < 8; ++j)
        {
          if (pt[j] (2) != 0 && cluster_info[i].min.y > 0.2)
          {
            pt_cloud.push_back (PointXYZ (pt[j] (0) / pt[j] (2), pt[j] (1) / pt[j] (2), 0));
          }
        }

        if (pt_cloud.size () > 0)
        {

          PointXYZ pt_min, pt_max;
          pcl::getMinMax3D (pt_cloud, pt_min, pt_max);

          if ( (pt_min.y >= 0 && pt_min.x >= 0 && pt_min.y < image_height && pt_min.x < image_width)
              || (pt_max.y >= 0 && pt_max.x >= 0 && pt_max.y < image_height && pt_max.x < image_width))
          {
            // 3 ___ 2
            //  |   |
            //  |___|
            // 0     1
            cluster_info[i].to_2d_points[0].x = pt_min.y;
            cluster_info[i].to_2d_points[0].y = pt_min.x;
            cluster_info[i].to_2d_points[1].x = pt_max.y;
            cluster_info[i].to_2d_points[1].y = pt_min.x;
            cluster_info[i].to_2d_points[2].x = pt_max.y;
            cluster_info[i].to_2d_points[2].y = pt_max.x;
            cluster_info[i].to_2d_points[3].x = pt_min.y;
            cluster_info[i].to_2d_points[3].y = pt_max.x;

            // h ___
            //  |   |
            //  |___|
            // 0     w
            cluster_info[i].to_2d_PointWL[0].x = pt_min.x;
            cluster_info[i].to_2d_PointWL[0].y = pt_min.y;
            cluster_info[i].to_2d_PointWL[1].x = (pt_max.x - pt_min.x);
            cluster_info[i].to_2d_PointWL[1].y = (pt_min.y - pt_max.y);
          }
          else
          {
            cluster_info[i].to_2d_points[0].x = -1;
            cluster_info[i].to_2d_points[0].y = -1;
            cluster_info[i].to_2d_points[1].x = -1;
            cluster_info[i].to_2d_points[1].y = -1;
            cluster_info[i].to_2d_points[2].x = -1;
            cluster_info[i].to_2d_points[2].y = -1;
            cluster_info[i].to_2d_points[3].x = -1;
            cluster_info[i].to_2d_points[3].y = -1;
            cluster_info[i].to_2d_PointWL[0].x = -1;
            cluster_info[i].to_2d_PointWL[0].y = -1;
            cluster_info[i].to_2d_PointWL[1].x = -1;
            cluster_info[i].to_2d_PointWL[1].y = -1;
          }
        }

      }

      if (is_debug)
      {
        for (int i = 0; i < cluster_size; ++i)
        {
          cout << cluster_info[i].to_2d_PointWL[0].x << endl;
          cout << cluster_info[i].to_2d_PointWL[0].y << endl;
          cout << cluster_info[i].to_2d_PointWL[1].x << endl;
          cout << cluster_info[i].to_2d_PointWL[1].y << endl << endl;

          viewer->addLine (PointXYZ (cluster_info[i].to_2d_points[0].x, cluster_info[i].to_2d_points[0].y, 0),
                           PointXYZ (cluster_info[i].to_2d_points[1].x, cluster_info[i].to_2d_points[1].y, 0), 255, 255, 255, to_string (*viewID));
          viewer->setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, to_string (*viewID));
          ++*viewID;

          viewer->addLine (PointXYZ (cluster_info[i].to_2d_points[1].x, cluster_info[i].to_2d_points[1].y, 0),
                           PointXYZ (cluster_info[i].to_2d_points[2].x, cluster_info[i].to_2d_points[2].y, 0), 255, 255, 255, to_string (*viewID));
          viewer->setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, to_string (*viewID));
          ++*viewID;

          viewer->addLine (PointXYZ (cluster_info[i].to_2d_points[2].x, cluster_info[i].to_2d_points[2].y, 0),
                           PointXYZ (cluster_info[i].to_2d_points[3].x, cluster_info[i].to_2d_points[3].y, 0), 255, 255, 255, to_string (*viewID));
          viewer->setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, to_string (*viewID));
          ++*viewID;

          viewer->addLine (PointXYZ (cluster_info[i].to_2d_points[3].x, cluster_info[i].to_2d_points[3].y, 0),
                           PointXYZ (cluster_info[i].to_2d_points[0].x, cluster_info[i].to_2d_points[0].y, 0), 255, 255, 255, to_string (*viewID));
          viewer->setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, to_string (*viewID));
          ++*viewID;
        }

        cout << transform << endl;

        PointCloud<PointXYZ>::Ptr buff (new PointCloud<PointXYZ>);

        for (int i = 0; i < cluster_size; ++i)
        {
          for (size_t j = 0; j < cluster_info[i].cloud.size (); ++j)
          {
            Eigen::Vector3f buffpoint = transform
                * Eigen::Vector4f (cluster_info[i].cloud.points[j].x, cluster_info[i].cloud.points[j].z, cluster_info[i].cloud.points[j].y, 1);

            buff->push_back (PointXYZ (buffpoint (1) / buffpoint (2), buffpoint (0) / buffpoint (2), 0));
          }

          if (!viewer->updatePointCloud (buff, "dHDL"))
          {
            viewer->addPointCloud (buff, "dHDL");
          }
        }
      }
    }

  private:
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    int *viewID;

    Eigen::MatrixXf transform;
    float image_width;
    float image_height;

};

#endif

