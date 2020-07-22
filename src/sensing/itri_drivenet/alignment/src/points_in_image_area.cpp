#include "points_in_image_area.h"

using namespace DriveNet;

void getPointCloudInImageFOV(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidarall_ptr,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_points_ptr,
                             std::vector<PixelPosition>& cam_pixels, int image_w, int image_h, Alignment& alignment)
{
  // std::cout << "===== getPointCloudInImageFOV... =====" << std::endl;
  /// create variable
  pcl::PointCloud<pcl::PointXYZI> cam_points;
  int cloud_sizes = 0;
  std::vector<std::vector<pcl::PointXYZI>> point_cloud(
      std::vector<std::vector<pcl::PointXYZI>>(image_w, std::vector<pcl::PointXYZI>(image_h)));

  /// copy from source
  pcl::copyPointCloud(*lidarall_ptr, *cams_points_ptr);
  cam_points = *cams_points_ptr;

  /// find 3d points in image coverage
#pragma omp parallel for
  for (size_t i = 0; i < lidarall_ptr->size(); i++)
  {
    PixelPosition pixel_position{ -1, -1 };
    pixel_position = alignment.projectPointToPixel(lidarall_ptr->points[i]);
    if (pixel_position.u >= 0 && pixel_position.v >= 0)
    {
      if (point_cloud[pixel_position.u][pixel_position.v].x > lidarall_ptr->points[i].x ||
          point_cloud[pixel_position.u][pixel_position.v].x == 0)
      {
        point_cloud[pixel_position.u][pixel_position.v] = lidarall_ptr->points[i];
      }
    }
  }
  /// record the 2d points(cam_pixels) and 3d points(cam_points)
  for (int u = 0; u < image_w; u++)
  {
    for (int v = 0; v < image_h; v++)
    {
      PixelPosition pixel_position{ -1, -1 };
      pixel_position.u = u;
      pixel_position.v = v;
      if (point_cloud[u][v].x != 0 && point_cloud[u][v].y != 0 && point_cloud[u][v].z != 0)
      {
        cam_points.points[cloud_sizes] = point_cloud[u][v];      
        cam_pixels.push_back(pixel_position);
        cloud_sizes++;
      }
    }
  }
  /// copy to destination
  cam_points.resize(cloud_sizes);
  *cams_points_ptr = cam_points;
}

void getPointCloudInImageFOV(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidarall_ptr,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_points_ptr,
                             /*std::vector<PixelPosition>& cam_pixels,*/ int image_w, int image_h, Alignment& alignment)
{
  // std::cout << "===== getPointCloudInImageFOV... =====" << std::endl;
  /// create variable
  pcl::PointCloud<pcl::PointXYZI> cam_points;
  int cloud_sizes = 0;
  std::vector<std::vector<pcl::PointXYZI>> point_cloud(
      std::vector<std::vector<pcl::PointXYZI>>(image_w, std::vector<pcl::PointXYZI>(image_h)));

  /// copy from source
  pcl::copyPointCloud(*lidarall_ptr, *cams_points_ptr);
  cam_points = *cams_points_ptr;

  /// find 3d points in image coverage
#pragma omp parallel for
  for (size_t i = 0; i < lidarall_ptr->size(); i++)
  {
    PixelPosition pixel_position{ -1, -1 };
    pixel_position = alignment.projectPointToPixel(lidarall_ptr->points[i]);
    if (pixel_position.u >= 0 && pixel_position.v >= 0)
    {
      if (point_cloud[pixel_position.u][pixel_position.v].x > lidarall_ptr->points[i].x ||
          point_cloud[pixel_position.u][pixel_position.v].x == 0)
      {
        point_cloud[pixel_position.u][pixel_position.v] = lidarall_ptr->points[i];
      }
    }
  }

  /// record the 3d points
#pragma omp parallel for collapse(2)
  for (int u = 0; u < image_w; u++)
  {
    for (int v = 0; v < image_h; v++)
    {
      if (point_cloud[u][v].x != 0 && point_cloud[u][v].y != 0 && point_cloud[u][v].z != 0)
      {
        cam_points.points[cloud_sizes] = point_cloud[u][v];
#pragma omp critical
        {
          cloud_sizes++;
        }
      }
    }
  }
  /// copy to destination
  cam_points.resize(cloud_sizes);
  *cams_points_ptr = cam_points;
}
void getPointCloudInBoxFOV(const msgs::DetectedObjectArray& objects,
                           const pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_points_ptr,
                           pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_bbox_points_ptr,
                           std::vector<PixelPosition>& cam_pixels, msgs::DetectedObjectArray& objects_2d_bbox,
                           std::vector<MinMax3D>& cam_bboxs_cube_min_max,
                           std::vector<pcl::PointCloud<pcl::PointXYZI>>& cam_bboxs_points, Alignment& alignment,
                           CloudCluster& cloud_cluster, bool is_enable_default_3d_bbox, bool do_clustering)
{
  // std::cout << "===== getPointCloudInBoxFOV... =====" << std::endl;
  /// create variable
  pcl::PointCloud<pcl::PointXYZI> cam_points;
  int cloud_sizes = 0;
  pcl::PointCloud<pcl::PointXYZI> point_cloud_src;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<pcl::PointXYZI> point_vector_object;
  std::vector<pcl::PointXYZI> point_vector_objects;

  /// copy from source
  pcl::copyPointCloud(*cams_points_ptr, *cams_bbox_points_ptr);
  cam_points = *cams_bbox_points_ptr;

  // std::cout << "objects.objects size: " << objects.objects.size() << std::endl;
  /// main
  for (const auto& obj : objects.objects)
  {
    MinMax3D cube_min_max;  // object min and max point
    for (const auto& point : cam_points.points)
    {
      // get the 2d box
      std::vector<PixelPosition> bbox_positions(2);
      bbox_positions[0].u = obj.camInfo.u;
      bbox_positions[0].v = obj.camInfo.v;
      bbox_positions[1].u = obj.camInfo.u + obj.camInfo.width;
      bbox_positions[1].v = obj.camInfo.v + obj.camInfo.height;
      transferPixelScaling(bbox_positions);

      // get points in the 2d box
      PixelPosition pixel_position{ -1, -1 };
      pixel_position = alignment.projectPointToPixel(point);
      if (pixel_position.u >= bbox_positions[0].u && pixel_position.v >= bbox_positions[0].v &&
          pixel_position.u <= bbox_positions[1].u && pixel_position.v <= bbox_positions[1].v)
      {
        cam_pixels.push_back(pixel_position);
        point_vector_object.push_back(point);
      }
    }
    // std::cout << "point_vector_object size: " << point_vector_object.size() << std::endl;

    if (!point_vector_object.empty())
    {
      // vector to point cloud
      pcl::PointCloud<pcl::PointXYZI> point_cloud_object;
      point_cloud_object.points.resize(point_vector_object.size());
      for (size_t i = 0; i < point_vector_object.size(); i++)
      {
        point_cloud_object.points[i] = point_vector_object[i];
      }
      point_vector_object.clear();

      // get points in the 3d box
      if (is_enable_default_3d_bbox)
      {
        getPointCloudIn3DBox(point_cloud_object, obj.classId, cloud_filtered_ptr);
      }
      else
      {
        cloud_filtered_ptr = point_cloud_object.makeShared();
      }

      // std::cout << "cloud_filtered_ptr->points size: " << cloud_filtered_ptr->points.size() << std::endl;

      // point cloud to vector
      for (const auto& point : cloud_filtered_ptr->points)
      {
        point_vector_object.push_back(point);
      }

      if (!cloud_filtered_ptr->points.empty())
      {
        if (do_clustering)
        {
          bool do_downsampling = true;  // default is ture
          std::vector<pcl::PointCloud<pcl::PointXYZI>> cluster_points;
          cluster_points = cloud_cluster.getClusters(cloud_filtered_ptr, do_downsampling);
          for (const auto& points : cluster_points)
          {
            pcl::getMinMax3D(points, cube_min_max.p_min, cube_min_max.p_max);
            // if(is_enable_default_3d_bbox)
            // {
            //   object_box bbox{};
            //   bbox = getDefaultObjectBox(obj.classId);
            //   cube_min_max.p_max.x = cube_min_max.p_min.x + bbox.length;
            //   cube_min_max.p_max.y = cube_min_max.p_min.y + bbox.width;
            //   cube_min_max.p_max.z = cube_min_max.p_min.z + bbox.height;
            // }
            cam_bboxs_cube_min_max.push_back(cube_min_max);
            cam_bboxs_points.push_back(points);
          }
        }
        else
        {
          pcl::getMinMax3D(*cloud_filtered_ptr, cube_min_max.p_min, cube_min_max.p_max);
          cam_bboxs_cube_min_max.push_back(cube_min_max);
          cam_bboxs_points.push_back(*cloud_filtered_ptr);
        }
        objects_2d_bbox.objects.push_back(obj);

        // concatenate the points of objects
        point_vector_objects.insert(point_vector_objects.begin(), point_vector_object.begin(),
                                    point_vector_object.end());
        point_vector_object.clear();
      }
    }
  }
  removeDuplePoints(point_vector_objects);
  for (size_t i = 0; i < point_vector_objects.size(); i++)
  {
    cam_points.points[i] = point_vector_objects[i];
    cloud_sizes++;
  }
  point_vector_objects.clear();

  /// copy to destination
  cam_points.resize(cloud_sizes);
  *cams_bbox_points_ptr = cam_points;
}
void getPointCloudInBoxFOV(const msgs::DetectedObjectArray& objects, msgs::DetectedObjectArray& remaining_objects,
                           const pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_points_ptr,
                           pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_bbox_points_ptr,
                           std::vector<PixelPosition>& cam_pixels, msgs::DetectedObjectArray& objects_2d_bbox,
                           std::vector<MinMax3D>& cam_bboxs_cube_min_max,
                           std::vector<pcl::PointCloud<pcl::PointXYZI>>& cam_bboxs_points, Alignment& alignment,
                           CloudCluster& cloud_cluster, bool is_enable_default_3d_bbox, bool do_clustering)
{
  // std::cout << "===== getPointCloudInBoxFOV... =====" << std::endl;
  /// create variable
  pcl::PointCloud<pcl::PointXYZI> cam_points;
  int cloud_sizes = 0;
  pcl::PointCloud<pcl::PointXYZI> point_cloud_src;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<pcl::PointXYZI> point_vector_object;
  std::vector<pcl::PointXYZI> point_vector_objects;

  /// copy from source
  pcl::copyPointCloud(*cams_points_ptr, *cams_bbox_points_ptr);
  cam_points = *cams_bbox_points_ptr;

  // std::cout << "objects.objects size: " << objects.objects.size() << std::endl;
  /// main
  if (!remaining_objects.objects.empty())
  {
    remaining_objects.objects.clear();
  }
  if (!objects_2d_bbox.objects.empty())
  {
    objects_2d_bbox.objects.clear();
  }

  for (const auto& obj : objects.objects)
  {
    MinMax3D cube_min_max;  // object min and max point
    for (const auto& point : cam_points.points)
    {
      // get the 2d box
      std::vector<PixelPosition> bbox_positions(2);
      bbox_positions[0].u = obj.camInfo.u;
      bbox_positions[0].v = obj.camInfo.v;
      bbox_positions[1].u = obj.camInfo.u + obj.camInfo.width;
      bbox_positions[1].v = obj.camInfo.v + obj.camInfo.height;
      transferPixelScaling(bbox_positions);

      // get points in the 2d box
      PixelPosition pixel_position{ -1, -1 };
      pixel_position = alignment.projectPointToPixel(point);
      if (pixel_position.u >= bbox_positions[0].u && pixel_position.v >= bbox_positions[0].v &&
          pixel_position.u <= bbox_positions[1].u && pixel_position.v <= bbox_positions[1].v)
      {
        cam_pixels.push_back(pixel_position);
        point_vector_object.push_back(point);
      }
    }
    // std::cout << "point_vector_object size: " << point_vector_object.size() << std::endl;

    if (!point_vector_object.empty())
    {
      // vector to point cloud
      pcl::PointCloud<pcl::PointXYZI> point_cloud_object;
      point_cloud_object.points.resize(point_vector_object.size());
      for (size_t i = 0; i < point_vector_object.size(); i++)
      {
        point_cloud_object.points[i] = point_vector_object[i];
      }
      point_vector_object.clear();

      // get points in the 3d box
      if (is_enable_default_3d_bbox)
      {
        getPointCloudIn3DBox(point_cloud_object, obj.classId, cloud_filtered_ptr);
      }
      else
      {
        cloud_filtered_ptr = point_cloud_object.makeShared();
      }

      // std::cout << "cloud_filtered_ptr->points size: " << cloud_filtered_ptr->points.size() << std::endl;

      // point cloud to vector
      for (const auto& point : cloud_filtered_ptr->points)
      {
        point_vector_object.push_back(point);
      }

      if (!cloud_filtered_ptr->points.empty())
      {
        if (do_clustering)
        {
          bool do_downsampling = true;  // default is ture
          std::vector<pcl::PointCloud<pcl::PointXYZI>> cluster_points;
          cluster_points = cloud_cluster.getClusters(cloud_filtered_ptr, do_downsampling);
          for (const auto& points : cluster_points)
          {
            pcl::getMinMax3D(points, cube_min_max.p_min, cube_min_max.p_max);
            // if(is_enable_default_3d_bbox)
            // {
            //   object_box bbox{};
            //   bbox = getDefaultObjectBox(obj.classId);
            //   cube_min_max.p_max.x = cube_min_max.p_min.x + bbox.length;
            //   cube_min_max.p_max.y = cube_min_max.p_min.y + bbox.width;
            //   cube_min_max.p_max.z = cube_min_max.p_min.z + bbox.height;
            // }
            cam_bboxs_cube_min_max.push_back(cube_min_max);
            cam_bboxs_points.push_back(points);
          }
        }
        else
        {
          pcl::getMinMax3D(*cloud_filtered_ptr, cube_min_max.p_min, cube_min_max.p_max);
          cam_bboxs_cube_min_max.push_back(cube_min_max);
          cam_bboxs_points.push_back(*cloud_filtered_ptr);
        }
        objects_2d_bbox.objects.push_back(obj);

        // concatenate the points of objects
        point_vector_objects.insert(point_vector_objects.begin(), point_vector_object.begin(),
                                    point_vector_object.end());
        point_vector_object.clear();
      }
      else
      {
        remaining_objects.objects.push_back(obj);
      }
    }
    else
    {
      remaining_objects.objects.push_back(obj);
    }
  }
  // std::cout << "remaining_objects.objects.size(): " << remaining_objects.objects.size() << std::endl;

  removeDuplePoints(point_vector_objects);
  for (size_t i = 0; i < point_vector_objects.size(); i++)
  {
    cam_points.points[i] = point_vector_objects[i];
    cloud_sizes++;
  }
  point_vector_objects.clear();

  /// copy to destination
  cam_points.resize(cloud_sizes);
  *cams_bbox_points_ptr = cam_points;
}

void getPointCloudIn3DBox(const pcl::PointCloud<pcl::PointXYZI>& cloud_src, int object_class_id,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_filtered_ptr)
{
  // std::cout << "===== getPointCloudIn3DBox... =====" << std::endl;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointXYZI min_pt, max_pt;

  /// get the box length of object
  pcl::getMinMax3D(cloud_src, min_pt, max_pt);
  object_box bbox{};
  bbox = getDefaultObjectBox(object_class_id);

  /// build the condition
  pcl::ConditionAnd<pcl::PointXYZI>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZI>());
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZI>("x", pcl::ComparisonOps::GT, min_pt.x)));
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZI>("x", pcl::ComparisonOps::LT, min_pt.x + bbox.length)));

  /// build the filter
  pcl::ConditionalRemoval<pcl::PointXYZI> condrem;
  condrem.setCondition(range_cond);
  cloud_ptr = cloud_src.makeShared();
  condrem.setInputCloud(cloud_ptr);
  condrem.setKeepOrganized(false);

  /// apply filter
  condrem.filter(*cloud_filtered_ptr);
}